"""将单条 Freiburg 3 MCAP 序列解码为带原始时间戳的 TUM 输入。"""

import argparse
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from prepare_tum import KINECT_K


TOPICS = ("/rgb", "/depth", "/ground-truth", "/rgb-calibration", "/depth-calibration")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DIMENSIONS = (640, 480)


def decoder_for(schema):
    """使用记录内嵌的 FileDescriptorSet 构造对应 protobuf 类。"""
    if schema.encoding != "protobuf":
        raise ValueError(f"Unexpected schema encoding: {schema.encoding}")
    pool = descriptor_pool.DescriptorPool()
    for item in descriptor_pb2.FileDescriptorSet.FromString(schema.data).file:
        pool.Add(item)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))


def timestamp(message, log_time):
    """验证两处时间记录一致，并保留纳秒精度的十进制字符串。"""
    seconds = message.timestamp.seconds
    nanos = message.timestamp.nanos
    if seconds * 1_000_000_000 + nanos != log_time:
        raise ValueError("TUM timestamp and MCAP log time disagree")
    return f"{seconds}.{nanos:09d}"


def validate_calibration(message):
    """校验 Freiburg 3 的已去畸变传感器内参与配准尺寸。"""
    if (message.width, message.height) != DIMENSIONS:
        raise ValueError("Unexpected TUM frame size")
    if not np.allclose(message.K, KINECT_K.ravel(), atol=1e-5, rtol=0) or np.any(message.D):
        raise ValueError("MCAP calibration is incompatible with undistorted Freiburg 3 frames")


def write_depth(path, message):
    """保留原始 16 位 PNG 字节及无效深度标记。"""
    data = message.data
    if message.format != "png" or not data.startswith(PNG_SIGNATURE):
        raise ValueError("Expected native TUM depth PNG")
    header = struct.unpack(">IIBB", data[16:26])
    if header != (*DIMENSIONS, 16, 0):
        raise ValueError("Expected 640x480 grayscale uint16 depth")
    path.write_bytes(data)


def unpack_records(mcap, scene, video):
    """抽取视频码流、深度、位姿及各通道的原始时间索引。"""
    from mcap.reader import make_reader

    rgb_index, depth_index, poses = [], [], []
    decoders = {}
    last_time = {}
    calibration = set()
    with mcap.open("rb") as source, video.open("wb") as h264:
        reader = make_reader(source, validate_crcs=True)
        summary = reader.get_summary()
        for schema, channel, record in reader.iter_messages(topics=TOPICS):
            if schema.id not in decoders:
                decoders[schema.id] = decoder_for(schema)
            message = decoders[schema.id].FromString(record.data)
            topic = channel.topic
            stamp = timestamp(message, record.log_time)
            if record.log_time <= last_time.get(topic, -1):
                raise ValueError(f"Non-increasing timestamps in {topic}")
            last_time[topic] = record.log_time
            if topic == "/rgb":
                if message.format != "h264":
                    raise ValueError("Expected Annex-B H.264")
                h264.write(message.data)
                rgb_index.append(f"{stamp} rgb/{stamp}.png")
            elif topic == "/depth":
                relative = f"depth/{stamp}.png"
                write_depth(scene / relative, message)
                depth_index.append(f"{stamp} {relative}")
            elif topic == "/ground-truth":
                if message.frame_id != "world":
                    raise ValueError("Motion-capture poses must be in world coordinates")
                pos = message.pose.position
                rot = message.pose.orientation
                poses.append(f"{stamp} " + " ".join(map(repr, (pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w))))
            else:
                validate_calibration(message)
                calibration.add(topic)
        expected = {channel.topic: summary.statistics.channel_message_counts[channel.id]
                    for channel in summary.channels.values() if channel.topic in TOPICS}
    observed = {"/rgb": len(rgb_index), "/depth": len(depth_index), "/ground-truth": len(poses),
                **{topic: 1 for topic in calibration}}
    if len(observed) != len(TOPICS) or expected != observed:
        raise ValueError(f"MCAP frame count mismatch: {observed} vs {expected}")
    for name, lines in (("rgb", rgb_index), ("depth", depth_index), ("groundtruth", poses)):
        (scene / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MCAP verified: {len(rgb_index)} RGB, {len(depth_index)} depth, {len(poses)} poses", flush=True)
    return rgb_index


def decode_rgb(video, scene, rgb_index):
    """将无 B 帧的码流按原时间戳顺序写成无损 PNG。"""
    stream = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        if not stream.isOpened():
            raise RuntimeError("Could not open h264 video")
        for i, line in enumerate(rgb_index, 1):
            ok, image = stream.read()
            if not ok or image.shape[1::-1] != DIMENSIONS:
                raise ValueError(f"Invalid decoded RGB frame {i}")
            path = scene / line.split()[1]
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not write {path}")
            if i % 200 == 0:
                print(f"Decoded {i}/{len(rgb_index)} RGB frames", flush=True)
        if stream.read()[0]:
            raise ValueError("More decoded RGB frames than timestamps")
    finally:
        stream.release()


def decode(source, destination):
    """成功验证所有通道后发布与 TUM 原始目录结构兼容的输入。"""
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent) as scratch:
        scene = Path(scratch) / "scene"
        (scene / "rgb").mkdir(parents=True)
        (scene / "depth").mkdir()
        video = Path(scratch) / "rgb.h264"
        rgb_index = unpack_records(source, scene, video)
        decode_rgb(video, scene, rgb_index)
        scene.rename(destination)
    print(f"Published decoded sequence: {destination}", flush=True)


def main():
    """从命令行解包一个 TUM MCAP 文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    decode(args.source, args.destination)


if __name__ == "__main__":
    main()
