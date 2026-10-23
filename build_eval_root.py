"""Build a reduced RE10K root holding only the chunks an eval index needs.

Every evaluation walks all 543 test chunks (~54 GB) before rendering its first
scene, so a sweep over N settings pays that scan N times. This tool scans
once, symlinks the chunks that actually contain the requested scenes into a
new root, and leaves the source data untouched:

    python build_eval_root.py \
        --index /data14T/xz/eval_index_200.json \
        --src /data14T/xz/data/re10k \
        --dst /data14T/xz/data/re10k_eval200

Then pass ``DATA_ROOT=<dst>`` to ``eval_sg.sh``. The train split is symlinked
wholesale: evaluations never read it, but the eval script checks that the
directory exists.

Exit status is non-zero when a requested scene is missing, so a silently
truncated index cannot turn into a quietly wrong score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: argument list, defaults to ``sys.argv[1:]``.

    Returns:
        Namespace with index/src/dst/stage/overwrite.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True, help="eval index json (scene -> views)")
    p.add_argument("--src", required=True, help="full RE10K root (train/, test/)")
    p.add_argument("--dst", required=True, help="new root to create")
    p.add_argument("--stage", default="test", choices=("test", "train"))
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing symlinks in dst/<stage>",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Scan the source chunks once and build the reduced root.

    Args:
        argv: CLI arguments (see :func:`parse_args`).

    Returns:
        Process exit code: 0 when every requested scene was found, 1 otherwise.
    """
    args = parse_args(argv)
    wanted = set(json.loads(Path(args.index).read_text()))
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_stage = dst_root / args.stage
    dst_stage.mkdir(parents=True, exist_ok=True)

    chunks = sorted((src_root / args.stage).glob("*.torch"))
    if not chunks:
        print(f"no .torch chunks under {src_root / args.stage}", file=sys.stderr)
        return 1

    found: dict[str, int] = {}
    seen: set[str] = set()
    t0 = time.time()
    for i, chunk_path in enumerate(chunks, 1):
        # 只读 key 字段，但 torch.load 仍会反序列化整块（图像在内），
        # 这是本工具唯一一次全量扫描。
        chunk = torch.load(chunk_path, weights_only=True)
        keys = {ex["key"] for ex in chunk if isinstance(ex, dict) and "key" in ex}
        hit = keys & wanted
        if not hit:
            if i % 50 == 0:
                print(f"  [{i}/{len(chunks)}] {time.time() - t0:.0f}s", flush=True)
            continue
        found[chunk_path.name] = len(hit)
        seen |= hit
        link = dst_stage / chunk_path.name
        if link.is_symlink() or link.exists():
            if args.overwrite:
                link.unlink()
            else:
                continue
        os.symlink(chunk_path.resolve(), link)
        print(
            f"  [{i}/{len(chunks)}] {chunk_path.name}: +{len(hit)} scenes "
            f"(total {len(seen)}/{len(wanted)}) {time.time() - t0:.0f}s",
            flush=True,
        )

    # train/ 只做存在性占位，评测阶段不会读它。
    train_link = dst_root / "train"
    if not train_link.exists():
        os.symlink((src_root / "train").resolve(), train_link)

    missing = wanted - seen
    print(
        f"\n{len(found)} chunks -> {dst_stage} for {len(seen)}/{len(wanted)} scenes "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    if missing:
        print(f"missing {len(missing)} scenes, e.g. {sorted(missing)[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
