"""运行 SG CUDA 对拍及微型渲染验证；首次执行包含扩展编译，由用户手动运行。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sg_wrapper.sg_colors import compute_sg_colors
from sg_wrapper.sg_cuda import compute_sg_colors_cuda


def test_cpu_guard():
    """确认 CPU 输入在编译扩展前被拒绝。"""
    m = torch.zeros(1, 2, 3)
    try:
        compute_sg_colors_cuda(m, m, m[:, :, None], m[:, :, :1, None],
                               m[:, :, None], torch.eye(4)[None, None])
    except ValueError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("CPU input accepted")


def check_cuda():
    """验证多相机、非连续输入、删除位置、空输入和实际 gsplat 图像。

    Returns:
        None: 不满足容差直接抛出异常。
    """
    torch.manual_seed(123)
    device = "cuda"
    for lobes in (0, 1, 3, 6):
        b, v, g = 2, 3, 37
        means = torch.randn(b, g, 3, device=device)
        base = torch.randn_like(means) * .2
        dirs = torch.randn(b, g, lobes, 3, device=device)
        sharp = torch.rand(b, g, lobes, 1, device=device) * 40
        amp = torch.randn_like(dirs) * .1
        cameras = torch.eye(4, device=device).repeat(b, v, 1, 1)
        cameras[:, :, :3, 3] = torch.randn(b, v, 3, device=device)
        means[:, 0] = cameras[:, 0, :3, 3]  # 零长度视线。
        if lobes:
            dirs[:, 1] = 0
            dirs[:, 2] *= 1e-10
            sharp[:, 3] = 0
            amp[:, ::2, -1] = 0  # 删除非前缀 lobe。
        # 转置的相机矩阵覆盖入口 contiguous 行为。
        cameras = cameras.transpose(-1, -2).contiguous().transpose(-1, -2)
        for clamp in (False, True):
            ref = compute_sg_colors(
                means.repeat_interleave(v, 0), base.repeat_interleave(v, 0),
                dirs.repeat_interleave(v, 0), sharp.repeat_interleave(v, 0),
                amp.repeat_interleave(v, 0), cameras.reshape(b*v, 4, 4), clamp,
            )
            with torch.no_grad():
                got = compute_sg_colors_cuda(means, base, dirs, sharp, amp, cameras, clamp)
            torch.testing.assert_close(got, ref, atol=1e-5, rtol=0)
            print(f"L={lobes}, clamp={clamp}: max error={(got-ref).abs().max().item():.3g}")
        means.requires_grad_(True)
        try:
            compute_sg_colors_cuda(means, base, dirs, sharp, amp, cameras)
        except RuntimeError as exc:
            assert "inference-only" in str(exc)
        else:
            raise AssertionError("gradient input accepted")
        with torch.no_grad():
            empty = compute_sg_colors_cuda(
                means[:, :0], base[:, :0], dirs[:, :0], sharp[:, :0], amp[:, :0], cameras,
            )
        assert empty.shape == (b*v, 0, 3)

    # 步骤2：真实光栅化接口对拍；不加载模型或数据集。
    from gsplat import rasterization
    n = 23
    means = torch.randn(1, n, 3, device=device) * .2
    means[..., 2] += 3
    base = torch.randn_like(means) * .1
    dirs = torch.randn(1, n, 3, 3, device=device)
    sharp = torch.rand(1, n, 3, 1, device=device) * 3
    amp = torch.randn_like(dirs) * .1
    cam = torch.eye(4, device=device)[None, None]
    ref = compute_sg_colors(means, base, dirs, sharp, amp, cam[:, 0])
    with torch.no_grad():
        got = compute_sg_colors_cuda(means, base, dirs, sharp, amp, cam)
    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1
    scales = torch.full((n, 3), .1, device=device)
    opacity = torch.full((n,), .7, device=device)
    K = torch.tensor([[[40.,0,16],[0,40,16],[0,0,1]]], device=device)
    images = []
    for colors in (ref, got):
        result = rasterization(means[0], quat, scales, opacity, colors[0],
                               cam[0], K, 32, 32, sh_degree=None,
                               render_mode="RGB+D", packed=False)
        images.append(result[:2])
    for expected, actual in zip(*images):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    print("CUDA color and gsplat RGB/depth/alpha parity passed")


if __name__ == "__main__":
    test_cpu_guard()
    if "--cpu-only" not in sys.argv:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; use --cpu-only for entry validation")
        check_cuda()
