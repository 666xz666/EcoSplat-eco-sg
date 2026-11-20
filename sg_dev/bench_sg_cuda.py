"""生产形状下 sg_color torch vs cuda 微基准（no_grad，隔离姿态对齐稀释）。

用法（仓库根，ecosplat 环境，单卡）：
    TORCH_EXTENSIONS_DIR=/data14T/xz/.cache/torch_extensions \
    CUDA_VISIBLE_DEVICES=<UUID> python sg_dev/bench_sg_cuda.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg_wrapper" / "sg_wrapper"))

import torch
from sg_wrapper.sg_colors import compute_sg_colors
from sg_wrapper.sg_cuda import compute_sg_colors_cuda


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    dev = "cuda"
    torch.manual_seed(0)
    # 生产形状：B=1 场景，V=20（16 context + 4 target），G=65536 点，L=3 lobe。
    for (b, v, g, L) in [(1, 20, 65536, 3), (1, 20, 14117, 3)]:
        means = torch.randn(b, g, 3, device=dev)
        base = torch.rand(b, g, 3, device=dev) * 0.6 + 0.2
        dirs = torch.randn(b, g, L, 3, device=dev)
        sharp = torch.rand(b, g, L, 1, device=dev) * 30
        amp = torch.randn(b, g, L, 3, device=dev) * 0.1
        cam = torch.eye(4, device=dev).repeat(b, v, 1, 1)
        cam[:, :, :3, 3] = torch.randn(b, v, 3, device=dev)

        # torch 参考路径需要沿 V 复制输入。
        m2 = means.repeat_interleave(v, 0)
        b2 = base.repeat_interleave(v, 0)
        d2 = dirs.repeat_interleave(v, 0)
        s2 = sharp.repeat_interleave(v, 0)
        a2 = amp.repeat_interleave(v, 0)
        c2 = cam.reshape(b * v, 4, 4)

        with torch.no_grad():
            t_torch = _time(lambda: compute_sg_colors(m2, b2, d2, s2, a2, c2))
            t_cuda = _time(lambda: compute_sg_colors_cuda(means, base, dirs, sharp, amp, cam))
            ref = compute_sg_colors(m2, b2, d2, s2, a2, c2)
            got = compute_sg_colors_cuda(means, base, dirs, sharp, amp, cam)
        err = (got - ref).abs().max().item()
        print(f"B{b} V{v} G{g} L{L}: torch={t_torch:.2f}ms cuda={t_cuda:.2f}ms "
              f"speedup={t_torch/t_cuda:.1f}x maxerr={err:.2e}", flush=True)


if __name__ == "__main__":
    main()
