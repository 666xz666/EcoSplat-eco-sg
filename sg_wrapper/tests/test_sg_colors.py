"""CPU-only sanity checks for sg_wrapper (seconds to run, no repo deps).

Run from the repo root (or anywhere with torch on the path):

    python sg_wrapper/tests/test_sg_colors.py
"""

import os
import sys

import torch

# Import the package without installing it: add sg_wrapper/ (the directory
# containing the sg_wrapper/ package) to the path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sg_wrapper import C0, compute_sg_colors  # noqa: E402


def reference_colors(means, rgb_base, dirs, sharp, rgb, extrinsics):
    """Direct transcription of MEGS2 compute_colors_precomp (training branch)."""
    b, g, l, _ = dirs.shape
    colors = rgb_base.clone()
    campos = extrinsics[:, :3, 3]
    for bi in range(b):
        view = campos[bi][None, :] - means[bi]
        view = view / (view.norm(dim=1, keepdim=True) + 1e-8)
        for gi in range(g):
            for li in range(l):
                d = torch.nn.functional.normalize(dirs[bi, gi, li], dim=0)
                cos = torch.dot(d, view[gi])
                scale = torch.exp(abs(sharp[bi, gi, li]) * (cos - 1.0))
                colors[bi, gi] += rgb[bi, gi, li] * scale
    return colors * C0 + 0.5


def main():
    torch.manual_seed(0)
    b, g, l = 3, 17, 3
    means = torch.randn(b, g, 3)
    rgb_base = torch.randn(b, g, 3) * 0.1
    dirs = torch.randn(b, g, l, 3)
    sharp = (torch.rand(b, g, l, 1) * 8).requires_grad_(True)
    rgb = (torch.randn(b, g, l, 3) * 0.1).requires_grad_(True)
    extrinsics = torch.eye(4).repeat(b, 1, 1)
    extrinsics[:, :3, 3] = torch.randn(b, 3) * 4

    got = compute_sg_colors(means, rgb_base, dirs, sharp, rgb, extrinsics)
    ref = reference_colors(means, rgb_base, dirs, sharp, rgb, extrinsics)
    err = (got - ref).abs().max().item()
    assert err < 1e-5, f"mismatch vs reference: max abs err {err}"
    print(f"OK  matches MEGS2 reference (max abs err {err:.2e})")

    # Gradients reach the lobe parameters.
    loss = got.square().mean()
    loss.backward()
    assert sharp.grad is not None and sharp.grad.abs().sum() > 0
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    print("OK  gradients flow to sharpness and rgb")

    # Degenerate checks: zero sharpness makes every lobe direction-constant
    # (infinitely wide), so the color reduces to diffuse + sum(amplitudes);
    # a lobe vanishes via zero amplitude, not via zero sharpness.
    sharp_zero = torch.zeros(b, g, l, 1)
    c = compute_sg_colors(means, rgb_base, dirs, sharp_zero, rgb, extrinsics)
    wide = (rgb_base + rgb.sum(dim=2)) * C0 + 0.5
    err = (c - wide).abs().max().item()
    assert err < 1e-6, f"zero-sharpness should be direction-constant, err {err}"
    print("OK  zero sharpness reduces to diffuse + summed amplitudes")

    # Zero amplitudes reduce to the pure diffuse term.
    c = compute_sg_colors(
        means, rgb_base, dirs, torch.rand(b, g, l, 1) * 8,
        torch.zeros(b, g, l, 3), extrinsics,
    )
    err = (c - (rgb_base * C0 + 0.5)).abs().max().item()
    assert err < 1e-6, f"zero amplitudes should reduce to diffuse, err {err}"
    print("OK  zero amplitudes reduce to diffuse term")

    # clamp_colors matches MEGS2's training-time clamp.
    big = torch.full((1, 1, 3), 10.0)
    c = compute_sg_colors(
        torch.zeros(1, 1, 3), big, dirs[:1, :1], torch.zeros(1, 1, l, 1),
        torch.zeros(1, 1, l, 3), torch.eye(4)[None], clamp_colors=True,
    )
    assert (c <= 1.0).all() and (c >= 0.0).all()
    print("OK  clamp_colors clamps to [0, 1]")

    print("all sg_wrapper color tests passed")


if __name__ == "__main__":
    main()
