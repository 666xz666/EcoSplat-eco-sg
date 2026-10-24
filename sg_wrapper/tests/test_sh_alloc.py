"""CPU-only checks for per-point SH degree allocation (seconds, no repo deps).

Run from the repo root (or anywhere with torch on the path):

    python sg_wrapper/tests/test_sh_alloc.py
"""

import dataclasses
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sg_wrapper.sh_prune import (  # noqa: E402
    allocate_degrees,
    allocation_floats,
    apply_runtime_sh_budget,
    apply_sh_allocation,
    band_energy,
    sh_num_coeffs,
    truncate_sh,
)


@dataclasses.dataclass
class FakeGaussians:
    """Mirror of src.model.types.Gaussians, kept local so no host import runs."""

    means: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor


def make_gaussians(G=4, degree=2, seed=0):
    torch.manual_seed(seed)
    return FakeGaussians(
        means=torch.randn(G, 3),
        harmonics=torch.randn(G, 3, sh_num_coeffs(degree)),
        opacities=torch.rand(G),
    )


def test_band_energy():
    # 单点 degree=1：band1 只在 R 通道放一个非零系数，能量应为 |v|。
    h = torch.zeros(1, 3, 4)
    h[0, 0, 1] = 2.0
    e = band_energy(h)
    assert e.shape == (1, 1) and abs(e.item() - 2.0) < 1e-6
    # 多 band：各 band 能量独立。
    h = torch.zeros(1, 3, 9)
    h[0, 0, 1:4] = 1.0
    h[0, 1, 4:9] = 3.0
    e = band_energy(h)
    assert abs(e[0, 0].item() - (3.0 ** 0.5)) < 1e-6
    assert abs(e[0, 1].item() - 9.0) < 1e-6
    print("OK  band_energy per-band L2")


def test_allocation_prefers_high_energy_points():
    # 3 个点、degree=2；预算恰好够 1 个点保满阶、其余降到 0。
    g = make_gaussians(G=3, degree=2)
    h = torch.zeros(3, 3, 9)
    h[:, :, 0] = 0.5          # DC 无关
    h[0, 0, 4:9] = 5.0        # 点 0 的 band2 能量最大
    h[1, 0, 1:4] = 1.0        # 点 1 只有 band1
    g.harmonics = h
    # 总色彩浮点 = 27 + 3 + 3 = 33 → 均值 11
    degrees = allocate_degrees(h, 33 / 3)
    assert degrees.tolist() == [2, 0, 0], degrees.tolist()
    assert allocation_floats(degrees) <= 11 + 1e-9
    print("OK  allocation gives the full degree to the highest-energy point")


def test_budget_never_exceeded():
    g = make_gaussians(G=64, degree=4)
    for budget in (3, 5, 9, 12, 17, 24, 40):
        degrees = allocate_degrees(g.harmonics, budget)
        avg = allocation_floats(degrees)
        # floor 语义允许 < 一次升级的松弛
        assert avg <= budget + 1e-9, (budget, avg)
        assert avg >= 3.0
    print("OK  budget respected across the range (floor semantics)")


def test_zeroing_equals_truncation():
    g = make_gaussians(G=8, degree=4, seed=3)
    out, degrees = apply_sh_allocation(g, 17)
    assert out.harmonics.shape == g.harmonics.shape
    for i in range(8):
        want = truncate_sh(
            FakeGaussians(g.means[i : i + 1], g.harmonics[i : i + 1], g.opacities[i : i + 1]),
            int(degrees[i]),
        ).harmonics
        got = out.harmonics[i]
        assert torch.equal(got, want[0]), i
    assert torch.equal(out.means, g.means) and torch.equal(g.harmonics[0, :, :1], out.harmonics[0, :, :1])
    print("OK  per-point zeroing matches per-point truncation; DC untouched")


def test_uniform_budget_degrades_to_uniform():
    # 预算正好等于 degree 1 的均匀存储（12 floats）→ 所有点 degree 1。
    g = make_gaussians(G=5, degree=2)
    degrees = allocate_degrees(g.harmonics, 12)
    assert degrees.tolist() == [1] * 5, degrees.tolist()
    print("OK  uniform budget collapses to uniform truncation")


def test_runtime_hook_precedence():
    g = make_gaussians(G=4, degree=4)
    os.environ.pop("SH_KEEP_DEGREE", None)
    os.environ.pop("SH_ALLOC_COLOR_FLOATS", None)
    from sg_wrapper.sh_prune import set_sh_alloc, set_sh_keep_degree

    set_sh_alloc(None); set_sh_keep_degree(None)
    assert apply_runtime_sh_budget(g) is g

    # 同时设置时 allocation 优先
    os.environ["SH_KEEP_DEGREE"] = "2"
    os.environ["SH_ALLOC_COLOR_FLOATS"] = "12"
    out = apply_runtime_sh_budget(g)
    assert out is not g
    degrees = allocate_degrees(g.harmonics, 12)
    assert torch.equal(
        out.harmonics != 0,
        (torch.arange(25) < (degrees + 1) ** 2).unsqueeze(-1).expand(4, 3, 25),
    )
    del os.environ["SH_KEEP_DEGREE"], os.environ["SH_ALLOC_COLOR_FLOATS"]

    # 只设 SH_ALLOC_COLOR_FLOATS 也生效
    os.environ["SH_ALLOC_COLOR_FLOATS"] = "40"
    out2 = apply_runtime_sh_budget(g)
    assert out2 is not g and out2.harmonics.shape == g.harmonics.shape
    del os.environ["SH_ALLOC_COLOR_FLOATS"]

    # 非法值（<3）必须在读取时报错，而不是静默通过
    os.environ["SH_ALLOC_COLOR_FLOATS"] = "2"
    try:
        apply_runtime_sh_budget(g)
    except ValueError:
        pass
    else:
        raise AssertionError("budget < 3 should raise")
    del os.environ["SH_ALLOC_COLOR_FLOATS"]
    print("OK  runtime hook: precedence, env, and guard")


def main():
    test_band_energy()
    test_allocation_prefers_high_energy_points()
    test_budget_never_exceeded()
    test_zeroing_equals_truncation()
    test_uniform_budget_degrades_to_uniform()
    test_runtime_hook_precedence()
    print("all sh_alloc tests passed")


if __name__ == "__main__":
    main()
