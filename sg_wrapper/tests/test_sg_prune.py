"""CPU-only checks for post-hoc lobe budgeting (seconds, no repo deps).

Run from the repo root (or anywhere with torch on the path):

    python sg_wrapper/tests/test_sg_prune.py
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sg_wrapper.sg_colors import compute_sg_colors  # noqa: E402
from sg_wrapper.sg_prune import (  # noqa: E402
    BASE_FLOATS,
    LOBE_FLOATS,
    apply_runtime_lobe_budget,
    avg_lobes_for_bytes,
    budget_floats,
    compensate_pruned_lobes,
    compensation_factor,
    compute_packed_sg_colors,
    get_lobe_budget,
    get_packed,
    lobe_budget_mask,
    lobe_importance,
    lobe_stats,
    lobes_for_bytes,
    lobes_to_keep,
    pack_into,
    pack_lobes,
    pack_lobes_to_bytes,
    packed_bytes,
    packed_floats,
    prune_lobes,
    set_lobe_budget,
    set_packed,
    storage_floats,
    storage_report,
)


def test_importance_matches_formula():
    torch.manual_seed(0)
    a = torch.randn(4, 5, 3, 3) * 0.2
    s = torch.rand(4, 5, 3, 1) * 5
    ref = a.abs().mean(dim=-1) * (1 - torch.exp(-2 * s.squeeze(-1)))
    err = (lobe_importance(a, s) - ref).abs().max().item()
    assert err < 1e-7, err
    print("OK  importance matches the MEGS2 range metric")


def test_budget_count_and_topk():
    torch.manual_seed(1)
    D = torch.rand(1000, 3)
    for avg, want in [(0.0, 0), (0.5, 500), (1.0, 1000), (1.5, 1500), (3.0, 3000)]:
        mask = lobe_budget_mask(D, avg)
        got = int(mask.sum())
        assert got == want, (avg, got, want)
        if 0 < got < mask.numel():
            kept = D[mask]
            dropped = D[~mask]
            assert kept.min() >= dropped.max() - 1e-6, "not a top-k selection"
    print("OK  budget counts and top-k property (single scene)")

    # Batched: one budget per scene, not one across the batch.
    Db = torch.rand(2, 50, 3)
    mask = lobe_budget_mask(Db, 2.0)
    assert mask.shape == Db.shape
    assert mask[0].sum().item() == 100 and mask[1].sum().item() == 100
    print("OK  per-scene budgets in a batch")

    assert lobes_to_keep(100, 1.5, 3) == 150
    assert lobes_to_keep(100, 0.0, 3) == 0
    assert lobes_to_keep(100, 3.0, 3) == 300
    print("OK  integer rounding / clamping")


def test_compensation_preserves_sphere_mean():
    torch.manual_seed(2)
    G, L = 200, 3
    c0 = torch.randn(G, 3) * 0.1
    a = torch.randn(G, L, 3) * 0.1
    s = torch.rand(G, L, 1) * 4
    factor = compensation_factor(s)
    mask = lobe_budget_mask(lobe_importance(a, s), 1.0)

    before = c0 + (a * factor).sum(dim=-2)
    new_c0 = compensate_pruned_lobes(c0, a, s, mask)
    after = new_c0 + (a * mask.unsqueeze(-1) * factor).sum(dim=-2)
    err = (before - after).abs().max().item()
    assert err < 1e-5, f"sphere mean changed by {err}"
    print(f"OK  compensation preserves analytic sphere mean (err {err:.2e})")

    # Monte Carlo cross-check of the closed form for one lobe.
    torch.manual_seed(3)
    v = torch.randn(200000, 3)
    v = v / v.norm(dim=-1, keepdim=True)
    mu = torch.tensor([0.3, -0.5, 0.8])
    mu = mu / mu.norm()
    for sv in (0.0, 0.5, 2.0, 10.0):
        mc = torch.exp(sv * (v @ mu - 1.0)).mean().item()
        exact = compensation_factor(torch.tensor([[[sv]]])).item()
        assert abs(mc - exact) < 0.01, (sv, mc, exact)
    print("OK  closed-form factor matches Monte Carlo sphere average")


def test_edge_cases():
    torch.manual_seed(4)
    G, L = 7, 3
    c0 = torch.randn(G, 3)
    a = torch.randn(G, L, 3) * 0.1
    s = torch.rand(G, L, 1) * 3
    factor = compensation_factor(s)

    # avg = 0: every lobe is removed, all energy moves to c0.
    mask0 = lobe_budget_mask(lobe_importance(a, s), 0.0)
    assert not mask0.any()
    new_c0 = compensate_pruned_lobes(c0, a, s, mask0)
    err = (new_c0 - (c0 + (a * factor).sum(dim=-2))).abs().max().item()
    assert err < 1e-6, err
    print("OK  avg_lobes=0 moves all lobe energy into c0")

    # avg = L: nothing is removed.
    maskL = lobe_budget_mask(lobe_importance(a, s), float(L))
    assert maskL.all()
    new_c0 = compensate_pruned_lobes(c0, a, s, maskL)
    err = (new_c0 - c0).abs().max().item()
    assert err < 1e-7, err
    print("OK  avg_lobes=L is a no-op")

    # Zero sharpness is direction-constant; the factor must be exactly 1.
    z = torch.zeros(G, L, 1)
    f = compensation_factor(z)
    assert torch.isfinite(f).all() and (f - 1).abs().max() < 1e-6
    tiny = torch.full((G, L, 1), 1e-7)
    ft = compensation_factor(tiny)
    assert torch.isfinite(ft).all()
    print("OK  zero / tiny sharpness stays finite and equals 1 at s=0")


def test_prune_container_and_storage():
    torch.manual_seed(5)
    G, L = 40, 3
    g = types.SimpleNamespace(
        rgb_base=torch.randn(G, 3),
        sg_rgb=torch.randn(G, L, 3) * 0.1,
        sg_sharpness=torch.rand(G, L, 1) * 4,
    )
    base_copy = g.rgb_base.clone()
    amp_copy = g.sg_rgb.clone()

    out, stats = prune_lobes(g, avg_lobes=1.0)
    assert stats["n_points"] == G
    assert stats["kept_lobes"] == G
    assert abs(stats["avg_lobes"] - 1.0) < 1e-6
    assert sum(stats["histogram"]) == G
    weighted = sum(d * n for d, n in enumerate(stats["histogram"]))
    assert weighted == G

    # Non-inplace: the input is untouched, the output really changed.
    assert torch.equal(g.rgb_base, base_copy) and torch.equal(g.sg_rgb, amp_copy)
    assert not torch.equal(out.rgb_base, base_copy)
    assert (out.sg_rgb == 0).any() and out.sg_rgb.shape == amp_copy.shape

    # Storage accounting follows the kept count, not the padded shape.
    assert storage_floats(100, 150) == BASE_FLOATS * 100 + LOBE_FLOATS * 150
    assert budget_floats(100, 1.5) == storage_floats(100, 150)
    rep = storage_report(out, (out.sg_rgb.abs().sum(-1) > 0))
    assert rep["n_points"] == G and rep["kept_lobes"] == G
    assert rep["floats"] == BASE_FLOATS * G + LOBE_FLOATS * G
    assert rep["bytes"] == rep["floats"] * 4
    print("OK  prune_lobes, histogram and storage accounting")


def test_packed_matches_padded():
    torch.manual_seed(7)
    G, L = 40, 3
    means = torch.randn(1, G, 3)
    c0 = torch.randn(1, G, 3) * 0.1
    dirs = torch.randn(G, L, 3)
    sharp = torch.rand(G, L, 1) * 4
    rgb = torch.randn(G, L, 3) * 0.1
    extr = torch.eye(4).unsqueeze(0)
    extr[0, :3, 3] = torch.tensor([2.0, -1.0, 3.0])
    g = types.SimpleNamespace(
        rgb_base=c0[0], sg_directions=dirs, sg_sharpness=sharp, sg_rgb=rgb
    )

    for avg in (3.0, 2.0, 1.0, 0.0):
        pruned, stats = prune_lobes(g, avg)
        padded = compute_sg_colors(
            means,
            pruned.rgb_base.unsqueeze(0),
            pruned.sg_directions.unsqueeze(0),
            pruned.sg_sharpness.unsqueeze(0),
            pruned.sg_rgb.unsqueeze(0),
            extr,
        )
        packed = pack_lobes(g, avg_lobes=avg)
        got = compute_packed_sg_colors(
            means,
            packed.rgb_base.unsqueeze(0),
            packed.directions,
            packed.sharpness,
            packed.rgb,
            packed.point_index,
            extr,
        )
        err = (padded - got).abs().max().item()
        assert err < 1e-6, (avg, err)
        assert packed.n_lobes == stats["kept_lobes"], (avg, packed.n_lobes, stats)
        assert int(packed.counts.sum()) == packed.n_lobes
        assert packed.counts.max().item() <= L

    # Decoder path uses a leading batch of 1 ([1, G, L, ...]).
    g_batched = types.SimpleNamespace(
        rgb_base=c0,
        sg_directions=dirs.unsqueeze(0),
        sg_sharpness=sharp.unsqueeze(0),
        sg_rgb=rgb.unsqueeze(0),
    )
    packed_b = pack_lobes(g_batched, avg_lobes=1.0)
    assert packed_b.n_lobes == G
    out_b, _ = pack_into(g_batched, 1.0)
    assert out_b.rgb_base.shape == (1, G, 3)
    assert out_b.sg_packed_index.numel() == G
    print("OK  packed colors match padded pruned colors (L=3/2/1/0)")


def test_packed_byte_budget():
    torch.manual_seed(8)
    G, L = 100, 3
    c0 = torch.randn(G, 3) * 0.1
    g = types.SimpleNamespace(
        rgb_base=c0,
        sg_directions=torch.randn(G, L, 3),
        sg_sharpness=torch.rand(G, L, 1) * 4,
        sg_rgb=torch.randn(G, L, 3) * 0.1,
    )
    target = (BASE_FLOATS * G + LOBE_FLOATS * 123) * 4
    assert lobes_for_bytes(G, target, L) == 123
    assert abs(avg_lobes_for_bytes(G, target, L) - 1.23) < 1e-9

    packed = pack_lobes_to_bytes(g, target)
    assert packed.n_lobes == 123
    assert packed_floats(packed) == storage_floats(G, 123)
    assert packed_bytes(packed) <= target

    # Tight budget: fewer bytes forces fewer lobes, but never a negative count.
    tiny = pack_lobes_to_bytes(g, 4 * BASE_FLOATS * G)
    assert tiny.n_lobes == 0
    assert packed_bytes(tiny) == 4 * BASE_FLOATS * G
    print("OK  byte budget -> exact lobe count and packed storage")


def test_packed_runtime_hook():
    torch.manual_seed(9)
    G, L = 12, 3
    g = types.SimpleNamespace(
        rgb_base=torch.randn(G, 3),
        sg_directions=torch.randn(G, L, 3),
        sg_sharpness=torch.rand(G, L, 1) * 4,
        sg_rgb=torch.randn(G, L, 3) * 0.1,
        clamp_colors=False,
    )
    os.environ.pop("SG_LOBE_AVG", None)
    os.environ.pop("SG_PACKED", None)
    set_lobe_budget(None)
    set_packed(False)
    assert not get_packed()
    assert apply_runtime_lobe_budget(g) is g

    set_lobe_budget(1.0)
    set_packed(True)
    assert get_packed()
    out = apply_runtime_lobe_budget(g)
    assert out.sg_directions is None and out.sg_sharpness is None and out.sg_rgb is None
    assert out.sg_packed_directions.shape[0] == G
    assert out.sg_packed_index.numel() == G

    set_lobe_budget(None)
    set_packed(True)
    os.environ["SG_BUDGET_MB"] = "0.0008"  # ~839 B for N=12 -> 6 lobes
    out = apply_runtime_lobe_budget(g)
    assert out.sg_packed_index.numel() == 6, out.sg_packed_index.numel()
    del os.environ["SG_BUDGET_MB"]
    set_packed(False)
    print("OK  runtime packed hook frees padded tensors at L=1 and honors SG_BUDGET_MB")


def test_runtime_budget_hook():
    torch.manual_seed(6)
    G, L = 12, 3
    g = types.SimpleNamespace(
        rgb_base=torch.randn(G, 3),
        sg_rgb=torch.randn(G, L, 3) * 0.1,
        sg_sharpness=torch.rand(G, L, 1) * 4,
    )
    os.environ.pop("SG_LOBE_AVG", None)
    set_lobe_budget(None)
    assert get_lobe_budget() is None
    assert apply_runtime_lobe_budget(g) is g

    set_lobe_budget(1.0)
    out = apply_runtime_lobe_budget(g)
    assert out is not g and out.sg_rgb.shape == g.sg_rgb.shape
    kept = (out.sg_rgb.abs().sum(-1) > 0).sum().item()
    assert kept == G, kept

    set_lobe_budget(None)
    os.environ["SG_LOBE_AVG"] = "0"
    out = apply_runtime_lobe_budget(g)
    assert (out.sg_rgb == 0).all()
    del os.environ["SG_LOBE_AVG"]
    print("OK  runtime budget hook (setter + SG_LOBE_AVG), unset is a no-op")


def main():
    test_importance_matches_formula()
    test_budget_count_and_topk()
    test_compensation_preserves_sphere_mean()
    test_edge_cases()
    test_prune_container_and_storage()
    test_packed_matches_padded()
    test_packed_byte_budget()
    test_packed_runtime_hook()
    test_runtime_budget_hook()
    print("all sg_prune tests passed")


if __name__ == "__main__":
    main()
