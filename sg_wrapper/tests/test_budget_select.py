"""CPU checks for sg_wrapper.budget_select.select_by_budget."""
import torch

from sg_wrapper.budget import dense_floats
from sg_wrapper.budget_select import select_by_budget


def _case(P=64, L=3, V=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    opacity = torch.rand(1, V, P, generator=g)
    value = torch.rand(1, V, P, L, generator=g)
    return opacity, value


def test_shapes_and_padding():
    o, d = _case()
    sel = select_by_budget(o, d, dense_floats(1, 8, 8, 3) // 4)
    assert sel.index.shape[:2] == (1, 2)
    assert sel.index.shape[2] == sel.valid.shape[2] == sel.keep_lobes.shape[2]
    assert sel.keep_lobes.shape[-1] == 3
    assert not torch.any(sel.keep_lobes[~sel.valid])
    for vi in range(2):
        n = int(sel.valid[0, vi].sum())
        assert 0 < n <= 64


def test_budget_accounting():
    o, d = _case()
    P = o.shape[-1]
    cap = 14 * P // 3  # roughly a third of the dense payload
    sel = select_by_budget(o, d, cap)
    target = int(round(sel.target))  # cap is snapped down to the 7-float grid
    for vi in range(o.shape[1]):
        n = int(sel.valid[0, vi].sum())
        lobes = int(sel.keep_lobes[0, vi].sum())
        achieved = 14 * n + 7 * lobes
        assert achieved <= target, (achieved, target)
        assert (target - achieved) % 7 == 0


def test_lobe_mask_keeps_original_order():
    # 只有最后一颗 lobe 值钱：分配器内部按 D 降序会把它排到第一列，mask 必须
    # 散射回原始的第 L-1 列，否则会剪错 lobe。
    P, L, V = 16, 3, 1
    opacity = torch.full((1, V, P), 0.9)
    value = torch.zeros((1, V, P, L))
    value[..., 2] = 10.0
    sel = select_by_budget(opacity, value, 14 * P + 7 * P, lobes_per_point=L)
    assert torch.all(sel.keep_lobes[..., 2])
    assert not torch.any(sel.keep_lobes[..., :2])


def test_full_budget_keeps_everything():
    P, L, V = 32, 3, 1
    o, d = _case(P=P, L=L, V=V)
    sel = select_by_budget(o, d, dense_floats(1, 1, P, L), lobes_per_point=L)
    assert int(sel.valid.sum()) == P
    assert torch.all(sel.keep_lobes)


def test_deterministic():
    o, d = _case()
    a = select_by_budget(o, d, 4096)
    b = select_by_budget(o, d, 4096)
    assert torch.equal(a.index, b.index)
    assert torch.equal(a.keep_lobes, b.keep_lobes)


def test_padding_slots_are_invisible():
    o, d = _case(V=3)
    sel = select_by_budget(o, d, dense_floats(1, 8, 8, 3) // 8)
    # 补位槽位必须全 invalid、且不带任何 lobe，调用方才能安全把密度置 0。
    for vi in range(3):
        n = int(sel.valid[0, vi].sum())
        assert not torch.any(sel.valid[0, vi, n:])
        assert not torch.any(sel.keep_lobes[0, vi, n:])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all checks passed")
