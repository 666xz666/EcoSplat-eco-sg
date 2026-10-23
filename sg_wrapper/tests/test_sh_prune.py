"""CPU-only checks for post-hoc SH band truncation (seconds, no repo deps).

Run from the repo root (or anywhere with torch on the path):

    python sg_wrapper/tests/test_sh_prune.py
"""

import dataclasses
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sg_wrapper.sh_prune import (  # noqa: E402
    SH_BASE_FLOATS,
    apply_runtime_sh_budget,
    degree_of,
    get_sh_keep_degree,
    lobe_equivalent,
    set_sh_keep_degree,
    sh_num_coeffs,
    sh_storage_floats,
    truncate_sh,
)


@dataclasses.dataclass
class FakeGaussians:
    """Mirror of src.model.types.Gaussians, kept local so no host import runs."""

    means: torch.Tensor
    covariances: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor


def make_gaussians(G=5, degree=4, seed=0):
    torch.manual_seed(seed)
    d_sh = sh_num_coeffs(degree)
    return FakeGaussians(
        means=torch.randn(G, 3),
        covariances=torch.eye(3).expand(G, 3, 3).clone(),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(G, 4).clone(),
        scales=torch.rand(G, 3) + 0.1,
        harmonics=torch.randn(G, 3, d_sh),
        opacities=torch.rand(G),
    )


def test_degree_and_storage():
    for d, n in [(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)]:
        assert sh_num_coeffs(d) == n
        assert degree_of(torch.zeros(2, 3, n)) == d
    assert SH_BASE_FLOATS == 11
    # SH buckets must bracket the SG arm's 14 + 7L (14/21/28/35).
    assert [sh_storage_floats(d) for d in range(5)] == [14, 23, 38, 59, 86]
    assert abs(lobe_equivalent(1) - 12 / 7) < 1e-9
    assert abs(lobe_equivalent(2) - 27 / 7) < 1e-9
    print("OK  degree/storage accounting (SH 14/23/38/59/86 vs SG 14/21/28/35)")


def test_truncate_slices_and_preserves_dc():
    g = make_gaussians(degree=4)
    out = truncate_sh(g, 2)
    assert out is not g
    assert out.harmonics.shape == (5, 3, 9)
    assert torch.equal(out.harmonics, g.harmonics[..., :9])
    # band 0 is the mean colour: truncation must leave it bit-identical.
    assert torch.equal(out.harmonics[..., :1], g.harmonics[..., :1])
    # a real slice, so the dropped bands stop occupying storage.
    assert out.harmonics.is_contiguous()
    assert out.harmonics.data_ptr() != g.harmonics.data_ptr()
    # other fields ride along by reference; the input is untouched.
    assert out.means is g.means and out.opacities is g.opacities
    assert g.harmonics.shape == (5, 3, 25)
    print("OK  truncation slices bands and keeps c0")


def test_identity_and_errors():
    g = make_gaussians(degree=2)
    assert truncate_sh(g, 2) is g
    for bad in (-1, 3):
        try:
            truncate_sh(g, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"keep_degree={bad} should raise")
    try:
        degree_of(torch.zeros(5, 3, 13))
    except ValueError:
        pass
    else:
        raise AssertionError("non-square width should raise")
    try:
        truncate_sh(object(), 1)
    except TypeError:
        pass
    else:
        raise AssertionError("non-dataclass should raise")
    print("OK  identity, range, width and type guards")


def test_degree_zero():
    g = make_gaussians(degree=4)
    out = truncate_sh(g, 0)
    assert out.harmonics.shape == (5, 3, 1)
    print("OK  degree-0 truncation leaves the c0-only model (14 floats/point)")


def test_runtime_hook():
    g = make_gaussians(degree=4)
    os.environ.pop("SH_KEEP_DEGREE", None)
    set_sh_keep_degree(None)
    assert get_sh_keep_degree() is None
    assert apply_runtime_sh_budget(g) is g

    set_sh_keep_degree(2)
    assert apply_runtime_sh_budget(g).harmonics.shape[-1] == 9

    set_sh_keep_degree(None)
    os.environ["SH_KEEP_DEGREE"] = "1"
    assert apply_runtime_sh_budget(g).harmonics.shape[-1] == 4

    os.environ["SH_KEEP_DEGREE"] = "4"
    assert apply_runtime_sh_budget(g) is g
    del os.environ["SH_KEEP_DEGREE"]
    print("OK  runtime hook (setter + SH_KEEP_DEGREE), unset is a no-op")


def main():
    test_degree_and_storage()
    test_truncate_slices_and_preserves_dc()
    test_identity_and_errors()
    test_degree_zero()
    test_runtime_hook()
    print("all sh_prune tests passed")


if __name__ == "__main__":
    main()
