"""End-to-end wiring smoke test for the eco-sg branch (no dataset needed).

Builds the encoder + decoder from an experiment yaml, runs one synthetic
forward (10 context views + 1 target, 256x256, random images), renders
through the gsplat colors_precomp path, backprops, and reports which
parameters received gradients. Verifies every integration point of the
SG branch: adapter swap, head width, flattening, decoder branch, color
evaluation, and head-only trainability.

Random weights by default; set CKPT to also exercise the stage-1 partial
loader (backbone/center/pose heads + aligned head channels).

Run from the repo root, inside the EcoSplat conda env (seconds to ~1 min
on GPU; the first call also JIT-compiles gsplat kernels if no cache):

    python sg_wrapper/tests/smoke_test_sg.py           # SG 3-lobe variant
    python sg_wrapper/tests/smoke_test_sg.py sh1       # SH degree-1 baseline
    python sg_wrapper/tests/smoke_test_sg.py sh2       # SH degree-2 baseline
    CKPT=pretrained_weights/re10k_10view.ckpt \
        python sg_wrapper/tests/smoke_test_sg.py       # + partial init check
"""

import os
import sys
import time

import torch

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# REPO/sg_wrapper (the vendored *project* dir, no __init__.py) must not
# shadow the real package: put the project dir first on the path so the
# regular package inside it wins over the namespace-package fallback that
# REPO itself would produce.
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sg_wrapper"))
os.chdir(REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from jaxtyping import install_import_hook  # noqa: E402

EXPERIMENTS = {
    "sg3": "spfsplat/re10k_10view_sg",
    "sh1": "spfsplat/re10k_10view_sh1",
    "sh2": "spfsplat/re10k_10view_sh2",
}


def make_views(b: int, v: int, h: int, w: int, device: str) -> dict:
    image = torch.rand(b, v, 3, h, w, device=device)
    intrinsics = torch.eye(3, device=device).repeat(b, v, 1, 1)
    intrinsics[..., 0, 0] = 0.7  # normalized focal length
    intrinsics[..., 1, 1] = 0.7
    extrinsics = torch.eye(4, device=device).repeat(b, v, 1, 1)
    return {
        "image": image,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,  # unused on the pose-free path
        "near": torch.full((b, v), 0.05, device=device),
        "far": torch.full((b, v), 20.0, device=device),
        "index": torch.arange(v, device=device),
    }


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "sg3"
    if variant not in EXPERIMENTS:
        raise SystemExit(f"unknown variant {variant!r}; expected one of {list(EXPERIMENTS)}")

    with install_import_hook(("src",), ("beartype", "beartype")):
        from src.config import load_typed_root_config
        from src.model.decoder import get_decoder
        from src.model.encoder import get_encoder

        overrides = [f"+experiment={EXPERIMENTS[variant]}"]
        ckpt = os.environ.get("CKPT", "")
        if not ckpt:
            # Random init: skip the partial stage-1 load inside encoder init.
            overrides.append('model.encoder.pretrained_weights=""')
        else:
            overrides.append(f"model.encoder.pretrained_weights={ckpt}")

        with initialize_config_dir(
            config_dir=os.path.join(REPO, "config"), version_base=None
        ):
            cfg_dict = compose(config_name="main", overrides=overrides)
        cfg = load_typed_root_config(cfg_dict)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[smoke] variant={variant} device={device}")

        encoder, _ = get_encoder(cfg.model.encoder)
        decoder = get_decoder(cfg.model.decoder)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        print(f"[smoke] raw_gs_dim={encoder.raw_gs_dim} "
              f"(SH4 would be 83; SG-3lobe 32; SH1 20; SH2 35)")
        if ckpt:
            # The encoder __init__ already ran the partial load; this only
            # reports what is trainable now.
            pass

        # Mirror train.freeze_pretrained + freeze_pose_head: only the
        # gaussian parameter heads are trainable.
        encoder.eval()  # skip the training-mode guard; grads still flow
        trainable = []
        for name, p in encoder.named_parameters():
            p.requires_grad_("gaussian_param_head" in name)
            if p.requires_grad:
                trainable.append(name)
        n_train = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        n_total = sum(p.numel() for p in encoder.parameters())
        print(f"[smoke] trainable tensors: {len(trainable)} "
              f"({n_train:,} / {n_total:,} params)")

        b, v_cxt, v_tgt, h, w = 1, 10, 1, 256, 256
        context = make_views(b, v_cxt, h, w, device)
        target = make_views(b, v_tgt, h, w, device)

        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        t0 = time.time()
        encoder_output = encoder(context, 0, target=target)
        gaussians = encoder_output["gaussians"]
        pred_cwt = encoder_output["extrinsics"]["cwt"]
        target_extrinsics = pred_cwt[:, v_cxt:]

        from sg_wrapper import SGGaussians

        if variant == "sg3":
            assert isinstance(gaussians, SGGaussians), type(gaussians)
            print(f"[smoke] gaussians: {gaussians.means.shape[1]} "
                  f"lobes/Gaussian: {gaussians.sg_directions.shape[-2]}")
        else:
            assert not isinstance(gaussians, SGGaussians)
            print(f"[smoke] gaussians: {gaussians.means.shape[1]} "
                  f"d_sh: {gaussians.harmonics.shape[-1]}")

        output = decoder.forward(
            gaussians,
            target_extrinsics,
            target["intrinsics"],
            target["near"],
            target["far"],
            (h, w),
        )
        loss = torch.nn.functional.mse_loss(output.color, target["image"])
        loss.backward()
        t1 = time.time()

        print(f"[smoke] render {tuple(output.color.shape)} "
              f"color range [{output.color.min():.3f}, {output.color.max():.3f}] "
              f"alpha mean {output.alpha.mean():.3f}")
        print(f"[smoke] loss={loss.item():.4f}  fwd+bwd={t1 - t0:.1f}s")
        if device == "cuda":
            print(f"[smoke] peak GPU mem: "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

        grad_names = sorted({
            n for n, p in encoder.named_parameters() if p.grad is not None
        })
        head_grads = [n for n in grad_names if "gaussian_param_head" in n]
        other_grads = [n for n in grad_names if "gaussian_param_head" not in n]
        print(f"[smoke] params with grads: {len(grad_names)} "
              f"(head: {len(head_grads)}, non-head: {len(other_grads)})")
        if other_grads:
            print(f"[smoke] WARNING non-head grads: {other_grads[:5]}")
        if variant == "sg3" and not head_grads:
            raise SystemExit("FAIL: no gradients reached the SG heads")
        print("[smoke] OK")


if __name__ == "__main__":
    main()
