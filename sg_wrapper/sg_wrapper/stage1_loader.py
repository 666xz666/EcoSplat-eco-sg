"""Partial initialization of an EcoSplat encoder from a stage-1 (PGT) checkpoint.

Purpose: the representation-swap experiments (SG head, reduced-SH baselines)
freeze the backbone + center head and retrain only the Gaussian parameter
heads, whose output width changes with the parametrization. This loader

  1. copies every checkpoint tensor whose name AND shape match the target
     module — this fully loads backbone, center heads and pose head, and
     within the parameter heads loads the whole DPT trunk;
  2. for the parameter heads' final pointwise conv (the only layer whose
     shape depends on the parametrization), copies the leading output
     channels that are semantically aligned across parametrizations:

         [0] density, [1:4] scale, [4:8] rotation, [8:11] c0 / SH-DC

     (11 channels; everything behind them — higher SH degrees or SG lobes —
     keeps its fresh initialization);
  3. leaves everything else untouched and reports what happened.

Works for SH-degree baselines (out 83 -> 20 / 35) and for the SG head
(83 -> 32) alike, giving all variants the same warm-start recipe.
"""

from __future__ import annotations

import torch


def _is_head_final_weight(own: torch.Tensor, ckpt: torch.Tensor) -> bool:
    return (
        own.ndim == 4
        and own.shape[1:] == ckpt.shape[1:]
        and own.shape[2:] == (1, 1)
        and own.shape[0] < ckpt.shape[0]
    )


def _is_head_final_bias(own: torch.Tensor, ckpt: torch.Tensor) -> bool:
    return own.ndim == 1 and own.shape[0] < ckpt.shape[0]


def is_stage1_checkpoint(path: str) -> bool:
    """判断文件是否为带 encoder 权重的 stage-1 Lightning checkpoint。

    表示替换实验要从 stage-1 热启动，而从零开始的第一阶段（PGT）从上游
    MASt3R/CroCo 文件初始化，两者都走 model.encoder.pretrained_weights 这个
    入口，只能靠文件结构区分：stage-1 顶层有 state_dict 且键以 encoder. 开头，
    上游文件顶层是 model。用 map_location="meta" 只读元数据、不搬张量。

    Args:
        path: checkpoint 或上游权重文件的路径。

    Returns:
        True 表示按 stage-1 部分加载处理，False 表示交给 src/main.py 加载。
    """
    try:
        ckpt = torch.load(path, map_location="meta")
    except Exception:
        # 结构无法识别时保留历史行为，避免已有实验悄悄失去热启动。
        return True
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    return isinstance(state_dict, dict) and any(
        key.startswith("encoder.") for key in state_dict
    )


def init_encoder_from_stage1(
    encoder: torch.nn.Module,
    ckpt_path: str,
    aligned_head_channels: int = 11,
    verbose: bool = True,
) -> dict:
    """Load a stage-1 checkpoint into ``encoder`` (see module docstring).

    Returns a summary dict: {"loaded", "aligned", "skipped", "missing"}.
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    prefix = "encoder."
    ckpt_enc = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    if not ckpt_enc:
        raise RuntimeError(
            f"No 'encoder.*' keys in {ckpt_path}; is this a stage-1 "
            "SPFSplat/EcoSplat checkpoint?"
        )

    own = encoder.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    aligned: list[str] = []
    skipped: list[str] = []

    for key, value in ckpt_enc.items():
        if key not in own:
            skipped.append(f"{key} (absent in target)")
            continue
        target = own[key]
        if target.shape == value.shape:
            to_load[key] = value
            continue
        is_param_head = "gaussian_param_head" in key
        if is_param_head and (
            _is_head_final_weight(target, value) or _is_head_final_bias(target, value)
        ):
            n = min(aligned_head_channels, target.shape[0])
            merged = target.clone()
            merged[:n] = value[:n]
            to_load[key] = merged
            aligned.append(f"{key} ({tuple(value.shape)} -> {tuple(target.shape)}, kept {n} ch)")
        else:
            skipped.append(f"{key} (shape {tuple(value.shape)} vs {tuple(target.shape)})")

    missing = [k for k in own if k not in to_load]
    encoder.load_state_dict(to_load, strict=False)

    summary = {
        "loaded": len(to_load),
        "aligned": aligned,
        "skipped": skipped,
        "missing": missing,
    }
    if verbose:
        print(f"[sg_wrapper] stage-1 partial init from {ckpt_path}")
        print(f"[sg_wrapper] loaded {len(to_load)} tensors")
        for line in aligned:
            print(f"[sg_wrapper] aligned-copy {line}")
        for line in skipped:
            print(f"[sg_wrapper] skipped {line}")
        if missing:
            print(f"[sg_wrapper] left fresh: {missing}")
    return summary
