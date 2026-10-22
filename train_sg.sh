#!/usr/bin/env bash
# Representation-swap experiments (eco-sg branch): train SG / SH1 / SH2 / SH4
# variants with the identical heads-only recipe.
#
#   ./train_sg.sh sg3          # 3-lobe spherical Gaussians (main variant)
#   ./train_sg.sh sh1          # SH degree 1 baseline
#   ./train_sg.sh sh2          # SH degree 2 baseline
#   ./train_sg.sh sh4          # SH degree 4, same-recipe reference
#
# Notes:
#   * No checkpointing.load/resume here: the encoder inits from
#     model.encoder.pretrained_weights via sg_wrapper's partial loader
#     (Lightning resume would fail on the resized parameter heads).
#   * Stage-1 checkpoint must exist at pretrained_weights/re10k_10view.ckpt
#     (SPFSplat model zoo: RanranHuang/SPFSplat on Hugging Face).
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

VARIANT="${1:-sg3}"
case "$VARIANT" in
    sg3) EXPERIMENT=spfsplat/re10k_10view_sg ;;
    sh1) EXPERIMENT=spfsplat/re10k_10view_sh1 ;;
    sh2) EXPERIMENT=spfsplat/re10k_10view_sh2 ;;
    sh4) EXPERIMENT=spfsplat/re10k_10view_sh4 ;;
    *) echo "unknown variant: $VARIANT (expected sg3 | sh1 | sh2 | sh4)" >&2; exit 1 ;;
esac

python -m src.main "+experiment=${EXPERIMENT}" \
    wandb.mode=offline "wandb.name=re10k-${VARIANT}" "$@"
