#!/usr/bin/env bash
# Evaluate a representation-swap variant checkpoint (eco-sg branch).
# The checkpoint must be loaded with the SAME experiment config it was
# trained with (head width differs per variant: sg3=32, sh1=20, sh2=35).
#
#   ./eval_sg.sh sg3 CKPT=outputs/.../checkpoint.ckpt
#   ./eval_sg.sh sh1 CKPT=... EVAL_INDEX=assets/evaluation_index_acid.json
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES

VARIANT="${1:?usage: ./eval_sg.sh sg3|sh1|sh2|sh4 (CKPT=... required)}"
shift
CKPT="${CKPT:?set CKPT=path/to/checkpoint.ckpt}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"
# Official RE10K protocol: the model is trained with 10 context views but
# evaluated with 24 (see eval_igf_rho0p02.sh and the paper's 16/20/24 tables).
# The index stores 2 context views and the sampler adds random interior views
# up to this number. ACID uses 2; a 10-view ablation can pass NUM_CONTEXT=10.
NUM_CONTEXT="${NUM_CONTEXT:-24}"

case "$VARIANT" in
    sg3) EXPERIMENT=spfsplat/re10k_10view_sg ;;
    sh1) EXPERIMENT=spfsplat/re10k_10view_sh1 ;;
    sh2) EXPERIMENT=spfsplat/re10k_10view_sh2 ;;
    sh4) EXPERIMENT=spfsplat/re10k_10view_sh4 ;;
    *) echo "unknown variant: $VARIANT (expected sg3 | sh1 | sh2 | sh4)" >&2; exit 1 ;;
esac

python -m src.main "+experiment=${EXPERIMENT}" \
    mode=test \
    wandb.mode=disabled "wandb.name=eval_${VARIANT}" \
    model.encoder.pretrained_weights="" \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path="${EVAL_INDEX}" \
    dataset.re10k.view_sampler.num_context_views="${NUM_CONTEXT}" \
    checkpointing.load="${CKPT}" \
    test.save_image=true test.align_pose=true \
    "test.output_path=outputs/test/eval_${VARIANT}" "$@"
