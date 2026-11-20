#!/usr/bin/env bash
# Evaluate a budget-mode SG-3 IGF checkpoint at one total-payload rate.
#
#   RHO=0.15357142857142858 ./eval_budget_sg3_rho.sh
#   RHO=0.9828421456473214 CKPT=/path/to.ckpt TAG=matched_sg3_30k ./eval_budget_sg3_rho.sh
#
# RHO is read as a fraction of the full payload (35 floats per pixel: 14 per
# point plus 7 per of 3 lobes), so the byte axis is 35*RHO floats per pixel.
# A rate that matches an EcoSplat column is 86 * rho_eff / 35, where rho_eff is
# their kept fraction after the 1/16 floor; eval_budget_sg3_all.sh carries the
# three values already evaluated.
set -euo pipefail
# A 5880 with room to spare; pass a UUID so the CUDA ordinal does not matter.
: "${CUDA_VISIBLE_DEVICES:=GPU-d46a2738-60b6-72a4-0ce0-e06727c3498c}"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID

RHO="${RHO:?set RHO to a payload fraction, e.g. 0.15357142857142858}"
CKPT="${CKPT:-/data14T/xz/ckpt_archive/igf_budget_sg3_step30000.ckpt}"
BASE="${BASE:-/data14T/xz/ckpt_archive/stage1_sg3_step40000.ckpt}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"
NUM_CTX="${NUM_CTX:-16}"
TAG="${TAG:-budget_sg3_30k}"
# The matched rates carry many digits; seven are enough to name a directory
# without colliding (0.15357 / 0.24569 / 0.98284).
RUN_NAME="${TAG}_rho${RHO:0:7}"
RUN_NAME="${RUN_NAME//./p}"
# The conda environment is not on PATH for a bare non-interactive shell, so the
# interpreter is taken from PY; a bare python has no hydra and dies at import.
PY="${PY:-python}"

# The encoder strict-loads BASE at init (the IGF branch requires a stage-1
# checkpoint) and Lightning then restores the IGF weights from CKPT.
"$PY" -u -m src.main +experiment=spfsplat/sg/re10k_10view_sg_budget_igf \
    mode=test \
    wandb.mode=disabled wandb.name="${RUN_NAME}" \
    model.encoder.igf.budget_mode=true \
    +model.encoder.igf.inference_rho="${RHO}" \
    model.encoder.pretrained_weights="${BASE}" \
    dataset.re10k.roots=[/data14T/xz/data/re10k] \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path="${EVAL_INDEX}" \
    dataset.re10k.view_sampler.num_context_views="${NUM_CTX}" \
    checkpointing.load="${CKPT}" \
    test.save_image=false \
    test.align_pose=true \
    test.compute_scores=true \
    test.output_path="outputs/test/${RUN_NAME}"
