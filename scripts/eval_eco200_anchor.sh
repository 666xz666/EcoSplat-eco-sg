#!/usr/bin/env bash
# Evaluate the released EcoSplat checkpoint at the paper's budget levels on the
# 200-scene index with 16 context views, so the numbers line up with our rungs.
set -u
cd /data14T/xz/repos/EcoSplat-eco-sg

PY="${PY:-/data14T/xz/miniconda3/envs/ecosplat/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-d46a2738-60b6-72a4-0ce0-e06727c3498c}"
CKPT="${CKPT:-pretrained_weights/ecosplat-spfsplat-re10k.ckpt}"
IDX="${EVAL_INDEX:-assets/evaluation_index_re10k_200.json}"
NUM_CTX="${NUM_CTX:-16}"
DATA_ROOT="${DATA_ROOT:-/data14T/xz/data/re10k}"
LOG="${LOG:-logs/eval_eco200_anchor.log}"
TAGPREFIX="${TAGPREFIX:-eco200}"
MIN_RATE="${MIN_RATE:-0.05}"

: > "$LOG"
echo "[eco] start $(date '+%F %T') ckpt=$CKPT index=$IDX ctx=$NUM_CTX min_rate=$MIN_RATE root=$DATA_ROOT" | tee -a "$LOG"

for RHO in 0.05 0.10 0.40; do
  TAG="${TAGPREFIX}_rho$(echo "$RHO" | tr '.' 'p')"
  echo "[eco] rho=$RHO start $(date '+%F %T')" | tee -a "$LOG"
  "$PY" -u -m src.main +experiment=spfsplat/re10k_10view \
    mode=test wandb.mode=disabled wandb.name="$TAG" \
    +model.encoder.igf.inference_rho="$RHO" \
    +model.encoder.min_protect_rate="$MIN_RATE" \
    dataset.re10k.roots=["$DATA_ROOT"] \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path="$IDX" \
    dataset.re10k.view_sampler.num_context_views="$NUM_CTX" \
    checkpointing.load="$CKPT" \
    test.save_image=false test.align_pose=true test.compute_scores=true \
    test.output_path="outputs/test/$TAG" >> "$LOG" 2>&1
  echo "[eco] rho=$RHO DONE rc=$? $(date '+%F %T')" | tee -a "$LOG"
done
echo "[eco] ALL DONE $(date '+%F %T')" | tee -a "$LOG"
