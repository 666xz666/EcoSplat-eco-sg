#!/usr/bin/env bash
# Dump the released EcoSplat checkpoint on the same four case scenes, targets
# and byte budgets as dump_cases.sh, so the two arms can be compared pixel by
# pixel (offset energy versus residual energy).
set -uo pipefail
cd /data14T/xz/repos/EcoSplat-eco-sg

export CUDA_VISIBLE_DEVICES=GPU-0f9270a7-eaf2-2c59-cf1e-12a729dc4223
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PY=/data14T/xz/miniconda3/envs/ecosplat/bin/python
export SG_PIXEL_DUMP_SCENES="47a66fa042406908,02b618a34bc12ff9,0b564b685315d8ff,cee475fc8a99c826"
export SG_PIXEL_DUMP_MAX=1

CKPT=pretrained_weights/ecosplat-spfsplat-re10k.ckpt
IDX=assets/evaluation_index_re10k_case4.json
NUM_CTX=16
DATA_ROOT=/data14T/xz/data/re10k
LOG=logs/dump_eco_cases.log

: > "$LOG"
echo "[ecodump] start $(date '+%F %T')" | tee -a "$LOG"

# Same byte budgets as the ours dump: 4.30 / 8.60 / 34.40 float per pixel.
for spec in 0.05:0p12283 0.10:0p24569 0.40:0p98284; do
  RHO_VAL="${spec%%:*}"
  RHO_TAG="${spec##*:}"
  export SG_PIXEL_DUMP_DIR="/data14T/xz/figs/pixel_dump/eco_rho_${RHO_TAG}"
  mkdir -p "$SG_PIXEL_DUMP_DIR"
  echo "[ecodump] rho=${RHO_VAL} dir=${SG_PIXEL_DUMP_DIR} $(date '+%F %T')" | tee -a "$LOG"
  for attempt in 1 2 3; do
    "$PY" -u -m src.main +experiment=spfsplat/re10k_10view \
      mode=test wandb.mode=disabled wandb.name="ecodump_${RHO_TAG}" \
      +model.encoder.igf.inference_rho="$RHO_VAL" \
      +model.encoder.min_protect_rate=0.05 \
      dataset.re10k.roots=["$DATA_ROOT"] \
      dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
      dataset.re10k.view_sampler.index_path="$IDX" \
      dataset.re10k.view_sampler.num_context_views="$NUM_CTX" \
      checkpointing.load="$CKPT" \
      test.save_image=false test.align_pose=true test.compute_scores=false \
      test.output_path="outputs/test/ecodump_${RHO_TAG}" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -n "$(ls -A "$SG_PIXEL_DUMP_DIR" 2>/dev/null)" ]; then
      break
    fi
    echo "[ecodump] rho=${RHO_VAL} attempt=${attempt} rc=${rc} $(date '+%F %T')" | tee -a "$LOG"
    sleep 20
  done
  echo "[ecodump] rho=${RHO_VAL} DONE rc=$? $(date '+%F %T')" | tee -a "$LOG"
done
echo "[ecodump] ALL DONE $(date '+%F %T')" | tee -a "$LOG"
