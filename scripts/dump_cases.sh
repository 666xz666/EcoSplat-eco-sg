#!/usr/bin/env bash
# Dump one rendered view per case scene at three payload rates, for the
# per-pixel budget figures. Four scenes: two where the budget mode wins and
# two where it loses, chosen by the sign of the paired per-scene delta.
set -uo pipefail
cd /data14T/xz/repos/EcoSplat-eco-sg

export CUDA_VISIBLE_DEVICES=GPU-74f46e3f-503a-76c0-68f9-1ba3b935f0eb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PY=/data14T/xz/miniconda3/envs/v100/bin/python
export SG_PIXEL_DUMP_SCENES="47a66fa042406908,02b618a34bc12ff9,0b564b685315d8ff,cee475fc8a99c826"
export SG_PIXEL_DUMP_MAX=1
export EVAL_INDEX=assets/evaluation_index_re10k_case4.json
export NUM_CTX=16
export CKPT=/data14T/xz/ckpt_archive/igf_budget_sg3_step50000.ckpt
export BASE=/data14T/xz/ckpt_archive/stage1_sg3_step40000.ckpt

for spec in 0.1228271484375:0p12283 0.24569178989955356:0p24569 0.9828421456473214:0p98284; do
  RHO_VAL="${spec%%:*}"
  RHO_TAG="${spec##*:}"
  export SG_PIXEL_DUMP_DIR="/data14T/xz/figs/pixel_dump/rho_${RHO_TAG}"
  mkdir -p "$SG_PIXEL_DUMP_DIR"
  echo "[dump] start rho=${RHO_VAL} dir=${SG_PIXEL_DUMP_DIR} $(date +%H:%M:%S)"
  RHO="$RHO_VAL" TAG="casedump_${RHO_TAG}" bash scripts/eval_budget_sg3_rho.sh
  echo "[dump] done rho=${RHO_VAL} rc=$? $(date +%H:%M:%S)"
done
echo "[dump] ALL DONE"
