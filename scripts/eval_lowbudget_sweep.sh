#!/usr/bin/env bash
# 低预算长尾扫描（我方 SG-IGF），口径与既有 16-view/200 场景字节表完全一致：
# EVAL_INDEX=200、NUM_CTX=16、align_pose=true、CKPT=igf_budget_sg3_step100000。
#
# f/px = 35 * RHO（14/点 + 7×3 lobe）。EcoSplat 的 5% 硬下限 = 86*0.05 = 4.30 f/px，
# 所以 4.30 以下只有我方能运行——这段长尾正是「SG 缓坡 vs SH 断崖」的主场。
#
#   bash scripts/eval_lowbudget_sweep.sh              # 全部六档
#   RHO_LIST="0.05 0.06" GPU=0 bash scripts/eval_lowbudget_sweep.sh   # 子集/指定卡
set -uo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a GPU UUID}"
PY="${PY:-python}"
CKPT="${CKPT:-/data14T/xz/ckpt_archive/igf_budget_sg3_step100000.ckpt}"
BASE="${BASE:-/data14T/xz/ckpt_archive/stage1_sg3_step40000.ckpt}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k_200.json}"
NUM_CTX="${NUM_CTX:-16}"
TAG="${TAG:-lowbudget_sg3_100k}"
RHO_LIST="${RHO_LIST:-0.05 0.06 0.07 0.08 0.09 0.10}"
mkdir -p logs

for rho in ${RHO_LIST}; do
    fpx=$(python -c "print(f'{35*${rho}:.2f}')" 2>/dev/null || echo "?")
    echo "[lowbudget] rho=${rho} (~${fpx} f/px) start $(date +%F' '%T)" | tee -a "logs/eval_${TAG}.log"
    if RHO="${rho}" CKPT="${CKPT}" BASE="${BASE}" EVAL_INDEX="${EVAL_INDEX}" \
       NUM_CTX="${NUM_CTX}" TAG="${TAG}" \
       bash scripts/eval_budget_sg3_rho.sh >> "logs/eval_${TAG}.log" 2>&1; then
        echo "[lowbudget] rho=${rho} DONE $(date +%F' '%T)" | tee -a "logs/eval_${TAG}.log"
    else
        echo "[lowbudget] rho=${rho} FAILED rc=$? $(date +%F' '%T)" | tee -a "logs/eval_${TAG}.log"
    fi
done
echo "[lowbudget] all done $(date +%F' '%T)" | tee -a "logs/eval_${TAG}.log"
