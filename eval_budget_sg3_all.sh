#!/usr/bin/env bash
# Walk the budget-mode checkpoint down four byte-aligned rungs on the full
# 16-view evaluation index.  0.1536 is the rung that matches EcoSplat's lowest
# budget, so its number is the same-bytes comparison; 0.40 is the high rung
# where the point axis already saturates.
#
#   bash eval_budget_sg3_all.sh                       # default checkpoint
#   CKPT=/data14T/xz/ckpt_archive/... bash eval_budget_sg3_all.sh
set -uo pipefail
cd "$(dirname "$0")"

CKPT="${CKPT:-/data14T/xz/ckpt_archive/igf_budget_sg3_step30000.ckpt}"
TAG="${TAG:-budget_sg3_30k}"
LOG="logs/eval_${TAG}.log"
mkdir -p logs

echo "[eval] start $(date +%F' '%T) ckpt=${CKPT}" | tee -a "${LOG}"
for rv in 0.05 0.10 0.154 0.40; do
    echo "[eval] rho=${rv} start $(date +%F' '%T)" | tee -a "${LOG}"
    if RHO="${rv}" CKPT="${CKPT}" TAG="${TAG}" bash eval_budget_sg3_rho.sh >> "${LOG}" 2>&1; then
        echo "[eval] rho=${rv} DONE rc=0 $(date +%F' '%T)" | tee -a "${LOG}"
    else
        echo "[eval] rho=${rv} FAILED rc=$? $(date +%F' '%T)" | tee -a "${LOG}"
    fi
done
echo "[eval] ALL DONE $(date +%F' '%T)" | tee -a "${LOG}"
