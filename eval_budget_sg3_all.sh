#!/usr/bin/env bash
# Walk the budget-mode checkpoint down the rungs that match EcoSplat's budget
# columns, byte for byte.
#
# EcoSplat keeps k = int(max(rho, 1/16) * H * W) points per view and every
# point carries 86 floats (opacity 1 + scale 3 + rotation 4 + c0 3 + SH4 75).
# Our full configuration carries 35 floats per point (14 + 3 lobes * 7), so the
# same payload needs rho_ours = 86 * rho_eff / 35, and the allocator then snaps
# it down to the 7-float grid (F* = floor(kappa / 7) * 7).  The constants below
# are that arithmetic evaluated at H = W = 256:
#
#   Eco column   rho_eff     Eco f/px    rho_ours               our F*    our f/px
#   5%           0.0625      5.37500     0.15357142857142858    352254    5.37497
#   10%          0.0999908   8.59921     0.24569178989955356    563556    8.59918
#   40%          0.3999939   34.39948    0.9828421456473214     2254399   34.39940
#   70%          0.7         60.20000    not reachable: 60.2 > 35 floats / pixel
#
# The rho_eff values for 10% and 40% are below the nominal column because
# int(0.10 * 65536) = 6553 and int(0.40 * 65536) = 26214 truncate.
#
#   bash eval_budget_sg3_all.sh
#   EVAL_INDEX=assets/evaluation_index_re10k_200.json bash eval_budget_sg3_all.sh
#   RUNGS="0.15357142857142858" bash eval_budget_sg3_all.sh
set -uo pipefail
cd "$(dirname "$0")"

CKPT="${CKPT:-/data14T/xz/ckpt_archive/igf_budget_sg3_step30000.ckpt}"
TAG="${TAG:-matched_sg3_30k}"
# The 200-scene index finishes a rung in about half an hour and keeps every
# model on the same scenes; the full index is for the table that goes in the
# paper and takes most of a day per rung.
export EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"
LOG="logs/eval_${TAG}.log"
# One census file for all rungs: the allocator writes one row per scene, so
# after the sweep it shows how the same rate is spent across scenes.
CENSUS="${CENSUS:-logs/budget_census_${TAG}.csv}"
export BUDGET_CENSUS_CSV="${CENSUS}"
RUNGS="${RUNGS:-0.15357142857142858 0.24569178989955356 0.9828421456473214}"
mkdir -p logs

echo "[eval] start $(date +%F' '%T) ckpt=${CKPT} index=${EVAL_INDEX} census=${CENSUS}" | tee -a "${LOG}"
for rv in ${RUNGS}; do
    echo "[eval] rho=${rv} start $(date +%F' '%T)" | tee -a "${LOG}"
    if RHO="${rv}" CKPT="${CKPT}" TAG="${TAG}" bash eval_budget_sg3_rho.sh >> "${LOG}" 2>&1; then
        echo "[eval] rho=${rv} DONE rc=0 $(date +%F' '%T)" | tee -a "${LOG}"
    else
        echo "[eval] rho=${rv} FAILED rc=$? $(date +%F' '%T)" | tee -a "${LOG}"
    fi
done
echo "[eval] ALL DONE $(date +%F' '%T)" | tee -a "${LOG}"
