#!/usr/bin/env bash
# Per-stage latency profile of the inference pipeline.
#
# Three configurations run over the same 20-scene slice so the stage table can
# be read side by side:
#
#   sg3_34   our budget mode, 34.40 floats per pixel (EcoSplat's 40% column)
#   sg3_4    our budget mode, 4.30 floats per pixel (EcoSplat's 5% column)
#   eco_40   EcoSplat SH4, 40% column (86 floats per point)
#
#   bash profile_stages.sh                 # all three
#   ONLY=sg3_34 bash profile_stages.sh     # one of them
#
# The card has to be idle: the numbers are wall clock, so a co-tenant on the
# same GPU shows up as noise in every stage. Pass a UUID in CUDA_VISIBLE_DEVICES.
#
# Profiles land in figs/prof/<tag>.json, logs in logs/prof_<tag>.log.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-/data14T/xz/miniconda3/envs/ecosplat/bin/python}"
: "${CUDA_VISIBLE_DEVICES:=GPU-f44dce23-ff1c-5f41-c1b5-d5aa1ad49cc5}"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID

INDEX=assets/evaluation_index_re10k_20.json
PROF_DIR="${PROF_DIR:-/data14T/xz/figs/prof}"
SKIP="${SKIP:-3}"
ONLY="${ONLY:-sg3_34 sg3_4 eco_40}"
mkdir -p "${PROF_DIR}" logs

# A twenty-scene slice keeps each run near three minutes once the kernels are
# compiled; the exact scenes do not matter, the stage split does.
if [ ! -f "${INDEX}" ]; then
    "${PY}" - <<'PY'
import json
src = json.load(open("assets/evaluation_index_re10k_200.json"))
keys = sorted(src)[:20]
json.dump({k: src[k] for k in keys}, open("assets/evaluation_index_re10k_20.json", "w"))
print(len(keys), "scenes")
PY
fi

run_one () {
    local tag="$1"; shift
    local log="logs/prof_${tag}.log"
    echo "[prof] ${tag} start $(date +%F' '%T)" | tee -a "${log}"
    if PROFILE_STAGES=1 PROFILE_STAGES_SKIP="${SKIP}" \
       PROFILE_STAGES_OUT="${PROF_DIR}/${tag}.json" "$@" >> "${log}" 2>&1; then
        echo "[prof] ${tag} DONE $(date +%F' '%T)" | tee -a "${log}"
    else
        echo "[prof] ${tag} FAILED rc=$? $(date +%F' '%T)" | tee -a "${log}"
    fi
}

common=(
    mode=test
    wandb.mode=disabled
    dataset.re10k.roots=[/data14T/xz/data/re10k]
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation
    dataset.re10k.view_sampler.index_path="${INDEX}"
    dataset.re10k.view_sampler.num_context_views=16
    test.save_image=false
    test.align_pose=true
    test.compute_scores=true
)

if [[ " ${ONLY} " == *" sg3_34 "* ]]; then
    run_one sg3_34 "${PY}" -u -m src.main \
        +experiment=spfsplat/sg/re10k_10view_sg_budget_igf "${common[@]}" \
        wandb.name=prof_sg3_34 \
        model.encoder.igf.budget_mode=true \
        +model.encoder.igf.inference_rho=0.9828421456473214 \
        model.encoder.pretrained_weights=/data14T/xz/ckpt_archive/stage1_sg3_step40000.ckpt \
        checkpointing.load=/data14T/xz/ckpt_archive/igf_budget_sg3_step100000.ckpt \
        test.output_path=outputs/test/prof_sg3_34
fi

if [[ " ${ONLY} " == *" sg3_4 "* ]]; then
    run_one sg3_4 "${PY}" -u -m src.main \
        +experiment=spfsplat/sg/re10k_10view_sg_budget_igf "${common[@]}" \
        wandb.name=prof_sg3_4 \
        model.encoder.igf.budget_mode=true \
        +model.encoder.igf.inference_rho=0.1228271484375 \
        model.encoder.pretrained_weights=/data14T/xz/ckpt_archive/stage1_sg3_step40000.ckpt \
        checkpointing.load=/data14T/xz/ckpt_archive/igf_budget_sg3_step100000.ckpt \
        test.output_path=outputs/test/prof_sg3_4
fi

if [[ " ${ONLY} " == *" eco_40 "* ]]; then
    run_one eco_40 "${PY}" -u -m src.main \
        +experiment=spfsplat/re10k_10view "${common[@]}" \
        wandb.name=prof_eco_40 \
        +model.encoder.igf.inference_rho=0.4 \
        +model.encoder.min_protect_rate=0.05 \
        checkpointing.load=pretrained_weights/ecosplat-spfsplat-re10k.ckpt \
        test.output_path=outputs/test/prof_eco_40
fi

echo "[prof] all done $(date +%F' '%T)"
