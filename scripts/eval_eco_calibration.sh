#!/usr/bin/env bash
# 校准：用官方 EcoSplat 权重在论文口径下跑 IGF ρ 扫描，目标复现 Table 1
# （5%/10%/40%/70% = 24.72/25.00/25.11/25.00，24 视角）。对上即证明我们的
# 评测 harness == 论文 harness，之后我方方法同口径的数字可直接与论文并列。
#
# 口径严格照作者随仓库发布的 eval_igf_rho0p*.sh（初始提交 12c6e82）：
#   +experiment=spfsplat/re10k_10view、num_context_views=24、全量 index、
#   align_pose=true、不设 min_protect_rate（用代码默认 1/16=6.25%，
#   即作者当年跑「5%」档的实际下限——复现必须照它，勿改成 0.05）。
#
#   CKPT=pretrained_weights/ecosplat-spfsplat-re10k.ckpt \
#   CUDA_VISIBLE_DEVICES=<UUID> bash scripts/eval_eco_calibration.sh
#   RHO_LIST="0.4 0.7" ...   # 先跑两档验证口径，再补齐
set -uo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a GPU UUID}"
PY="${PY:-python}"
CKPT="${CKPT:?set CKPT to the official ecosplat-spfsplat-re10k.ckpt}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"   # 全量，论文口径
NUM_CTX="${NUM_CTX:-24}"
# 默认四档；代码默认 1/16 下限会把 0.05 抬到 0.0625，与作者当年一致
RHO_LIST="${RHO_LIST:-0.05 0.1 0.4 0.7}"
mkdir -p logs

for rho in ${RHO_LIST}; do
    name="calib_eco_24v_rho${rho}"
    log="logs/${name}.log"
    echo "[calib] rho=${rho} start $(date +%F' '%T)" | tee -a "${log}"
    if "$PY" -u -m src.main +experiment=spfsplat/re10k_10view \
        mode=test wandb.mode=disabled wandb.name="${name}" \
        "+model.encoder.igf.inference_rho=${rho}" \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path="${EVAL_INDEX}" \
        dataset.re10k.view_sampler.num_context_views="${NUM_CTX}" \
        checkpointing.load="${CKPT}" \
        test.save_image=false test.align_pose=true test.compute_scores=true \
        test.output_path="outputs/test/${name}" >> "${log}" 2>&1; then
        echo "[calib] rho=${rho} DONE $(date +%F' '%T)" | tee -a "${log}"
    else
        echo "[calib] rho=${rho} FAILED rc=$? $(date +%F' '%T)" | tee -a "${log}"
    fi
done
echo "[calib] all done $(date +%F' '%T)" | tee -a "logs/calib_eco_24v.log"
