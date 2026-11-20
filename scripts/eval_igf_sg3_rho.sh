#!/usr/bin/env bash
# Evaluate an sg3_igf checkpoint (SG 3-lobe head + IGF) at a given rho.
#   RHO=0.4 ./eval_igf_sg3_rho.sh
#   RHO=0.02 CKPT=<igf_ckpt> BASE=<sg_base_ckpt> ./eval_igf_sg3_rho.sh
# NOTE: unlike eval_igf_rho0pXX.sh (stock SH4 base), this uses the SG experiment;
# the encoder strict-loads BASE at init (IGF branch requirement) and Lightning
# then restores the trained IGF weights from CKPT.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID

RHO="${RHO:?set RHO=0.02|0.1|0.4|0.7}"
CKPT="${CKPT:-/data14T/xz/outputs/sg3_igf_ddp/checkpoints/epoch_1-step_30000.ckpt}"
BASE="${BASE:-/data14T/xz/ckpt_archive/sg3_step20000.ckpt}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"
RUN_NAME="sg3_igf_rho${RHO//./p}"

python -m src.main +experiment=spfsplat/sg/re10k_10view_sg \
    mode=test \
    wandb.mode=disabled wandb.name="${RUN_NAME}" \
    "+model.encoder.igf={inference_rho:${RHO}}" \
    +model.encoder.min_protect_rate=0.05 \
    model.encoder.pretrained_weights="${BASE}" \
    dataset.re10k.roots=[/data14T/xz/data/re10k] \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path="${EVAL_INDEX}" \
    dataset.re10k.view_sampler.num_context_views=24 \
    checkpointing.load="${CKPT}" \
    test.save_image=false \
    test.align_pose=true \
    test.compute_scores=true \
    test.output_path="outputs/test/${RUN_NAME}"
