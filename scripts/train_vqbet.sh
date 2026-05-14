#!/usr/bin/env bash
# VQ-BeT on PushT — canonical hyperparameters per the upstream model card
# (https://huggingface.co/lerobot/vqbet_pusht).
# Target: pc_success_max_overlap ≈ 63.8%, avg_max_reward ≈ 0.895.
#
# IMPORTANT — NO --dataset.use_imagenet_stats flag here, on purpose.
# VQ-BeT's config (src/lerobot/policies/vqbet/configuration_vqbet.py in
# lerobot 0.5.1) sets normalization_mapping["VISUAL"] = IDENTITY, i.e. no
# normalization of image observations at all. ImageNet stats are therefore
# irrelevant; passing the flag would be a no-op. We omit it explicitly to
# make this a clean control: if Diffusion+ACT improve from the fix but
# VQ-BeT is unchanged (because there was no bug for it), we've localized
# the root cause to the image-normalization path.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pusht

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/vqbet_pusht}"

WANDB_ARG="--wandb.enable=false"
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARG="--wandb.enable=true"
fi

# Hub push opt-in via HF_REPO_ID (see train_diffusion.sh for rationale).
HUB_ARGS="--policy.push_to_hub=false"
if [ -n "${HF_REPO_ID:-}" ]; then
  HUB_ARGS="--policy.push_to_hub=true --policy.repo_id=${HF_REPO_ID}"
fi

RESUME_ARG=""
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
  RESUME_ARG="--resume=true"
  echo "[train_vqbet] Resuming from $OUTPUT_DIR"
fi

exec lerobot-train \
  --policy.type=vqbet \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --batch_size=64 \
  --steps=250000 \
  --eval_freq=25000 \
  --save_freq=25000 \
  --seed=100000 \
  --output_dir="$OUTPUT_DIR" \
  $WANDB_ARG \
  $HUB_ARGS \
  $RESUME_ARG \
  "$@"
