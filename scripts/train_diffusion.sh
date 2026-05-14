#!/usr/bin/env bash
# Diffusion Policy on PushT — canonical hyperparameters per the upstream
# model card (https://huggingface.co/lerobot/diffusion_pusht).
# Target: pc_success_max_overlap ≈ 65.4%, avg_max_reward ≈ 0.955.
#
# --dataset.use_imagenet_stats=true is the lerobot ≥0.5.0 default but is
# passed explicitly here: (a) documents intent, (b) is defensive against
# a future default flip, (c) is the load-bearing fix for lerobot 0.4.x
# where the default was False (issue #3221).
#
# VRAM: batch_size=64 needs ≳12GB. Will OOM on the local GTX 1650 (4GB).
# Use scripts/train_diffusion_dryrun.sh for local sanity; full run on RunPod.
#
# Resume: if $OUTPUT_DIR already has checkpoints, the script adds --resume=true
# automatically so interrupted RunPod runs continue rather than restart.
#
# wandb: enabled iff WANDB_API_KEY is set in the environment (avoids the
# interactive login prompt that would otherwise hang headless runs).
#
# Hub push: lerobot 0.5.x defaults policy.push_to_hub=true (which forces
# a repo_id requirement at config-validation time). We default to false
# here and opt-in via HF_REPO_ID, e.g.:
#     HF_REPO_ID=adimunot/diffusion_pusht ./scripts/train_diffusion.sh

set -euo pipefail

# Caller is responsible for activating the right env. We just check the
# lerobot CLI is on PATH so the failure mode is clear instead of cryptic.
command -v lerobot-train >/dev/null || {
  echo "[train_diffusion] lerobot-train not on PATH. Activate the env first." >&2
  exit 1
}

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/diffusion_pusht}"

WANDB_ARG="--wandb.enable=false"
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARG="--wandb.enable=true"
fi

HUB_ARGS="--policy.push_to_hub=false"
if [ -n "${HF_REPO_ID:-}" ]; then
  HUB_ARGS="--policy.push_to_hub=true --policy.repo_id=${HF_REPO_ID}"
fi

RESUME_ARG=""
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
  RESUME_ARG="--resume=true"
  echo "[train_diffusion] Resuming from $OUTPUT_DIR"
fi

exec lerobot-train \
  --policy.type=diffusion \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/pusht \
  --dataset.use_imagenet_stats=true \
  --env.type=pusht \
  --batch_size=64 \
  --steps=200000 \
  --eval_freq=25000 \
  --save_freq=25000 \
  --seed=100000 \
  --output_dir="$OUTPUT_DIR" \
  $WANDB_ARG \
  $HUB_ARGS \
  $RESUME_ARG \
  "$@"
