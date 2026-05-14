#!/usr/bin/env bash
# ACT (Action Chunking Transformer) on PushT.
#
# There is no published lerobot/act_pusht model card, so this run is a
# sanity-check that the imagenet-stats fix generalizes across policies
# that share the ImageNet-pretrained ResNet encoder. Expected outcome:
# trains stably, eval clears chance (avg_max_reward > 0.5), even if it
# doesn't match Diffusion's 65%.
#
# --dataset.use_imagenet_stats=true: same rationale as train_diffusion.sh.
#
# Hyperparameters: smaller batch (ACT is heavier per-sample) and fewer
# steps than Diffusion. Standard lerobot defaults except for the eval/save
# frequencies, which are set to match the 80k-step horizon.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pusht

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_pusht}"

WANDB_ARG="--wandb.enable=false"
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARG="--wandb.enable=true"
fi

RESUME_ARG=""
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
  RESUME_ARG="--resume=true"
  echo "[train_act] Resuming from $OUTPUT_DIR"
fi

exec lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/pusht \
  --dataset.use_imagenet_stats=true \
  --env.type=pusht \
  --batch_size=8 \
  --steps=80000 \
  --eval_freq=10000 \
  --save_freq=10000 \
  --seed=100000 \
  --output_dir="$OUTPUT_DIR" \
  $WANDB_ARG \
  $RESUME_ARG \
  "$@"
