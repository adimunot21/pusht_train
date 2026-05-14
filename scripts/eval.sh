#!/usr/bin/env bash
# Unified PushT eval. Wraps lerobot-eval and runs report.py to print both
# success criteria side-by-side (the eval-criterion gotcha — lerobot issue
# #470 — made visible by design).
#
# Usage: scripts/eval.sh <checkpoint-dir-or-hub-id> [n_episodes]
#
# <checkpoint-dir> can be a local lerobot checkpoint dir (e.g.
# outputs/train/diffusion_pusht/checkpoints/200000) or an HF hub id
# (e.g. lerobot/diffusion_pusht). Both are passed verbatim to
# --policy.pretrained_path.
#
# n_episodes defaults to 100 (CI is roughly ±10pp at n=100). Use 500 to
# match the model-card sample size, with longer wall-clock.

set -euo pipefail

CKPT="${1:?usage: $0 <checkpoint-dir-or-hub-id> [n_episodes]}"
N_EPISODES="${2:-100}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pusht

TS="$(date +%Y%m%d_%H%M%S)"
SAFE_NAME="$(echo "$CKPT" | tr '/' '_')"
EVAL_OUT="${EVAL_OUT_DIR:-outputs/eval/${SAFE_NAME}_${TS}}"
mkdir -p "$EVAL_OUT"

echo "[eval] policy:     $CKPT"
echo "[eval] n_episodes: $N_EPISODES"
echo "[eval] output:     $EVAL_OUT"

lerobot-eval \
  --policy.pretrained_path="$CKPT" \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=10 \
  --output_dir="$EVAL_OUT"

python "$(dirname "$0")/report.py" "$EVAL_OUT"
