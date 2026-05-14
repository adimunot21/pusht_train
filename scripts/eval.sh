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

command -v lerobot-eval >/dev/null || {
  echo "[eval] lerobot-eval not on PATH. Activate the env first." >&2
  exit 1
}

TS="$(date +%Y%m%d_%H%M%S)"
SAFE_NAME="$(echo "$CKPT" | tr '/' '_')"
EVAL_OUT="${EVAL_OUT_DIR:-outputs/eval/${SAFE_NAME}_${TS}}"
mkdir -p "$EVAL_OUT"

# lerobot 0.5.x's draccus parser needs --policy.type alongside
# --policy.pretrained_path to know which policy class to instantiate.
# Auto-detect from the checkpoint's config.json. For hub-id checkpoints,
# the user can override via POLICY_TYPE=... ./scripts/eval.sh ...
POLICY_TYPE="${POLICY_TYPE:-}"
if [ -z "$POLICY_TYPE" ] && [ -f "$CKPT/config.json" ]; then
  POLICY_TYPE=$(python -c "import json; print(json.load(open('$CKPT/config.json')).get('type',''))" 2>/dev/null || true)
fi
if [ -z "$POLICY_TYPE" ]; then
  echo "[eval] ERROR: could not detect policy type from $CKPT/config.json." >&2
  echo "[eval] Set POLICY_TYPE explicitly: POLICY_TYPE=diffusion $0 $CKPT $N_EPISODES" >&2
  exit 1
fi

echo "[eval] policy:     $CKPT"
echo "[eval] policy.type: $POLICY_TYPE"
echo "[eval] n_episodes: $N_EPISODES"
echo "[eval] output:     $EVAL_OUT"

lerobot-eval \
  --policy.type="$POLICY_TYPE" \
  --policy.pretrained_path="$CKPT" \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=10 \
  --output_dir="$EVAL_OUT"

python "$(dirname "$0")/report.py" "$EVAL_OUT"
