#!/usr/bin/env bash
# Full pipeline: train Diffusion -> 500-ep eval -> optionally stop the pod.
#
# Designed to be launched once and walked away from:
#     nohup ./scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
#     disown
#
# Behaviour:
#   - Always runs training, then (if training succeeded) the 500-ep eval.
#   - If BOTH RUNPOD_API_KEY and RUNPOD_POD_ID are set, sends a podStop
#     to the RunPod GraphQL API after eval finishes successfully.
#       * Stopping preserves the /workspace volume; only the GPU is
#         released and the GPU-hour billing halts.
#       * Without those env vars set, the pod is left running so you can
#         inspect or rerun manually.
#   - On training or eval failure, the pod is intentionally LEFT RUNNING
#     so you can debug. Failure of auto-stop itself is logged but never
#     overrides the eval result.

set -uo pipefail

cd /workspace/pusht_train
mkdir -p logs

ts() { date -Iseconds; }

echo "[pipeline] $(ts)  start"

# ----- TRAIN -----------------------------------------------------------------
echo "[pipeline] $(ts)  training: ./scripts/train_diffusion.sh"
./scripts/train_diffusion.sh > logs/train_diffusion.log 2>&1
TRAIN_EXIT=$?
echo "[pipeline] $(ts)  training exited with code $TRAIN_EXIT"
if [ "$TRAIN_EXIT" -ne 0 ]; then
  echo "[pipeline] training failed. Pod NOT stopped — inspect logs/train_diffusion.log and decide manually."
  exit 10
fi

# ----- EVAL ------------------------------------------------------------------
CKPT="$(realpath outputs/train/diffusion_pusht/checkpoints/last/pretrained_model)"
echo "[pipeline] $(ts)  eval (500 episodes): ./scripts/eval.sh $CKPT 500"
./scripts/eval.sh "$CKPT" 500 > logs/eval_diffusion_500.log 2>&1
EVAL_EXIT=$?
echo "[pipeline] $(ts)  eval exited with code $EVAL_EXIT"
if [ "$EVAL_EXIT" -ne 0 ]; then
  echo "[pipeline] eval failed. Pod NOT stopped — inspect logs/eval_diffusion_500.log."
  exit 20
fi

# Print the report.py tail so 'tail logs/pipeline.log' shows the final numbers.
echo "[pipeline] $(ts)  final eval summary:"
tail -n 25 logs/eval_diffusion_500.log

# ----- STOP POD (optional) ---------------------------------------------------
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[pipeline] $(ts)  stopping pod $RUNPOD_POD_ID via RunPod API..."
  resp=$(curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H "content-type: application/json" \
    -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" 2>&1) || true
  echo "[pipeline] API response: $resp"
  if echo "$resp" | grep -q '"desiredStatus":"EXITED"'; then
    echo "[pipeline] $(ts)  pod stop accepted; GPU billing will halt shortly."
  else
    echo "[pipeline] $(ts)  WARNING: pod stop response did NOT confirm EXITED."
    echo "[pipeline] You will need to stop the pod manually from the RunPod console."
  fi
else
  echo "[pipeline] $(ts)  RUNPOD_API_KEY and/or RUNPOD_POD_ID not set; pod left running."
fi

echo "[pipeline] $(ts)  done"
