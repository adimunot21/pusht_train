# PushT Training Plan

Status as of 2026-05-14. lerobot 0.5.1 + gymnasium 1.3.0 + torch 2.10/cu128.
Scaffold verified end-to-end on CPU: the saved normalizer safetensors contain
exact ImageNet stats, so the load-bearing fix is wired correctly.

This document is the plan I'd run on RunPod. **Nothing in here is launched
yet** — waiting on your go-ahead per the project ground rules.

## TL;DR

| Policy | Steps | Batch | Eval-time wall-clock (RTX 4090, 1×) | RunPod cost @ $0.69/h |
|---|---:|---:|---:|---:|
| Diffusion | 200,000 | 64 | ~9–11 h (incl. in-train eval) | **$6–8** |
| ACT       |  80,000 |  8 | ~2–3 h | **$1.5–2** |
| VQ-BeT    | 250,000 | 64 | ~11–13 h | **$8–9** |
| **Total (one seed of each)** | | | ~22–27 h | **$15–19** |

These are point estimates with ±25% slack. Real first run will dial them in.

## Recommended order (budget-limited)

1. **Diffusion first.** It's the primary benchmark (model card: 65.4% / 0.955),
   it's the original Diffusion-Policy-on-PushT result, and it's the headline
   "does the ImageNet-stats fix actually reproduce the published number" test.
   Single most informative run.
2. **VQ-BeT second.** Control showing the fix does the *right thing*: VQ-BeT
   shouldn't need it (VISUAL=IDENTITY in its config), so it should hit
   ~63.8% / 0.895 without any flag. Catches accidental regressions where I
   broke something globally.
3. **ACT last.** No published `lerobot/act_pusht` on the Hub, so this run
   only proves "the fix generalizes to another ImageNet-encoder policy" —
   informative but lower priority. Cheap enough to slot in anywhere.

If you only run one: Diffusion. If only two: Diffusion + VQ-BeT.

## Expected eval numbers

Both criteria reported per `scripts/eval.sh`. The model cards report the
max-overlap criterion (issue #470 gotcha).

| Policy | `pc_success_env` (gymnasium-strict) | `pc_success_max_overlap` (≥0.95) | `avg_max_reward` | Source |
|---|---|---|---|---|
| Diffusion | unpublished (expect 30–50%) | **65.4%** (n=500) | **0.955** | [model card](https://huggingface.co/lerobot/diffusion_pusht) |
| VQ-BeT    | unpublished (expect 30–50%) | **63.8%** (n=500) | **0.895** | [model card](https://huggingface.co/lerobot/vqbet_pusht) |
| ACT       | unpublished | unpublished — guess 40–60% | guess 0.85±0.05 | no upstream PushT card; ACT-ALOHA exists but isn't comparable |

Independent reference point (from the diagnosis you handed me):
applying the inverse fix at eval-time to the published
`lerobot/diffusion_pusht` checkpoint raised its 50-episode
`pc_success` from ~0–30% up to ~50%, with `avg_max_reward` 0.93.
Our retrained-from-scratch number should match or exceed that.

A run is **healthy** if `pc_success_max_overlap` lands within ±5pp of the
published number with 100-episode CI (Wilson ±~10pp) overlapping the
target. A run is **suspect** if `avg_max_reward` is below 0.85 for
Diffusion or VQ-BeT — that suggests the encoder didn't lock onto the
visual features, which is the imagenet-stats bug surfacing in a new way.

## Wall-clock breakdown

Estimates assume a single RTX 4090 (24GB) on RunPod community cloud.
Numbers below are throughput times only; add ~10–20% for dataset download,
checkpointing, and per-25k-step in-training evals (`--eval_freq=25000`,
default `eval.n_episodes=50`).

### Diffusion (200k × batch 64)
- Model: 263M params (ResNet18 vision + UNet diffusion head).
- Expected throughput: ~5.5 it/s on 4090 (similar A100 reports range 5–7 it/s).
- Train time alone: 200,000 / 5.5 ≈ 36,400 s ≈ **10.1 h**.
- Eight in-train evals × ~6 min ≈ 50 min.
- Plus dataset download (first time) ~2 min, checkpoint saves negligible.
- **Total: 10–11 h.**
- VRAM at peak: ~9–11 GB (model + Adam state + activations + batch 64).

### ACT (80k × batch 8)
- Model: ~80M params (ResNet18 vision + small Transformer enc/dec).
- Expected throughput: ~12 it/s on 4090.
- Train time alone: 80,000 / 12 ≈ 6,700 s ≈ **1.9 h**.
- Eight in-train evals × ~5 min ≈ 40 min (faster sampler than Diffusion).
- **Total: 2.5–3 h.**
- VRAM at peak: ~4–6 GB.

### VQ-BeT (250k × batch 64)
- Model: ~120M params (ResNet18 + VQ codebook + transformer head).
- Expected throughput: ~5.5 it/s.
- Train time alone: 250,000 / 5.5 ≈ 45,500 s ≈ **12.6 h**.
- Ten in-train evals × ~5 min ≈ 50 min.
- **Total: 13–14 h.**
- VRAM at peak: ~8–10 GB.

## Policy-specific failure modes

### Diffusion
- **In-train eval is slow.** Diffusion sampling is iterative
  (default ~100 denoising steps). At `eval.n_episodes=50` each eval can take
  5–10 min, vs <1 min for transformer policies. If a run hangs at "Eval at
  step N", that's not a hang — give it 10 min.
- **EMA model is the one to eval.** lerobot's diffusion config uses EMA;
  the checkpoint at `pretrained_model/model.safetensors` is the EMA model
  (not the live model). Don't be surprised that training loss is higher
  than eval-from-checkpoint loss.
- **OOM on <16GB VRAM** at batch 64. The dry-run hit OOM on the local
  GTX 1650 (4GB) at batch 2, *just from Adam state init*. Don't try this
  on a T4. RTX 4090 (24GB) has comfortable headroom.

### ACT
- **PushT action dim is 2.** ACT was designed around higher-DOF bimanual
  manipulation with action chunks of 100. PushT will work but the chunked-action
  prior is mostly wasted. Don't expect ACT to beat Diffusion here.
- **No upstream baseline** to compare against. If ACT plateaus at, say, 30%
  `pc_success_max_overlap`, we can't tell if it's a bug or just the policy's
  natural ceiling on this task. Treat this run as qualitative only.

### VQ-BeT
- **Two-phase training** (VQ codebook warmup, then policy) happens inside
  `lerobot-train` and adds initial overhead — the first few thousand steps
  log low it/s before stabilizing. Don't kill the run.
- **VISUAL=IDENTITY** means raw [0,1] RGB is fed to the encoder. This is
  intentional in the VQ-BeT config and is the reason
  `--dataset.use_imagenet_stats` is omitted from the train script. The
  control claim only holds if `lerobot 0.5.1`'s
  `policies/vqbet/configuration_vqbet.py` still has
  `normalization_mapping["VISUAL"] = IDENTITY` — confirmed today,
  re-check before each retry if lerobot is upgraded.

## What I will save, per run

- `outputs/train/<policy>_pusht/checkpoints/<step>/pretrained_model/` —
  model + preprocessor (`*_normalizer_processor.safetensors`) + config.
- `outputs/train/<policy>_pusht/checkpoints/<step>/training_state/` —
  optimizer + scheduler + RNG (for `--resume=true`).
- `outputs/train/<policy>_pusht/train.log` — full stdout.
- Final eval at end of training: `outputs/eval/<policy>_pusht_<ts>/eval_info.json`.

All of the above is gitignored (large) but explicitly preserved on the
RunPod volume. The plan: when a run finishes, `rsync` the checkpoints
directory back to local before stopping the pod.

## Open questions for you before launch

1. **Hub push.** Default off. Want me to flip it on with
   `HF_REPO_ID=adimunot/<policy>_pusht ./scripts/train_<policy>.sh` so
   each finished run auto-uploads to your HF account? Otherwise we just
   keep checkpoints on the pod/local.
2. **wandb.** Same gating: `WANDB_API_KEY` env var enables it. Off by
   default. Want it on for the real runs?
3. **n_episodes for the final reproduction eval.** Plan uses 100
   (CI ≈ ±10pp). The model cards used 500 (CI ≈ ±4pp). Quintupling the
   episode count quintuples eval wall-clock, but for Diffusion that's
   ~5 min × 5 = 25 min — basically free given the 10h training time.
   Recommend bumping to 500 for the final report-out, 100 for any
   intermediate sanity-checks. Confirm?
4. **Seeds.** Plan above is one seed each (matches the model cards). If
   you want error bars on the reproduction, 3 seeds per policy ≈ 3× the
   above costs. Recommend 1 seed first, then add seeds only if Diffusion
   doesn't reproduce cleanly.
