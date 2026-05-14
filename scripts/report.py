#!/usr/bin/env python
"""Report PushT eval metrics with both success criteria and 95% Wilson CIs.

LeRobot's `pc_success` in eval_info.json uses gymnasium's `is_success` flag,
which requires SUSTAINED overlap of the block with the target. The model
cards (e.g. lerobot/diffusion_pusht) report a different criterion: the
fraction of episodes where the per-episode MAX overlap reaches >= 0.95.
These two metrics produce different numbers for the same rollouts.

This script reports BOTH from a single eval, so the criterion gap
(lerobot issue #470) is visible at a glance.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def find_eval_json(d: Path) -> Path:
    direct = d / "eval_info.json"
    if direct.is_file():
        return direct
    candidates = sorted(d.rglob("eval_info.json"))
    if not candidates:
        raise SystemExit(f"No eval_info.json found under {d}")
    return candidates[0]


def coerce_success(value) -> bool:
    """Eval JSON stores success as bool, 0/1, or list-per-step. Reduce to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, Iterable):  # e.g. per-step success list -> any()
        return any(bool(v) for v in value)
    return False


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: report.py <eval-output-dir>", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(sys.argv[1])
    json_path = find_eval_json(out_dir)
    data = json.loads(json_path.read_text())

    per_episode = data.get("per_episode") or []
    if not per_episode:
        raise SystemExit(f"per_episode is empty in {json_path}")

    n = len(per_episode)
    max_rewards: list[float] = []
    n_success_env = 0
    n_success_max = 0
    for ep in per_episode:
        r = ep.get("max_reward")
        if r is None:
            continue
        max_rewards.append(float(r))
        if coerce_success(ep.get("success")):
            n_success_env += 1
        if r >= 0.95:
            n_success_max += 1

    if not max_rewards:
        raise SystemExit(f"No max_reward values found in {json_path}")

    avg_max = sum(max_rewards) / len(max_rewards)
    pc_env = n_success_env / n
    pc_max = n_success_max / n
    ci_env = wilson_ci(n_success_env, n)
    ci_max = wilson_ci(n_success_max, n)

    aggregated = data.get("aggregated") or {}
    reported_pc = aggregated.get("pc_success")
    reported_avg = aggregated.get("avg_max_reward")

    bar = "=" * 64
    print(f"\n{bar}")
    print(f" PushT eval — {n} episodes")
    print(f" source: {json_path}")
    print(bar)
    print(f"  pc_success_env         (gymnasium strict, sustained):  "
          f"{pc_env:>7.1%}   95% CI [{ci_env[0]:.1%}, {ci_env[1]:.1%}]")
    print(f"  pc_success_max_overlap (model-card, max>=0.95):         "
          f"{pc_max:>7.1%}   95% CI [{ci_max[0]:.1%}, {ci_max[1]:.1%}]")
    print(f"  avg_max_reward:                                         "
          f"{avg_max:>7.3f}")
    if reported_pc is not None or reported_avg is not None:
        print()
        print(f"  cross-check vs aggregated in eval_info.json:")
        if reported_pc is not None:
            # lerobot stores pc_success as a percentage (already *100).
            print(f"    aggregated.pc_success:    {reported_pc:>7.3f} (== "
                  f"{pc_env * 100:.3f} computed)")
        if reported_avg is not None:
            print(f"    aggregated.avg_max_reward: {reported_avg:>6.3f} (== "
                  f"{avg_max:.3f} computed)")
    print(bar)


if __name__ == "__main__":
    main()
