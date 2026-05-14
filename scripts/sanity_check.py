"""Pod sanity check: confirms torch sees the GPU, lerobot 0.5.x defaults are
right, and the imagenet_stats fix is wired in.

Run from any cwd inside the active conda env:
    python scripts/sanity_check.py

Prints one line per check; exits non-zero if any required invariant fails.
"""
from __future__ import annotations

import dataclasses
import sys


def main() -> int:
    import torch
    print(f"torch:           {torch.__version__}")
    print(f"cuda available:  {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available", file=sys.stderr)
        return 1
    print(f"device:          {torch.cuda.get_device_name(0)}")
    print(f"capability:      {torch.cuda.get_device_capability(0)}")

    from lerobot.configs.default import DatasetConfig
    found = False
    for f in dataclasses.fields(DatasetConfig):
        if f.name == "use_imagenet_stats":
            print(f"imagenet_stats:  default={f.default}")
            if f.default is not True:
                print("FAIL: use_imagenet_stats default should be True on "
                      "lerobot >= 0.5.0", file=sys.stderr)
                return 1
            found = True
            break
    if not found:
        print("FAIL: use_imagenet_stats field not found on DatasetConfig",
              file=sys.stderr)
        return 1

    import lerobot
    print(f"lerobot:         {lerobot.__version__}")
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
