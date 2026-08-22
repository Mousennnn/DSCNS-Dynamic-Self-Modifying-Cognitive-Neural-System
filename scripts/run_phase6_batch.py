"""v0.6.0 / Phase 6 Batch Runner.

Checks for missing conditions/seeds and runs only those.
Handles interruption gracefully.

Usage:
    python scripts/run_phase6_batch.py --rounds 450 --seeds 5
"""
from __future__ import annotations
import argparse, json, os, sys, subprocess
from typing import List

BASE_DIR = os.path.join("experiments", "phase6")
CONDITIONS = [
    "FullPolicy", "NoMemory", "FrozenPolicy", "RandomMemory",
    "ZeroMemory", "NoCredit", "NoAlternatives", "NoExploration",
    "NoOutcomeReward", "Oracle", "Random",
]


def find_missing(out_dir, seeds, conditions):
    """Find missing condition x seed combinations."""
    raw_dir = os.path.join(out_dir, "raw")
    missing = []
    for si in range(seeds):
        seed_dir = os.path.join(raw_dir, f"seed_{si}")
        for cond in conditions:
            result_file = os.path.join(seed_dir, f"{cond}_result.json")
            if not os.path.exists(result_file):
                missing.append((cond, si, 42 + si))
            else:
                # check if result is complete
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    if "rounds" not in data or data.get("rounds", 0) < 450:
                        missing.append((cond, si, 42 + si))
                except (json.JSONDecodeError, KeyError):
                    missing.append((cond, si, 42 + si))
    return missing


def main():
    ap = argparse.ArgumentParser(description="v0.6.0 batch runner")
    ap.add_argument("--rounds", type=int, default=450)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", default=BASE_DIR)
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    missing = find_missing(args.out, args.seeds, conditions)

    if not missing:
        print("All conditions complete!")
        return

    print(f"Missing {len(missing)} experiments:")
    for cond, si, seed in missing:
        print(f"  {cond} seed_{si} (seed={seed})")

    # run missing experiments one by one
    for cond, si, seed in missing:
        print(f"\nRunning {cond} seed_{si} (seed={seed})...")
        try:
            cmd = [
                sys.executable, "scripts/run_phase6.py",
                "--rounds", str(args.rounds),
                "--seeds", "1",
                "--conditions", cond,
                "--out", args.out,
            ]
            result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout per experiment
            if result.returncode != 0:
                print(f"  WARNING: {cond} seed_{si} returned exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {cond} seed_{si} (>2 hours)")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return

    # check what's still missing
    still_missing = find_missing(args.out, args.seeds, conditions)
    if still_missing:
        print(f"\nStill missing {len(still_missing)} experiments:")
        for cond, si, seed in still_missing:
            print(f"  {cond} seed_{si}")
    else:
        print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
