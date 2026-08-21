"""Batch runner for v0.5.3 — runs only missing condition/seed combos.

Usage: D:\Anaconda3\envs\rtdetr\python.exe scripts/run_v053_batch.py
"""
from __future__ import annotations
import os, sys, subprocess, json

PYTHON = r"D:\Anaconda3\envs\rtdetr\python.exe"
BASE_DIR = os.path.join("experiments", "phase5_3_v053")
ROUND_SCRIPT = os.path.join("scripts", "run_v053.py")
CONDITIONS = ["FullPolicy", "NoMemory", "FrozenPolicy", "RandomMemory",
              "ZeroMemory", "NoCredit", "NoAlternatives", "NoExploration"]
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 450


def is_seed_complete(condition: str, seed_idx: int) -> bool:
    path = os.path.join(BASE_DIR, "raw", f"seed_{seed_idx}", f"{condition}_result.json")
    return os.path.exists(path) and os.path.getsize(path) > 100


def main():
    missing = []
    for cond in CONDITIONS:
        for si, seed in enumerate(SEEDS):
            if not is_seed_complete(cond, si):
                missing.append((cond, si, seed))

    if not missing:
        print("All conditions/seeds complete!")
        return

    print(f"Missing {len(missing)} condition/seed combos:")
    for cond, si, seed in missing:
        print(f"  {cond} seed_{si} (seed={seed})")
    print()

    # Group by condition
    cond_groups = {}
    for cond, si, seed in missing:
        cond_groups.setdefault(cond, []).append((si, seed))

    for cond, seeds in cond_groups.items():
        print(f"\n{'='*60}")
        print(f"Running {cond} ({len(seeds)} seeds)")
        print(f"{'='*60}")

        # Run all seeds for this condition at once
        # The script iterates seeds internally, but completed ones will
        # be overwritten with identical results (same seed = deterministic)
        seed_list = ",".join(str(s) for _, s in seeds)
        cmd = [PYTHON, ROUND_SCRIPT,
               "--rounds", str(ROUNDS),
               "--seeds", str(len(SEEDS)),  # run all 5
               "--conditions", cond]
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"  WARNING: exit code {result.returncode}")
        else:
            # Verify completion
            for si, seed in seeds:
                if is_seed_complete(cond, si):
                    print(f"  OK {cond} seed_{si} complete")
                else:
                    print(f"  MISSING {cond} seed_{si} still missing!")

    # Final verification
    print(f"\n{'='*60}")
    print("FINAL VERIFICATION")
    print(f"{'='*60}")
    all_complete = True
    for cond in CONDITIONS:
        n_done = sum(1 for si in range(len(SEEDS)) if is_seed_complete(cond, si))
        status = "OK" if n_done == len(SEEDS) else "INCOMPLETE"
        print(f"  {status} {cond}: {n_done}/{len(SEEDS)} seeds")
        if n_done < len(SEEDS):
            all_complete = False

    if all_complete:
        print("ALL EXPERIMENTS COMPLETE")
    else:
        print("SOME EXPERIMENTS STILL MISSING")


if __name__ == "__main__":
    main()
