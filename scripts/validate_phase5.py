"""Phase 5 validation orchestration.

Runs the core validation suite (Tests 1-6 + P5-A modulation) and the
negative-control suite (random / constant / shuffled deltas), then writes
a combined report to experiments/phase5/validation.json.

Run:  python scripts/validate_phase5.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.test_phase5_core as core
import tests.test_phase5_validation as val


def main():
    t0 = time.time()
    core_results, core_ok = core.run_all()
    val_results, val_ok = val.run_all()
    report = {
        "core_tests": core_results,
        "negative_controls": val_results,
        "core_ok": core_ok,
        "controls_ok": val_ok,
        "overall_ok": core_ok and val_ok,
        "wall_seconds": time.time() - t0,
    }
    out_dir = os.path.join("experiments", "phase5")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("=== PHASE 5 VALIDATION:", "PASS" if report["overall_ok"] else "FAIL",
          f"({report['wall_seconds']:.0f}s) ===")
    sys.exit(0 if report["overall_ok"] else 1)


if __name__ == "__main__":
    main()
