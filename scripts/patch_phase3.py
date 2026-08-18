"""Patch phase3 JSONs: add final_mean / code_delta derived fields."""
import json
import os
import sys

import numpy as np

DOMAINS = ["general", "math", "logic", "code", "science"]
base = "experiments/phase3"
for tag in ["fixed", "evolve"]:
    p = os.path.join(base, f"{tag}.json")
    if not os.path.exists(p):
        continue
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    pm = np.asarray(r["performance_matrix"])
    r["final_mean"] = float(np.mean(pm[-1]))
    r["code_adapt_speed"] = {
        "round4": float(pm[3][DOMAINS.index("code")]),
        "round7": float(pm[6][DOMAINS.index("code")]),
        "delta": float(pm[6][DOMAINS.index("code")] - pm[3][DOMAINS.index("code")]),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, default=str)
    print("patched", tag, "| code delta:", round(r["code_adapt_speed"]["delta"], 4),
          "| final_mean:", round(r["final_mean"], 4))
