import json, re, sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
line = d["log"][1]
m = re.search(r"trials.: ([0-9]+)", line)
print("mode:", d["mode"], "| rounds:", len(d["performance_matrix"]))
print("trials in round-1 internalize:", m.group(1) if m else "?")
print("round1:", line[:170])
print("final round:", d["log"][-1][:170])
print("final_perf:", {k: round(v, 4) for k, v in d["final_performance"].items()})
print("metrics:", {k: round(v, 5) for k, v in d["metrics"].items()})
print("per_domain forgetting:", {k: round(v["forgetting_rate"], 4)
                                 for k, v in d["per_domain"].items()})
print("acq/ret:", {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in d["acquisition_retention"].items()})
