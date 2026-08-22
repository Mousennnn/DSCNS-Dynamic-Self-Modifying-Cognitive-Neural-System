"""Generate checkpoint and release manifests for v0.6.0."""
import json, os, hashlib

base = "experiments/phase6"
conds = ["FullPolicy","NoMemory","FrozenPolicy","RandomMemory","ZeroMemory",
         "NoCredit","NoAlternatives","NoExploration","NoOutcomeReward","Oracle","Random"]

manifest = {
    "version": "v0.6.0",
    "phase": "P6",
    "total_conditions": len(conds),
    "seeds_per_condition": 5,
    "seed_values": [42,43,44,45,46],
    "rounds": 450,
    "conditions": conds,
    "best_checkpoints": [],
    "final_checkpoints": [],
    "relay_checkpoints": [],
}

for cond in conds:
    for si in range(5):
        seed = 42 + si
        ckpt_base = os.path.join(base, "checkpoints", cond, f"seed_{seed}")
        
        best_meta = os.path.join(ckpt_base, "best", "metadata.json")
        best_pt = os.path.join(ckpt_base, "best", "model.pt")
        if os.path.exists(best_meta):
            with open(best_meta) as f:
                meta = json.load(f)
            sha = ""
            if os.path.exists(best_pt):
                h = hashlib.sha256()
                with open(best_pt, "rb") as fp:
                    for chunk in iter(lambda: fp.read(8192), b""):
                        h.update(chunk)
                sha = h.hexdigest()
            manifest["best_checkpoints"].append({
                "condition": cond, "seed": seed,
                "score": meta.get("score", 0),
                "round": meta.get("round", 0),
                "sha256": sha, "path": best_meta,
            })
        
        final_meta = os.path.join(ckpt_base, "final", "metadata.json")
        if os.path.exists(final_meta):
            with open(final_meta) as f:
                meta = json.load(f)
            manifest["final_checkpoints"].append({
                "condition": cond, "seed": seed,
                "round": meta.get("round", 0),
                "path": final_meta,
            })
        
        relay_lineage = os.path.join(base, "relay", cond, f"seed_{seed}",
                                       "relay_v0.6.0_stage_450", "lineage.json")
        if os.path.exists(relay_lineage):
            with open(relay_lineage) as f:
                lineage = json.load(f)
            manifest["relay_checkpoints"].append({
                "condition": cond, "seed": seed,
                "round": lineage.get("continued_rounds", 450),
                "total_lineage_rounds": lineage.get("total_lineage_rounds", 450),
                "path": relay_lineage,
            })

print(f"Best: {len(manifest['best_checkpoints'])}")
print(f"Final: {len(manifest['final_checkpoints'])}")
print(f"Relay: {len(manifest['relay_checkpoints'])}")

with open(os.path.join(base, "checkpoints", "MANIFEST_v060.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("Written to experiments/phase6/checkpoints/MANIFEST_v060.json")

# Also create release manifest
release = {
    "version": "v0.6.0",
    "phase": "P6",
    "tag": "v0.6.0",
    "experiment_protocol": {
        "conditions": len(conds),
        "seeds": 5,
        "rounds_per_seed": 450,
        "total_rounds": len(conds) * 5 * 450,
    },
    "checkpoints": {
        "best": len(manifest["best_checkpoints"]),
        "final": len(manifest["final_checkpoints"]),
        "relay": len(manifest["relay_checkpoints"]),
    },
    "acceptance_criteria": {
        "minimum_pass": "PASS",
        "mechanism_pass": "PASS",
        "strong_pass": "PASS",
        "full_pass": "FAIL",
    },
    "evidence_matrix": {
        "experience_to_policy": "SUPPORTED",
        "policy_to_target": "SUPPORTED",
        "policy_to_magnitude": "SUPPORTED",
        "modification_to_outcome": "SUPPORTED",
        "outcome_to_credit": "PARTIAL",
        "credit_to_policy": "NOT_ESTABLISHED",
        "full_closed_loop": "NOT_ESTABLISHED",
    },
    "documentation": {
        "readme": True,
        "phase6_spec": True,
        "evidence_matrix": True,
        "relay_doc": True,
        "inference_doc": True,
        "figures_readme": True,
        "report": True,
    },
}

with open("RELEASE_MANIFEST_v0.6.0.json", "w") as f:
    json.dump(release, f, indent=2)
print("Written to RELEASE_MANIFEST_v0.6.0.json")
