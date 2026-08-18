"""Phase 5 negative controls (design report section 10.2).

Three arms inject *non-intrinsic* deltas through the exact same
trigger -> apply -> validate -> accept/rollback protocol:

  * random   -- Control A: gaussian delta with the same total norm as a
                reference intrinsic delta (per network, cached on first
                trigger);
  * constant -- Control B: the *same* delta is applied at every trigger
                (state-independent; delta variance across inputs is 0 by
                construction);
  * shuffled -- Control C: the delta is generated from a *different* input
                batch than the one used for validation (state<->delta
                pairing is shuffled).

If P5's closed loop is genuinely state-dependent, intrinsic deltas should
be distinguishable from these controls (e.g. in delta/behavior statistics
and in the pass rates of the safety validation).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import make_config, prepare_data, save_results
from dscns.utils import set_seed
from phase5_common import make_phase5_stream
from run_phase5_b import run_arm


def shuffled_texts_fn(batch, r, net, data):
    """Return texts from a domain *different* from the trigger batch's."""
    rng = np.random.RandomState(r * 1000 + hash(net.id) % 1000)
    domain = batch[0]["domain"]
    others = [d for d in data["train"].keys() if d != domain]
    other_domain = others[rng.randint(len(others))]
    pool = data["train"][other_domain]
    k = min(8, len(pool))
    return [str(t) for t in rng.choice(pool, size=k, replace=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="random,constant,shuffled")
    args = ap.parse_args()

    config = make_config(cfg_path="config/phase5.yaml")
    config.seed = args.seed
    set_seed(config.seed)
    data = prepare_data(config)
    eval_sets = data["eval"]
    rng = np.random.RandomState(config.seed)
    stream = make_phase5_stream(config, data, rng)

    all_results = {}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    specs = [("random", "random", None, None),
             ("constant", "constant", None, None),
             ("shuffled", "shuffled", None, shuffled_texts_fn)]
    for tag, mode, cdelta, other_fn in specs:
        if tag not in arms:
            continue
        all_results[tag] = run_arm(tag, mode, config, data, eval_sets,
                                   stream, constant_delta=cdelta,
                                   other_texts_fn=other_fn)
        save_results(args.out, tag, all_results[tag])
    save_results(args.out, "controls_summary", all_results)

    print("=== Phase 5 negative controls ===")
    for tag, res in all_results.items():
        cl = res.get("closed_loop", {})
        print(f"{tag}: triggers={res['triggers']} "
              f"accept={res['acceptance_rate']:.2f}"
              f" | delta_norm {cl.get('delta_norm_mean', float('nan')):.5f}"
              f" var={cl.get('delta_norm_variance', float('nan')):.2e}"
              f" | pred_change={cl.get('pred_change_mean', float('nan')):.4f}"
              f" | AF={res['AF']:.4f} CLS={res['CLS']:.4f}")


if __name__ == "__main__":
    main()
