#!/usr/bin/env python3
"""
analyze.py
==========
Reads the experiment results JSON and computes the quantities reported in:
  - Table 3/4: per-generation metric panel
  - Table 5:   detection generation for each metric, and lead time over PPL

Also prints LaTeX-ready row strings you can paste straight into the paper.

USAGE
    python experiments/analyze.py --input results/all_results.json
    python experiments/analyze.py --input results/gpt2_replace_seed42.json --tier1 0.10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List


# --------------------------------------------------------------------------- #
def load_one_file(path: str) -> Dict[str, Dict[int, Dict]]:
    """
    Returns a dict { experiment_name: { generation: metric_panel } }.
    Supports both the multi-seed all_results.json format and per-run JSON.
    """
    with open(path) as f:
        data = json.load(f)

    out: Dict[str, Dict[int, Dict]] = {}
    # Heuristic: if top-level keys look like experiment names, descend one level
    if all(isinstance(v, dict) for v in data.values()) and \
       any(isinstance(vv, dict) and any(k.isdigit() or isinstance(k, int) for k in vv) for vv in data.values()):
        for exp_name, gens in data.items():
            out[exp_name] = {int(g): m for g, m in gens.items()}
    else:
        # Flat: dict keyed by generation
        out[os.path.basename(path)] = {int(g): m for g, m in data.items()}
    return out


def mean_by_generation(panels: Dict[int, Dict], key: str) -> Dict[int, float]:
    """For multi-seed combined panels where each value is already a mean (or a list)."""
    out = {}
    for g, m in panels.items():
        v = m.get(key)
        if v is None:
            continue
        if isinstance(v, list):
            out[g] = mean(v)
        else:
            out[g] = float(v)
    return out


def first_crossing(series: Dict[int, float], direction: str = "drop", frac: float = 0.10):
    """First generation crossing baseline*(1±frac), or None."""
    if 0 not in series:
        return None
    base = series[0]
    for g in sorted(series):
        if g == 0:
            continue
        if direction == "drop" and series[g] <= base * (1 - frac):
            return g
        if direction == "rise" and series[g] >= base * (1 + frac):
            return g
    return None


# --------------------------------------------------------------------------- #
def report_one(name: str, panels: Dict[int, Dict], tier1: float, tier2: float):
    d1 = mean_by_generation(panels, "distinct_1")
    ttr = mean_by_generation(panels, "ttr")
    ent = mean_by_generation(panels, "entropy")
    ppl = mean_by_generation(panels, "perplexity")
    vocab = mean_by_generation(panels, "vocab_size")
    if not d1:
        print(f"\n[{name}] no distinct_1 values; skipping.")
        return

    print(f"\n=== {name} ===")
    print(f"{'Gen':>3} {'D-1':>7} {'TTR':>7} {'Entropy':>8} {'PPL':>7} {'Vocab':>7} {'dD-1':>8} {'dPPL':>8}")
    base_d1 = d1.get(0)
    base_ppl = ppl.get(0)
    for g in sorted(d1):
        dd = "" if base_d1 is None else f"{(d1[g]-base_d1)/base_d1*100:+.1f}%"
        dp = "" if base_ppl is None or g not in ppl else f"{(ppl[g]-base_ppl)/base_ppl*100:+.1f}%"
        print(f"{g:>3} {d1.get(g, float('nan')):>7.3f} {ttr.get(g, float('nan')):>7.3f} "
              f"{ent.get(g, float('nan')):>8.2f} {ppl.get(g, float('nan')):>7.2f} "
              f"{vocab.get(g, float('nan')):>7.0f} {dd:>8} {dp:>8}")

    d1_det = first_crossing(d1, "drop", tier1)
    ttr_det = first_crossing(ttr, "drop", tier1)
    ent_det = first_crossing(ent, "drop", tier1)
    ppl_det = first_crossing(ppl, "rise", tier2)
    print(f"\nDetection timing  (Tier 1 = -{int(tier1*100)}%, Tier 2 = +{int(tier2*100)}%):")
    print(f"  Distinct-1 detects at: {'Gen '+str(d1_det) if d1_det is not None else 'within horizon: no'}")
    print(f"  TTR        detects at: {'Gen '+str(ttr_det) if ttr_det is not None else 'within horizon: no'}")
    print(f"  Entropy    detects at: {'Gen '+str(ent_det) if ent_det is not None else 'within horizon: no'}")
    print(f"  Perplexity detects at: {'Gen '+str(ppl_det) if ppl_det is not None else 'within horizon: no'}")
    if d1_det is not None and ppl_det is not None:
        lead = ppl_det - d1_det
        print(f"  Lead time (diversity before perplexity): {lead} generation(s)")
        if lead >= 1:
            print("  => Ordering replicated: diversity precedes perplexity.")

    # LaTeX rows (paste directly into the Springer paper tables)
    print("\nLaTeX rows for Table 4/X (Gen, D-1, TTR, Entropy, PPL, Vocab):")
    for g in sorted(d1):
        print(f"  {g} & {d1.get(g, float('nan')):.3f} & {ttr.get(g, float('nan')):.3f} & "
              f"{ent.get(g, float('nan')):.2f} & {ppl.get(g, float('nan')):.2f} & "
              f"{vocab.get(g, float('nan')):.0f} \\\\")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON file or glob")
    ap.add_argument("--tier1", type=float, default=0.10, help="diversity drop threshold")
    ap.add_argument("--tier2", type=float, default=0.10, help="perplexity rise threshold")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input)) or [args.input]
    for f in files:
        if not os.path.exists(f):
            print(f"skip (not found): {f}")
            continue
        all_runs = load_one_file(f)
        for name, panels in all_runs.items():
            report_one(name, panels, args.tier1, args.tier2)


if __name__ == "__main__":
    main()
