#!/usr/bin/env python3
"""
threshold_sensitivity.py
========================
Reproduces the threshold sensitivity analysis (Table 7 in the paper).

The 2-generation lead time should hold across a range of detection thresholds.
This script does NOT re-run training; it post-processes the metric-vs-generation
series already produced by run_experiment.py and reports, for each threshold
T in {5%, 10%, 15%, 20%}, the generation at which Distinct-1 first drops by T
and perplexity first rises by T.

USAGE
    python experiments/threshold_sensitivity.py --input results/all_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from analyze import load_one_file, mean_by_generation, first_crossing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")

    all_runs = load_one_file(args.input)
    for name, panels in all_runs.items():
        d1 = mean_by_generation(panels, "distinct_1")
        ppl = mean_by_generation(panels, "perplexity")
        if not d1 or not ppl:
            print(f"[{name}] missing distinct_1 or perplexity series; skipping.")
            continue
        print(f"\n=== {name} ===")
        print(f"{'Threshold':>10} {'D-1 det.':>10} {'PPL det.':>10} {'Lead':>6}")
        for T in args.thresholds:
            d = first_crossing(d1, "drop", T)
            p = first_crossing(ppl, "rise", T)
            lead = (p - d) if (d is not None and p is not None) else None
            print(f"{int(T*100):>9}% {('Gen '+str(d)) if d is not None else '--':>10} "
                  f"{('Gen '+str(p)) if p is not None else '--':>10} "
                  f"{('+'+str(lead)) if lead is not None else '--':>6}")
        print("\nReading: if 'Lead' column is consistently positive across thresholds,")
        print("the diversity-before-perplexity ordering is robust (paper's central claim).")


if __name__ == "__main__":
    main()
