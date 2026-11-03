#!/usr/bin/env python3
"""Cost-weighted comparison of checker configurations (paper, Section 6,
"Cost-weighted comparison" and the rho* column of the ablation table).

For every configuration X and the reference hybrid H (rules + cross), with
c_FN the cost of a missed error and c_FP the cost of an unnecessary review,
the expected cost on the checked rows is

    cost(X) = c_FN * FN_X + c_FP * FP_X .

X is cheaper than H exactly when

    c_FN / c_FP  >  rho* = (FP_X - FP_H) / (FN_H - FN_X).

rho* is reported per dataset. "dominated" marks configurations with
FN >= FN_H and FP >= FP_H (never preferable); "< r" marks configurations
with fewer FP but more FN than H, preferable only when c_FN/c_FP < r. The two
model-free reference rows are included: flag-everything (every row NOT_OK)
and the two-input comparator, which on these tiers equals the oracle
(FN = FP = 0 by construction).

Usage:  python cost_breakeven.py [runs_dir]      (default: runs)
"""
import json
import os
import sys

MODES = ["circular", "reflexion", "rules-only", "cross-only", "hybrid", "hybrid-circular"]
LABELS = {
    "flag-everything": "Flag-everything (reference)",
    "comparator": "Two-input comparator (reference)",
    "circular": "Circular (same-family LLM)",
    "reflexion": "Reflexion (self-critique)",
    "rules-only": "Deterministic rules only",
    "cross-only": "Cross-architecture only",
    "hybrid": "Hybrid (rules + cross)",
    "hybrid-circular": "Hybrid (rules + same-family)",
}
DATASETS = [("UCI Adult", "uci"), ("KramaBench", "kb")]
SEEDS = (1, 2, 3)


def confusion(runs, prefix, mode):
    tp = fp = fn = tn = 0
    for s in SEEDS:
        path = os.path.join(runs, f"{prefix}_{mode}_seed{s}_final_report.json")
        with open(path) as f:
            report = json.load(f)
        for row in report["row_level_results"]:
            flag = row["verdict"] == "NOT_OK"
            bad = bool(row["ground_truth_bad"])
            tp += flag and bad
            fp += flag and not bad
            fn += (not flag) and bad
            tn += (not flag) and not bad
    return tp, fp, fn, tn


def main(runs="runs"):
    for name, prefix in DATASETS:
        res = {m: confusion(runs, prefix, m) for m in MODES}
        pos = res["hybrid"][0] + res["hybrid"][2]
        neg = res["hybrid"][1] + res["hybrid"][3]
        res["flag-everything"] = (pos, neg, 0, 0)
        res["comparator"] = (pos, 0, 0, neg)
        _, fp_h, fn_h, _ = res["hybrid"]
        print(f"\n== {name}  (reference: hybrid rules+cross, FN={fn_h}, FP={fp_h})")
        print(f"{'configuration':34s} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}   rho* = c_FN/c_FP")
        for m in ["flag-everything", "comparator"] + MODES:
            tp, fp, fn, tn = res[m]
            d_fn, d_fp = fn_h - fn, fp - fp_h
            # X cheaper than H  <=>  c_FN*(-d_fn) + c_FP*d_fp < 0
            if m == "hybrid":
                rho = "(ref.)"
            elif d_fn <= 0 and d_fp >= 0:
                rho = "dominated (never cheaper)"
            elif d_fn >= 0 and d_fp <= 0:
                rho = "0 (dominates hybrid)"
            elif d_fn > 0:            # more FP, fewer FN: cheaper above rho*
                rho = f"> {d_fp / d_fn:.2f}"
            else:                     # fewer FP, more FN: cheaper below rho*
                rho = f"< {(-d_fp) / (-d_fn):.2f}"
            print(f"{LABELS[m]:34s} {tp:3d} {fp:3d} {fn:3d} {tn:3d}   {rho}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
