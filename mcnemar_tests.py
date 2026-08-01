#!/usr/bin/env python3
"""Paired McNemar tests + minimum detectable effects for the checker ablation.

The original five configurations checked identical row sets with identical
ground-truth labels per seed (verified below), so per-row verdicts are paired.
For each comparison and stratum we count discordant pairs (b: only A correct;
c: only B correct) and report the exact two-sided McNemar p-value (binomial
test on b successes in b+c trials at p=0.5). The MDE column reports the
approximate smallest difference in paired proportions detectable at
alpha=0.05, power=0.80 given the observed discordance rate:
    MDE ~= (z_a/2 + z_b) * sqrt(psi / N),  psi = (b+c)/N.

Strata: recall (ground-truth-bad rows; "correct" = flagged NOT_OK) and
false-positive behavior (clean rows; "correct" = not flagged).

Usage: python mcnemar_tests.py [runs_dir]
"""
import glob
import json
import math
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
CONFIGS = ["circular", "reflexion", "rules-only", "cross-only", "hybrid"]
COMPARISONS = [
    ("circular", "reflexion"),
    ("circular", "cross-only"),
    ("hybrid", "rules-only"),
    ("hybrid", "cross-only"),
    ("circular", "hybrid"),
]


def load(ds):
    """rows[(seed, idx)] = (gt_bad, {config: flagged})  -- verifying pairing."""
    rows = {}
    for cfg in CONFIGS:
        for f in sorted(glob.glob(f"{RUNS}/{ds}_{cfg}_seed*_final_report.json")):
            seed = int(f.split("seed")[1].split("_")[0])
            for r in json.load(open(f))["row_level_results"]:
                key = (seed, r["row_idx"])
                gt, flagged = bool(r["ground_truth_bad"]), r["verdict"] == "NOT_OK"
                if key not in rows:
                    rows[key] = (gt, {})
                assert rows[key][0] == gt, f"ground-truth mismatch at {ds} {key}"
                rows[key][1][cfg] = flagged
    for key, (gt, v) in rows.items():
        assert set(v) == set(CONFIGS), f"missing config at {ds} {key}"
    return rows


def exact_binom_two_sided(b, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(b, n - b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mde(psi, n):
    return (1.96 + 0.8416) * math.sqrt(psi / n) if n and psi > 0 else float("nan")


for ds in ("uci", "kb"):
    rows = load(ds)
    print(f"\n=== {ds} (N rows: {len(rows)}; positives: {sum(1 for g, _ in rows.values() if g)}) ===")
    for a, bcfg in COMPARISONS:
        for stratum, keep, correct in (
            ("recall", True, lambda fl: fl),
            ("FP", False, lambda fl: not fl),
        ):
            sel = [(v[a], v[bcfg]) for g, v in rows.values() if g == keep]
            n = len(sel)
            b = sum(1 for fa, fb in sel if correct(fa) and not correct(fb))
            c = sum(1 for fa, fb in sel if correct(fb) and not correct(fa))
            p = exact_binom_two_sided(min(b, c), b + c)
            d = mde((b + c) / n if n else 0, n)
            sig = "*" if p == p and p < 0.05 else " "
            print(f"  {a:>9} vs {bcfg:<10} [{stratum:6}] N={n:3} b={b:2} c={c:2} "
                  f"exact-McNemar p={p:.4f}{sig} MDE~{d:.3f}")
