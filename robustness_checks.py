#!/usr/bin/env python3
"""Row-level checks on the ablation reports that the pooled table cannot show.

  1. Channel overlap measured on row ids (not inferred as |A|+|B|-|hybrid|).
  2. Whether the unscored rows of the same-family configurations are the same rows.
  3. Discrimination: flag rate on bad vs. clean rows, exact Fisher test.
  4. Label sensitivity: primary labels vs. labels that treat format-only
     differences as clean: a missing value serialized as '' (injected NaNs reach
     the worker as empty CSV fields, so echoing them back as "" differs from the
     oracle's NaN without violating any typed rule) and a date or number written
     differently but with the same value (2019-07-08 00:00:00 vs. its ISO 8601
     form, 2019.0 vs. '2019').
  5. Cost: rho* with a row-bootstrap interval, and the lower envelope (which
     configuration is cheapest overall at each c_FN/c_FP).
  6. Share of positives on rows with no injected fault.
  7. Reversed pairing, when rev_* reports are present: worker x checker table.

Usage:  python robustness_checks.py [runs_dir]      (default: runs)
"""
import json
import os
import random
import re
import sys
from datetime import datetime
from math import comb

MODES = ["circular", "reflexion", "rules-only", "cross-only", "hybrid", "hybrid-circular"]
SEEDS = (1, 2, 3)
BOOTSTRAPS = 2000
EMPTY_FOR_NAN = re.compile(r"^.+?: expected (np\.float64\()?nan\)?, got ''$")
EMPTY_CANONICAL = re.compile(r"^.+?: '' violates canonical set ")
EXPECTED_GOT = re.compile(r"^.+?: expected (.+), got (.+)$")


def report(runs, prefix, mode, seed):
    with open(os.path.join(runs, f"{prefix}_{mode}_seed{seed}_final_report.json")) as f:
        return json.load(f)


def rows(runs, prefix, mode):
    """{(seed, row_idx): row} for every checked row of a configuration."""
    return {(s, r["row_idx"]): r for s in SEEDS
            for r in report(runs, prefix, mode, s)["row_level_results"]}


def flagged(row):
    return row["verdict"] == "NOT_OK"


def literal(text):
    return re.sub(r"^np\.\w+\((.*)\)$", lambda m: m.group(1), text).strip("'\"")


def same_value(reason):
    m = EXPECTED_GOT.match(reason)
    if not m:
        return False
    expected, got = literal(m.group(1)), literal(m.group(2))
    try:
        return float(expected) == float(got)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(expected.replace("Z", "+00:00")).replace(tzinfo=None) ==             datetime.fromisoformat(got.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return False


def format_only(row):
    reasons = row["ground_truth_reasons"]
    return bool(reasons) and all(EMPTY_FOR_NAN.match(x) or EMPTY_CANONICAL.match(x) or same_value(x) for x in reasons)


def confusion(table, keys=None, strict=False):
    tp = fp = fn = tn = 0
    for k in (table if keys is None else keys):
        r = table[k]
        bad = bool(r["ground_truth_bad"]) and not (strict and format_only(r))
        tp += flagged(r) and bad
        fp += flagged(r) and not bad
        fn += (not flagged(r)) and bad
        tn += (not flagged(r)) and not bad
    return tp, fp, fn, tn


def fisher(a, b, c, d):
    n, r, k = a + b + c + d, a + b, a + c
    pmf = lambda x: comb(r, x) * comb(n - r, k - x) / comb(n, k)
    p0 = pmf(a)
    return sum(pmf(x) for x in range(max(0, k - (n - r)), min(r, k) + 1) if pmf(x) <= p0 * (1 + 1e-9))


def ratio(c):
    return c[0] / (c[0] + c[2]) if c[0] + c[2] else float("nan"), c[0] / (c[0] + c[1]) if c[0] + c[1] else float("nan")


def breakeven(x, h):
    """(FP_X-FP_H)/(FN_H-FN_X) when X has fewer FN and more FP, else None."""
    d_fn, d_fp = h[2] - x[2], x[1] - h[1]
    return d_fp / d_fn if d_fn > 0 and d_fp > 0 else None


def envelope(costs):
    """Cheapest configuration as c_FN/c_FP grows: walk the (FN, FP) lower hull."""
    cur = min(costs, key=lambda m: (costs[m][1], costs[m][2]))
    out, at = [], 0.0
    while True:
        out.append((at, cur))
        best, best_r = None, None
        for m, c in costs.items():
            if c[2] < costs[cur][2]:
                r = (c[1] - costs[cur][1]) / (costs[cur][2] - c[2])
                if best_r is None or r < best_r or (r == best_r and c[2] < costs[best][2]):
                    best, best_r = m, r
        if best is None:
            return out
        cur, at = best, max(best_r, 0.0)


def bootstrap_rho(tables, mode, keys, rng):
    vals = []
    for _ in range(BOOTSTRAPS):
        sample = [rng.choice(keys) for _ in keys]
        x, h = confusion(tables[mode], sample), confusion(tables["hybrid"], sample)
        d_fn, d_fp = h[2] - x[2], x[1] - h[1]
        vals.append(float("inf") if d_fn <= 0 else max(d_fp, 0) / d_fn)
    vals.sort()
    return vals[int(.025 * BOOTSTRAPS)], vals[int(.975 * BOOTSTRAPS) - 1]


def analyze(runs, name, prefix, rng):
    print(f"\n===== {name} ({prefix}) =====")
    t = {m: rows(runs, prefix, m) for m in MODES}
    keys = sorted(t["hybrid"])

    rules_tp = {k for k, r in t["rules-only"].items() if flagged(r) and r["ground_truth_bad"]}
    cross_tp = {k for k, r in t["cross-only"].items() if flagged(r) and r["ground_truth_bad"]}
    hybrid_tp = {k for k, r in t["hybrid"].items() if flagged(r) and r["ground_truth_bad"]}
    llm = [k for k, r in t["hybrid"].items() if r["checker_path"] == "llm" and t["cross-only"][k]["checker_path"] == "llm"]
    differ = sum(flagged(t["hybrid"][k]) != flagged(t["cross-only"][k]) for k in llm)
    print(f"[1] cross TP {len(cross_tp)}, also caught by rules {len(cross_tp & rules_tp)}; "
          f"cross TP lost inside hybrid {len(cross_tp - hybrid_tp)}; union of channels "
          f"{len(cross_tp | rules_tp)} vs hybrid {len(hybrid_tp)}; "
          f"same checker, same row, different verdict {differ}/{len(llm)}")

    missing = {m: {k for k, r in t[m].items() if r["checker_path"] == "missing"} for m in MODES}
    print("[2] unscored rows:", {m: (len(v), sum(bool(t[m][k]["ground_truth_bad"]) for k in v))
                                 for m, v in missing.items() if v},
          "| circular&reflexion", len(missing["circular"] & missing["reflexion"]),
          "| circular&hybrid-circular", len(missing["circular"] & missing["hybrid-circular"]))

    print("[3] discrimination           TPR   FPR     J  Fisher p")
    for m in MODES:
        c = confusion(t[m])
        tpr, fpr = c[0] / (c[0] + c[2]), c[1] / (c[1] + c[3])
        print(f"    {m:18s}     {tpr:.3f} {fpr:.3f} {tpr - fpr:+.3f}  {fisher(c[0], c[2], c[1], c[3]):.4f}")

    only = [k for k in keys if format_only(t["hybrid"][k])]
    pos = sum(bool(t["hybrid"][k]["ground_truth_bad"]) for k in keys)
    print(f"[4] positives {pos}; format-only {len(only)}; remaining {pos - len(only)}")
    print("    configuration       primary TP/FN/FP recall prec | strict TP/FN/FP recall prec")
    for m in MODES:
        a, b = confusion(t[m]), confusion(t[m], strict=True)
        ra, pa, rb, pb = *ratio(a), *ratio(b)
        print(f"    {m:18s} {a[0]:3d}/{a[2]:3d}/{a[1]:3d}  {ra:.3f} {pa:.3f} | "
              f"{b[0]:3d}/{b[2]:3d}/{b[1]:3d}  {rb:.3f} {pb:.3f}")

    for strict in (False, True):
        c = {m: confusion(t[m], strict=strict) for m in MODES}
        p = c["hybrid"][0] + c["hybrid"][2]
        n = c["hybrid"][1] + c["hybrid"][3]
        c["flag-everything"] = (p, n, 0, 0)
        print(f"[5] cost, {'strict' if strict else 'primary'} labels; rho* against hybrid (FN={c['hybrid'][2]}, FP={c['hybrid'][1]}):")
        for m in MODES + ["flag-everything"]:
            if m == "hybrid":
                continue
            r = breakeven(c[m], c["hybrid"])
            text = "no finite break-even" if r is None else f"{r:.2f}"
            if r is not None and m != "flag-everything" and not strict:
                lo, hi = bootstrap_rho(t, m, keys, rng)
                text += f"  95% bootstrap [{lo:.2f}, {'inf' if hi == float('inf') else f'{hi:.2f}'}]"
            print(f"    {m:18s} {text}")
        steps = envelope(c)
        print("    cheapest overall:", "; ".join(f"from {at:.2f}: {m}" for at, m in steps))

    nat = sum(bool(t["hybrid"][k]["ground_truth_bad"]) for k in keys if k[1] not in injected_rows(runs, prefix, k[0]))
    print(f"[6] positives on rows with no injected fault: {nat}/{pos}")
    return {m: confusion(t[m]) for m in MODES}, {m: confusion(t[m], strict=True) for m in MODES}


_INJECTED = {}


def injected_rows(runs, prefix, seed):
    key = (runs, prefix, seed)
    if key not in _INJECTED:
        with open(os.path.join(runs, f"{prefix}_hybrid_seed{seed}_injected_errors.json")) as f:
            _INJECTED[key] = {e["row"] for e in json.load(f)}
    return _INJECTED[key]


def main(runs="runs"):
    rng = random.Random(0)
    results = {}
    for name, prefix in [("UCI Adult", "uci"), ("KramaBench", "kb"),
                         ("UCI Adult, reversed", "rev_uci"), ("KramaBench, reversed", "rev_kb")]:
        if os.path.exists(os.path.join(runs, f"{prefix}_hybrid-circular_seed{SEEDS[-1]}_final_report.json")):
            results[prefix] = analyze(runs, name, prefix, rng)

    if not {"rev_uci", "rev_kb"} <= results.keys():
        return
    print("\n===== [7] Worker x checker (single-pass LLM checker, no rules) =====")
    print("dataset labels   worker/checker    relation   TP/FN/FP  recall  prec  flag%")
    cells = [("Llama", "Llama", "same", "", "circular"), ("Llama", "Mistral", "cross", "", "cross-only"),
             ("Mistral", "Mistral", "same", "rev_", "circular"), ("Mistral", "Llama", "cross", "rev_", "cross-only")]
    for ds in ("uci", "kb"):
        for i, labels in enumerate(("primary", "strict")):
            for w, ch, rel, pre, mode in cells:
                c = results[pre + ds][i][mode]
                r, p = ratio(c)
                print(f"{ds:7s} {labels:8s} {w:7s}/{ch:8s} {rel:6s}   {c[0]:3d}/{c[2]:3d}/{c[1]:3d}  {r:.3f}  {p:.3f}  "
                      f"{100 * (c[0] + c[1]) / sum(c):.0f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
