#!/usr/bin/env python3
"""Pilot on requirements that resist compilation (paper, Section 6).

Each of the 60 authored items in datasets/semantic_tier_cases.json pairs one
source record with a correct and a wrong transformation; the requirement is that
the transformation denotes the same real-world entity as the source. No worker is
involved: this measures checkers on fixed pairs, so "same-family" does not apply.

Model-free baselines (no gold lookup table is available to any checker; a complete
table would compile the requirement away and is out of scope by construction):
  - vocabulary:  one-input; value is well-formed, so nothing is flagged
  - comparator:  two-input; flags any field not equal to the source after trim/case
  - similarity:  flags when string similarity to the source is below a threshold,
                 tuned on these same items in hindsight (an optimistic baseline)

LLM checkers run with two rule texts: the fixed ablation rule ("ablation") and one
stating the same-entity requirement ("semantic"), sharing one neutral exemplar.
Seeds change only row order, hence batch composition, so items are scored by
majority verdict across seeds.

Usage:  python semantic_tier.py run [--seeds 1,2,3] [--runs runs]   (GPU, HF_TOKEN)
        python semantic_tier.py score [--runs runs]                 (no GPU)
"""
import argparse
import difflib
import json
import os
import random
from math import comb

CASES = "datasets/semantic_tier_cases.json"
MODELS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.2"}
BATCH = 5

ABLATION_RULE = (
    "Compare every field of 'transformed' against 'original' character by character. "
    "Return verdict NOT_OK if ANY field has: leading/trailing whitespace that was not "
    "trimmed; a label that was not normalized (e.g. a trailing period, or wrong casing "
    "such as lowercase where canonical casing is expected); or a value that does not "
    "belong in its column. Return OK only if EVERY field is correct."
)
SEMANTIC_RULE = (
    "The 'transformed' record must state the same real-world facts as 'original'. "
    "Different wording, language, abbreviation or category labels are fine. Return "
    "verdict NOT_OK if a transformed value denotes a different real-world entity or "
    "category than the original implies; use all fields as context. Return OK otherwise."
)
EXEMPLAR = (
    "## EXAMPLE:\n### INPUT:\n"
    '[\n  {"row_idx": 998, "original": {"city": "Roma"}, "transformed": {"city": "Rome"}},\n'
    '  {"row_idx": 999, "original": {"city": "Firenze"}, "transformed": {"city": "Venice"}}\n]\n'
    "### CORRECT OUTPUT:\n"
    '[\n  {"row_idx": 998, "verdict": "OK", "notes": "Roma is Rome."},\n'
    '  {"row_idx": 999, "verdict": "NOT_OK", "notes": "Firenze is Florence, not Venice."}\n]\n\n'
)


def load_items():
    with open(CASES, encoding="utf-8") as f:
        cases = json.load(f)
    items = []
    for i, c in enumerate(cases):
        items.append({"id": 2 * i, "cat": c["cat"], "source": c["source"], "transformed": c["good"], "bad": False})
        items.append({"id": 2 * i + 1, "cat": c["cat"], "source": c["source"], "transformed": c["bad"], "bad": True})
    return items


def prompt_for(rule, cases):
    return (
        "You are a silent, precise JSON-generating API. You will be given a rule and a JSON "
        "array of data cases. For each case, decide whether the 'transformed' data is a valid "
        "transformation of the 'original' data according to the rule. Your entire response "
        "MUST be a single JSON array, one object per case, no other text or markdown.\n\n"
        f"## RULE:\n{rule}\n\n{EXEMPLAR}"
        f"## ACTUAL TASK:\n### INPUT:\n{json.dumps(cases, indent=2)}\n### CORRECT OUTPUT:"
    )


def run(args):
    from safeguarded_LLM_in_the_loop import extract_json_block, get_llm_client, seed_everything

    items = load_items()
    for short, model in MODELS.items():
        client = get_llm_client(os.environ["HF_TOKEN"], model, "auto")
        for pname, rule in (("ablation", ABLATION_RULE), ("semantic", SEMANTIC_RULE)):
            for seed in args.seeds:
                seed_everything(seed)
                order = list(items)
                random.Random(seed).shuffle(order)
                verdicts, raw = {}, []
                for start in range(0, len(order), BATCH):
                    batch = order[start:start + BATCH]
                    cases = [{"row_idx": it["id"], "original": it["source"], "transformed": it["transformed"]}
                             for it in batch]
                    out = client.call(prompt_for(rule, cases), max_length=1024, temperature=0.0).replace(r"\_", "_")
                    raw.append(out)
                    try:
                        parsed = extract_json_block(out)
                        for v in parsed if isinstance(parsed, list) else []:
                            verdicts[int(v["row_idx"])] = {"verdict": v.get("verdict"), "notes": v.get("notes")}
                    except Exception as e:
                        print(f"  unparsable batch {start}: {e}", flush=True)
                path = os.path.join(args.runs, f"sem_{short}_{pname}_seed{seed}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"model": model, "prompt": pname, "seed": seed, "verdicts": verdicts, "raw": raw}, f, indent=1)
                print(f"{short}/{pname}/seed{seed}: {len(verdicts)}/{len(items)} verdicts", flush=True)


def cp(k, n, a=0.05):
    def tail(p, lo):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in (range(k, n + 1) if lo else range(0, k + 1)))

    def solve(lo):
        l, h = 0.0, 1.0
        for _ in range(60):
            m = (l + h) / 2
            if lo:
                l, h = (m, h) if tail(m, True) < a / 2 else (l, m)
            else:
                l, h = (m, h) if tail(m, False) > a / 2 else (l, m)
        return (l + h) / 2

    return 0.0 if k == 0 else solve(True), 1.0 if k == n else solve(False)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def text(rec):
    return " ".join(str(v) for v in rec.values()).strip().lower()


def similarity_flags(items):
    sims = {it["id"]: difflib.SequenceMatcher(None, text(it["source"]), text(it["transformed"])).ratio() for it in items}
    best = max(sorted(set(sims.values())) + [1.01],
               key=lambda t: sum((sims[it["id"]] < t) == it["bad"] for it in items))
    return {it["id"]: sims[it["id"]] < best for it in items}, best


def line(name, flags, items):
    tp = sum(flags[i["id"]] and i["bad"] for i in items)
    fp = sum(flags[i["id"]] and not i["bad"] for i in items)
    pos = sum(i["bad"] for i in items)
    lo, hi = cp(tp, pos)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    acc = (tp + (len(items) - pos - fp)) / len(items)
    print(f"{name:26s} TP {tp:2d}/{pos} recall {tp / pos:.3f} [{lo:.3f}, {hi:.3f}]  FP {fp:2d}  prec {prec:.3f}  acc {acc:.3f}")


def score(args):
    items = load_items()
    baselines = {
        "vocabulary (one-input)": {i["id"]: False for i in items},
        "comparator (two-input)": {i["id"]: text(i["source"]) != text(i["transformed"]) for i in items},
    }
    sim, thr = similarity_flags(items)
    baselines[f"similarity (t={thr:.2f})"] = sim
    for name, flags in baselines.items():
        line(name, flags, items)
    results = {}
    for short in MODELS:
        for pname in ("ablation", "semantic"):
            seeds, missing = [], 0
            for s in args.seeds:
                path = os.path.join(args.runs, f"sem_{short}_{pname}_seed{s}.json")
                with open(path, encoding="utf-8") as f:
                    v = json.load(f)["verdicts"]
                seeds.append({i["id"]: v.get(str(i["id"]), {}).get("verdict") == "NOT_OK" for i in items})
                missing += sum(str(i["id"]) not in v for i in items)
            flags = {i["id"]: sum(s[i["id"]] for s in seeds) * 2 > len(seeds) for i in items}
            results[short, pname] = flags
            line(f"{short}/{pname} (miss {missing})", flags, items)
            unstable = sum(len({s[i["id"]] for s in seeds}) > 1 for i in items)
            correct = lambda f: {i["id"] for i in items if f[i["id"]] == i["bad"]}
            b = len(correct(flags) - correct(sim))
            c = len(correct(sim) - correct(flags))
            print(f"{'':26s} verdict changes across seeds on {unstable}/{len(items)} items; "
                  f"vs similarity b={b} c={c} exact McNemar p={mcnemar(b, c):.4f}")
    print("\nPer category (accuracy on 2 x items):")
    cats = sorted({i["cat"] for i in items})
    print(f"{'':26s}" + " ".join(f"{c[:12]:>12s}" for c in cats))
    for name, flags in [(f"similarity", sim)] + [(f"{k[0]}/{k[1]}", v) for k, v in results.items()]:
        vals = []
        for c in cats:
            sub = [i for i in items if i["cat"] == c]
            vals.append(sum(flags[i["id"]] == i["bad"] for i in sub) / len(sub))
        print(f"{name:26s}" + " ".join(f"{v:12.3f}" for v in vals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3])
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    (run if a.cmd == "run" else score)(a)
