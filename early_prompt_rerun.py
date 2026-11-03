#!/usr/bin/env python3
"""Rerun the harness's early checker prompt under the ablation's controls (paper, Section 6).

The early prompt (a vague rule and a single exemplar whose verdict is OK; replaced
in commit 443a34d, and distinct from the pilot notebook's prompt) is applied to the archived worker transforms, with the same
sampled rows, batching, token cap and scoring as the ablation. Only the prompt
differs, so verdicts pair with the circular and cross-only reports on the same rows.
Rows without a parsable verdict, and verdict words other than OK and NOT_OK, are
counted separately instead of being scored as not flagged.

Usage:  python early_prompt_rerun.py run [--dry] [--runs runs]   (GPU, cached models)
        python early_prompt_rerun.py score [--runs runs]         (no GPU)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

MODELS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.2"}
KB = dict(data_csv="datasets/water-body-testing-2019.csv", rules_json="datasets/kramabench_water_rules.json", checked=40)
UCI = dict(data_csv=None, rules_json=None, checked=30)
# (report prefix, dataset, worker, seeds). Only transforms that reproduce the labels of
# their ablation report qualify, which excludes the archived Llama-worker UCI tables.
TRANSFORMS = [("rev_uci", UCI, "mistral", (1, 2, 3)), ("rev_kb", KB, "mistral", (1, 2, 3)),
              ("kb", KB, "llama", (1,))]
EARLY_HEADER = (
    "You are a silent, precise JSON-generating API. You will be given a rule and a JSON "
    "array of data cases. For each case, you will check if the 'transformed' data is a "
    "valid transformation of the 'original' data according to the rule. Your entire "
    "response MUST be a single JSON array, with one object per case. Do not include any "
    "text, explanations, or markdown before or after the JSON array.\n\n"
    "## RULE:\n"
    "The 'transformed' row should be a semantically correct version of the 'original' "
    "row, with whitespace trimmed and labels normalized.\n\n"
    "## EXAMPLE:\n### INPUT:\n"
    "[\n  {\n"
    '    "row_idx": 999,\n'
    '    "original": {"name": " john ", "income": "<=50K."},\n'
    '    "transformed": {"name": "john", "income": "<=50K"}\n'
    "  }\n]\n"
    "### CORRECT OUTPUT:\n"
    "[\n  {\n"
    '    "row_idx": 999,\n'
    '    "verdict": "OK",\n'
    '    "notes": "Whitespace was trimmed and income label was normalized correctly."\n'
    "  }\n]\n\n"
)
TASK = "## ACTUAL TASK:"


class EarlyPrompt:
    """Swaps the rule and exemplars of each checker prompt; the cases are untouched."""

    def __init__(self, client):
        self.client = client
        self.model_name = client.model_name

    def call(self, prompt, **kw):
        assert prompt.count(TASK) == 1
        return self.client.call(EARLY_HEADER + prompt[prompt.index(TASK):], **kw)


def baseline_mode(worker, checker):
    return "circular" if worker == checker else "cross-only"


def report_path(runs, prefix, mode, seed):
    return os.path.join(runs, f"{prefix}_{mode}_seed{seed}_final_report.json")


def run(args):
    import safeguarded_LLM_in_the_loop as h

    for prefix, ds, worker, seeds in TRANSFORMS:
        rules = h.load_business_rules(ds["rules_json"]) if ds["rules_json"] else h.BUSINESS_RULES
        for seed in seeds:
            h.seed_everything(seed)
            df = h.load_dataset(ds["data_csv"])
            sample = df.sample(min(100, len(df)), random_state=seed).reset_index(drop=True)
            source, _ = h.inject_errors(sample, {"missing_value": 0.03, "swap_columns": 0.02, "bad_casing": 0.03,
                                                 "stray_whitespace": 0.03}, seed=seed)
            with open(os.path.join(args.runs, f"{prefix}_seed{seed}_transformed.json")) as f:
                transformed = pd.DataFrame(json.load(f)).reindex(columns=source.columns)
            idxs = sorted(int(i) for i in np.random.default_rng(seed).choice(len(transformed), size=ds["checked"], replace=False))
            truth = {i: h.row_has_residual_error(transformed.iloc[i], source.iloc[i], rules) for i in idxs}
            for checker, model in MODELS.items():
                with open(report_path(args.runs, prefix, baseline_mode(worker, checker), seed)) as f:
                    base = {r["row_idx"]: r for r in json.load(f)["row_level_results"]}
                same = sorted(base) == idxs and all(bool(truth[i][0]) == bool(base[i]["ground_truth_bad"]) for i in idxs)
                print(f"{prefix} seed{seed} checker={checker}: {sum(truth[i][0] for i in idxs)}/{len(idxs)} bad; "
                      f"rows and labels match the ablation report: {same}", flush=True)
                if args.dry or not same:
                    continue
                out = os.path.join(args.runs, f"oldprompt_{prefix}_{checker}_seed{seed}")
                client = EarlyPrompt(h.get_llm_client(os.environ.get("HF_TOKEN", "cached"), model, "auto"))
                verdicts = {v.get("row_idx"): v for v in h.run_checker("cross-only", client, source, transformed, idxs, out, rules)}
                rows = [{"row_idx": i, "verdict": verdicts.get(i, {}).get("verdict"), "notes": verdicts.get(i, {}).get("notes"),
                         "ground_truth_bad": bool(truth[i][0])} for i in idxs]
                with open(out + "_report.json", "w") as f:
                    json.dump({"prefix": prefix, "worker": worker, "checker": checker, "seed": seed, "rows": rows}, f, indent=1)


def tally(rows):
    """Exact NOT_OK is what the harness scores; "other" is any other non-OK verdict word."""
    bad = [r for r in rows if r["ground_truth_bad"]]
    exact = sum(r["verdict"] == "NOT_OK" for r in rows)
    missing = sum(not r["verdict"] for r in rows)
    other = sum(bool(r["verdict"]) and r["verdict"] not in ("OK", "NOT_OK") for r in rows)
    flagged_bad = sum(bool(r["verdict"]) and r["verdict"] != "OK" for r in bad)
    return dict(n=len(rows), pos=len(bad), exact=exact, other=other, missing=missing,
                flags=exact + other, tp=flagged_bad, fp=exact + other - flagged_bad)


def score(args):
    print("flags = any non-OK verdict; exact = literally NOT_OK; miss = no parsable verdict")
    print("transforms  checker  rel    rows pos | early prompt: exact other miss flags TP FP | ablation prompt: exact miss TP FP")
    groups, words, totals = {}, {}, {}
    for prefix, _, worker, seeds in TRANSFORMS:
        for checker in MODELS:
            for seed in seeds:
                with open(os.path.join(args.runs, f"oldprompt_{prefix}_{checker}_seed{seed}_report.json")) as f:
                    new = json.load(f)["rows"]
                with open(report_path(args.runs, prefix, baseline_mode(worker, checker), seed)) as f:
                    old = {r["row_idx"]: r for r in json.load(f)["row_level_results"]}
                base = [{"verdict": None if old[r["row_idx"]]["checker_path"] == "missing" else old[r["row_idx"]]["verdict"],
                         "ground_truth_bad": bool(old[r["row_idx"]]["ground_truth_bad"])} for r in new]
                g = groups.setdefault((prefix, checker, worker), {"new": [], "base": []})
                g["new"] += new
                g["base"] += base
                for r in new:
                    if r["verdict"] != "OK":
                        words.setdefault(checker, {}).setdefault(str(r["verdict"]), 0)
                        words[checker][str(r["verdict"])] += 1
    for (prefix, checker, worker), g in groups.items():
        a, b = tally(g["new"]), tally(g["base"])
        rel = "same" if checker == worker else "cross"
        print(f"{prefix:10s}  {checker:7s}  {rel:5s}  {a['n']:4d} {a['pos']:3d} | {a['exact']:3d} {a['other']:3d} {a['missing']:3d} {a['flags']:3d} {a['tp']:3d} {a['fp']:3d} "
              f"| {b['exact']:3d} {b['missing']:3d} {b['tp']:3d} {b['fp']:3d}")
        t = totals.setdefault(checker, {"a": [], "b": []})
        t["a"] += g["new"]
        t["b"] += g["base"]
    for checker, t in totals.items():
        a, b = tally(t["a"]), tally(t["b"])
        for name, c in (("early prompt", a), ("ablation prompt", b)):
            prec = c["tp"] / c["flags"] if c["flags"] else float("nan")
            print(f"TOTAL {checker} {name}: {c['n']} rows, {c['pos']} bad; exact {c['exact']}; other words {c['other']}; missing {c['missing']}; "
                  f"flags {c['flags']} ({100 * c['flags'] / c['n']:.0f}%); recall {c['tp'] / c['pos']:.3f}; precision {prec:.3f}")
    print("non-OK verdict words:", words)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    (run if a.cmd == "run" else score)(a)
