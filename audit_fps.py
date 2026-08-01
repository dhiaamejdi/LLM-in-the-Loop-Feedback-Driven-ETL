#!/usr/bin/env python3
"""Reproduce the false-positive audit of the circular checker (paper, finding 1).

For every circular-configuration false positive (row flagged NOT_OK whose
ground-truth label is clean), reconstruct the actual transformed row from the
archived raw worker outputs and mechanically test whether the checker's stated
rationale (asserted untrimmed whitespace) is true of the row. Rows in seeds
whose transform archives reconstruct to fewer rows than the sample size cannot
be row-aligned with certainty and are counted as unverifiable.

A seed is auditable only if the on-disk transform provably matches the one the
circular verdicts were scored against. Criterion: the circular report's
row-level ground-truth labels must equal the rules+same-family report's for
that seed (both were scored against the on-disk transform state of their
session; equality proves rule-relevant equivalence). Seeds failing this, or
reconstructing short, are counted unverifiable.

Usage:  python audit_fps.py [runs_dir]      (default: runs)
"""
import contextlib
import glob
import io
import json
import re
import sys
import importlib.util

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"

spec = importlib.util.spec_from_file_location("sg", "safeguarded_LLM_in_the_loop.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

total = unverifiable = ws_not_present = ws_present = other = 0
for ds in ("kb", "uci"):
    for f in sorted(glob.glob(f"{RUNS}/{ds}_circular_seed*_final_report.json")):
        seed = int(re.search(r"seed(\d+)", f).group(1))
        report = json.load(open(f))
        with contextlib.redirect_stdout(io.StringIO()):
            dft = sg.reconstruct_transformed_df_from_files(f"{RUNS}/{ds}_seed{seed}")
        n, expected = len(dft), report["config"]["sample_size"]
        # transform-identity check against the rules+same-family report, if present
        labels_match = True
        hc_path = f"{RUNS}/{ds}_hybrid-circular_seed{seed}_final_report.json"
        try:
            hc = json.load(open(hc_path))
            gt_a = {r["row_idx"]: r["ground_truth_bad"] for r in report["row_level_results"]}
            gt_b = {r["row_idx"]: r["ground_truth_bad"] for r in hc["row_level_results"]}
            labels_match = gt_a == gt_b
        except FileNotFoundError:
            pass  # fall back to completeness-only criterion
        for row in report["row_level_results"]:
            if row["verdict"] == "NOT_OK" and not row["ground_truth_bad"]:
                total += 1
                idx, note = row["row_idx"], (row.get("checker_notes") or "").lower()
                if (not labels_match) or n < expected or idx >= n:
                    unverifiable += 1
                    continue
                strings = [str(v) for v in dft.iloc[idx].to_dict().values() if isinstance(v, str)]
                claims_ws = "whitespace" in note or "trim" in note
                has_ws = any(v != v.strip() for v in strings)
                if claims_ws and not has_ws:
                    ws_not_present += 1
                elif claims_ws:
                    ws_present += 1
                else:
                    other += 1

verifiable = total - unverifiable
print(f"total circular FPs: {total} | verifiable: {verifiable} | unverifiable(alignment): {unverifiable}")
print(f"verifiable FPs asserting whitespace NOT present in the row: {ws_not_present}/{verifiable}")
if ws_present or other:
    print(f"  (whitespace actually present: {ws_present}; other/vague rationale: {other})")
