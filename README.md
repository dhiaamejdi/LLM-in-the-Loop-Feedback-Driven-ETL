# LLM-in-the-Loop Feedback-Driven ETL

A modular architecture for trustworthy Extract-Transform-Load (ETL) pipelines augmented by large language models (LLMs). Each ETL stage is run by a **Worker LLM** and validated by a paired **LLM-as-a-Checker**, with a feedback loop that consolidates human corrections and model-flagged discrepancies.

This repository contains the prototype and evaluation harness for the paper:

> **LLM-in-the-Loop Feedback-Driven ETL: Toward Trustworthy, Self-Adaptive Data Pipelines**
> Anonymous Authors.

The paper's arc, honestly: an early pilot suggested a **circular validation problem** (same-family checkers blind to their own errors, echoing LLM self-correction limits [Kamoi et al. 2024] and evaluator self-preference [Panickssery et al. 2024]). A controlled six-configuration, multi-seed ablation — the primary evidence of the paper — did **not** reproduce that blindness: with a balanced checker prompt, the same-family checker in our pairing *over-flags* (recall 0.84–0.93 at precision 0.49–0.64, barely above a flag-everything floor of 0.43–0.59), cross-architecture checkers are conservative-but-precise (0.22–0.28 recall at 0.94–1.00 precision), Reflexion-style self-critique shows no detectable benefit at this scale, and the durable positive result is that deterministic rules and cross-model checking catch **complementary** errors: the rules+cross hybrid achieves the best near-zero-false-positive coverage (0.45–0.51 recall at ≥0.96 precision) while sending 14–17% fewer rows to the LLM, and a rules+same-family hybrid attains the highest observed recall (0.96 on UCI) but inherits the same-family false-positive flood — hybridization complements precision, it does not repair it. The pilot's apparent blindness traces to an OK-anchored checker prompt — prompt specification is a first-order variable in LLM-validation studies. All numbers reproduce from the committed `runs/` artifacts.

---

## What changed after review (v2)

The harness was rewritten to make every claim in the paper attributable and reproducible:

1. **Ablation modes.** `--mode {circular, reflexion, cross-only, rules-only, hybrid, hybrid-circular}` runs all six cells of the design (same-family vs. cross-family checker × with/without deterministic rules, plus a Reflexion-style self-critique baseline), so the contribution of each safeguard can be measured separately instead of being confounded.
2. **Ground-truth metrics.** `inject_errors()` now logs every injected fault as `(row, column, error_type)`, and a deterministic oracle labels every transformed row as clean / residual-error. Checker verdicts are scored against this ground truth: TP/FP/FN/TN, recall, precision, and **exact Clopper–Pearson 95% confidence intervals** (pure Python, no scipy).
3. **Per-path attribution.** Every verdict is tagged `checker="deterministic"` or `checker="llm"`, and the final report breaks metrics down per path.
4. **Full seeding and multi-seed runs.** `--seed N` or `--seeds 1,2,...,10`.
5. **No secrets in source.** The token is read from the `HF_TOKEN` environment variable only. (A previous revision accidentally contained a hardcoded token; it has been removed and revoked.)
6. **`--resume-from-files` is a real CLI flag** (it was previously hardcoded to `True`, silently overriding the flag).
7. **Bug fix in error injection.** Bad-casing and stray-whitespace injections previously reused the same `random_state`, corrupting the same rows twice; they now draw independently from the seeded generator. String-column detection also works under both classic (`object`) and modern (`str`) pandas dtypes.

## Repository contents

| File | Purpose |
|------|---------|
| `safeguarded_LLM_in_the_loop.py` | Evaluation harness for the **Safeguarded Architecture**: worker transformation, hybrid validation, ablation modes, ground-truth scoring. |
| `LLM_in_the_Loop_Driven_ETL.ipynb` | Notebook for the **full four-stage pipeline** and per-stage Worker/Checker accuracy measurements against deterministic oracle functions. |
| `requirements.txt` | Python dependencies. |

## Which file produces which result

| Paper claim / table | Where it is produced |
|---------------------|----------------------|
| Worker accuracy per stage (100% / 94.73% / 95.47% / 94.35%) | Pilot notebook, via oracle comparison (pilot scope; see paper §5.2) |
| Hallucination 4.2%, reproducibility 0.87, prompt-success 73.2%, checker 42%/8.3% | **Withdrawn** — no artifact source; see paper §5.2 |
| Circular validation baseline (Llama checks Llama) | `--mode circular` |
| Safeguarded results (recall/precision + 95% CI, Tables 4–5) | `--mode hybrid` → `*_final_report.json` |
| Cross-architecture-only / rules-only ablation cells | `--mode cross-only` / `--mode rules-only` |
| Reflexion baseline / rules+same-family hybrid | `--mode reflexion` / `--mode hybrid-circular` |
| FP audit (finding 1, 39/39) | `python audit_fps.py runs` |
| Cost analysis (Table 2) | Estimated from token counts; not auto-generated |

## Setup

Requires Python 3.9+ and a GPU for the LLM stages (`Llama-3.1-8B-Instruct` and `Mistral-7B-Instruct-v0.2`, both gated on Hugging Face).

```bash
pip install -r requirements.txt
export HF_TOKEN="your_token_here"   # never hardcode it
```

## Running

```bash
# Full safeguarded run (GPU + gated-model access required)
python safeguarded_LLM_in_the_loop.py --mode hybrid

# Circular-validation baseline (worker model checks itself)
python safeguarded_LLM_in_the_loop.py --mode circular

# Deterministic rules only (no LLM checker)
python safeguarded_LLM_in_the_loop.py --mode rules-only

# Reproduce the reported run from archived raw Worker outputs (no GPU for transform)
python safeguarded_LLM_in_the_loop.py --mode hybrid --resume-from-files

# Full 2x2 ablation across 10 seeds
for m in circular cross-only rules-only hybrid; do
  python safeguarded_LLM_in_the_loop.py --mode $m --seeds 1,2,3,4,5,6,7,8,9,10
done

# Run on a dataset OTHER than UCI Adult (any tabular CSV), with your own rules
python safeguarded_LLM_in_the_loop.py --mode hybrid \
  --data-csv path/to/nyc_open_data_export.csv \
  --rules-json my_rules.json --seeds 1,2,3
```

`my_rules.json` is a JSON **list of typed rules** (see `datasets/kramabench_water_rules.json`
for a working example). Each rule has a `type`, a natural-language `description` used in the
worker prompt, and type-specific fields:

```json
[
  {"type": "trim_whitespace",
   "description": "Trim whitespace in all string columns"},
  {"type": "uppercase", "column": "Violation",
   "description": "Normalize the Violation flag to uppercase"},
  {"type": "canonical_values", "column": "Violation", "allowed": ["YES", "NO"],
   "description": "Violation must be exactly 'YES' or 'NO'"}
]
```

Supported types: `trim_whitespace`, `strip_suffix`, `capitalize`, `uppercase`, `titlecase`,
`canonical_values`. The prompt, deterministic fast path, and ground-truth oracle are all
derived from these same objects.

Rules are **typed** (`trim_whitespace`, `strip_suffix`, `capitalize`, `uppercase`,
`titlecase`, `canonical_values`), and the LLM prompt, the deterministic fast path, and the
ground-truth oracle are all derived from the same rule objects — so they cannot drift apart.

## KramaBench experiment (reviewer-recommended benchmark) — included & reproducible

`datasets/` ships the Massachusetts water-body-testing table (15,571 rows) from KramaBench's
*environment* data lake, a matching typed rule set, and `fetch_kramabench.sh` to pull the full
benchmark. The deterministic fast path needs no GPU, so this runs anywhere:

```bash
cd runs
# Detection condition: injected faults remain (passthrough worker), 10 seeds x 300 checked rows
python ../safeguarded_LLM_in_the_loop.py --mode rules-only --worker-mode passthrough \
  --data-csv ../datasets/water-body-testing-2019.csv \
  --rules-json ../datasets/kramabench_water_rules.json \
  --sample-size 1000 --heldout-size 300 --seeds 1,2,3,4,5,6,7,8,9,10 --outfile-prefix kb_water

# Clean-data condition: oracle worker (measures false positives)
python ../safeguarded_LLM_in_the_loop.py --mode rules-only --worker-mode oracle \
  --data-csv ../datasets/water-body-testing-2019.csv \
  --rules-json ../datasets/kramabench_water_rules.json \
  --sample-size 1000 --heldout-size 300 --seeds 1,2,3,4,5,6,7,8,9,10 --outfile-prefix kb_water_oracle
```

**Full-scale run (entire 22-year lake, 2002–2023, 319,911 rows).** Fetch the benchmark with
`datasets/fetch_kramabench.sh`, then glob all years:

```bash
python ../safeguarded_LLM_in_the_loop.py --mode rules-only --worker-mode passthrough \
  --data-csv "Kramabench-main/data/environment/input/water-body-testing-*.csv" \
  --rules-json ../datasets/kramabench_water_rules.json \
  --sample-size 5000 --heldout-size 1000 --seeds 1,2,3,4,5,6,7,8,9,10 --outfile-prefix kbfull
```

Pooled results:

| Run | Scale | TP | FN | FP | TN | Recall (95% CI) | Precision |
|---|---|---|---|---|---|---|---|
| Single file (2019), detection | 10 seeds × 300 checks | 796 | 0 | 0 | 2204 | 100% [0.995, 1.00] | 100% |
| Single file (2019), clean data | 10 seeds × 300 checks | 15 | 0 | 0 | 2985 | 100% | 100% (0 FP) |
| **Full lake (22 files)**, detection | **10 seeds × 1000 checks** | **9102** | **0** | **0** | **898** | **100% [0.9996, 1.00]** | **100%** |

A bonus finding from the full lake: the `Violation` column's vocabulary drifts across years
(`no/yes` → `NO/YES` → `No/Yes`), so ~half of the detected faults are **naturally occurring**
data-quality errors, not injected ones — real faults, caught alongside synthetic ones.

These runs validate the **deterministic path** and dataset generality. The **LLM** worker/checker
modes were subsequently run on this same table as part of the six-configuration ablation
(see `runs/` and the paper, Section 7.3).

`--worker-mode` options: `llm` (default; GPU), `oracle` (deterministic reference worker, no GPU,
measures checker FPs), `passthrough` (no repair, no GPU, measures checker detection).

## False-positive audit

`audit_fps.py` regenerates the paper's FP-audit claim (finding 1) from the committed
artifacts: it reconstructs transformed rows from the raw worker outputs, certifies per-seed
transform identity against the rules+same-family reports, and mechanically checks each
circular-checker false positive's stated rationale. Expected output on this repo:
`verifiable: 39 ... 39/39`.

## Reflexion baseline (agentic self-feedback)

`--mode reflexion` runs a same-family checker with a Reflexion-style second pass (the model is
shown its own verdicts and asked to reflect and revise). This is the agentic baseline requested
in review: if the circular-validation hypothesis (H1) holds, self-critique alone should *not*
recover the blind-spot errors that a different-family checker catches. Comparing
`circular` vs `reflexion` vs `cross-only` tests exactly that.

## One-command full evaluation

```bash
export HF_TOKEN=...            # GPU machine
./run_full_ablation.sh 1,2,3,4,5 runs
```

runs the complete matrix — {rules-only, circular, reflexion, cross-only, hybrid} × {UCI Adult,
KramaBench water-quality} × seeds — with each model loaded once per mode, then calls
`aggregate_results.py`, which writes:

- `runs/ablation_summary.md` — pooled TP/FP/FN/TN, recall/precision with exact 95% CIs
- `runs/ablation_table.tex` — a paste-ready LaTeX table for Section 7.3 of the paper

No local GPU? Open **`run_ablation_kaggle.ipynb`** on Kaggle or Colab (T4 is enough), add your
`HF_TOKEN` as a secret, and run all cells; the last cell prints the LaTeX table.

Each run writes:
- `<prefix>_<mode>_seed<k>_injected_errors.json` — ground-truth fault log
- `<prefix>_<mode>_seed<k>_raw_*_batch_*.txt` — raw LLM outputs (audit trail)
- `<prefix>_<mode>_seed<k>_final_report.json` — config, ground truth, checker metrics (with CIs and per-path attribution), and row-level results

## Scope and known limitations (post-ablation)

- **Executed.** The Low-Ambiguity tier (UCI Adult); the six-configuration, three-seed checker ablation on UCI Adult **and** the KramaBench water-quality table (the paper's primary evidence; Section 7.3); and the deterministic-path consistency run over the full 320k-row KramaBench lake (Section 7.4). **Planned:** the CORD and NYC Open Data tiers, DS-1000, and the DAB benchmark.
- **Sample sizes.** The ablation checks 90 (UCI) and 120 (KramaBench) rows across 3 seeds per configuration — usable but with wide confidence intervals; the superseded single-seed pilot (N=30) is retained in the paper only as a record.
- **Fault realism.** UCI-tier errors are synthetically injected; on the KramaBench tier roughly half of detected faults are *naturally occurring* vocabulary-drift errors (Section 7.4), dominated by one error class (label casing).
- **Prompt sensitivity.** The paper's own results show checker-prompt specification can qualitatively change conclusions; all findings are conditional on the fixed, balanced prompt used in the ablation.
- **Quantization & hardware.** Ablation models ran in 4-bit NF4 on a single 16 GB GPU; full-precision behavior may differ. Outputs are not bit-reproducible across hardware, which is why seeds, model commit pins (`runs/model_pins.json`), and raw outputs are archived.
- **Known engineering hazard.** Same-family (Llama) checkers degenerated into unparsable repetition on 12.5% of checked KramaBench rows' batches, while Mistral parsed cleanly on every batch (388 checked rows) — see paper Section 7.3, finding 5. The recursive batch-splitting transform and per-verdict attribution exist because of this.
