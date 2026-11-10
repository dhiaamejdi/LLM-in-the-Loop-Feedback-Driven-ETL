# LLM-in-the-Loop Feedback-Driven ETL

A modular architecture for trustworthy Extract-Transform-Load (ETL) pipelines augmented by large language models (LLMs). Each ETL stage is run by a **Worker LLM** and validated by a paired **Checker**. The feedback loop that would consolidate human corrections is specified but not implemented; this repository evaluates the checking.

This repository contains the prototype and evaluation harness for the paper:

> **LLM-in-the-Loop Feedback-Driven ETL: Toward Trustworthy Data Pipelines**
> Dhiaa Mejdi and Adel M. Alimi. NewInML Workshop at NeurIPS 2026, Paris.

The paper's arc: an early pilot suggested a **circular validation problem** (a model checking its own output misses its own errors). A controlled six-configuration, multi-seed ablation, run in **both worker/checker role assignments**, does not support an effect of the worker-checker relationship in either direction: in this model pair, behavior follows the checker model. Under the ablation's prompt, Llama flags 72-87% of rows and is the only model with parse failures, whichever worker it checks; Mistral flags 9-18% and never fails. Rules and LLMs catch largely disjoint errors, so hybrids usually raise recall, while precision comes from the LLM chosen. On compilable requirements a two-input comparator reproduces ground truth by construction and dominates every checker. About half of the positive labels are format-only differences, which strongly affects recall and break-even costs. In a small exploratory pilot, on items we authored, of requirements that resist compilation, LLM checkers reached accuracy 0.87-0.88 against 0.55 for the best string baseline, and rewording the checker's rule alone moved Llama from 0.54 to 0.88. An early prompt that never named the rejection verdict produced an apparent 0 flags, a scoring artifact: the models wrote other rejection words, which exact matching read as not flagged. Ablation numbers reproduce from the committed `runs/` artifacts.

---

## Harness design guarantees

The harness was rewritten to make checker results attributable and reproducible:

1. **Ablation modes.** `--mode {circular, reflexion, cross-only, rules-only, hybrid, hybrid-circular}` runs all six cells of the design (same-model vs. cross-model checker × with/without deterministic rules, plus a Reflexion-style self-critique baseline), so the contribution of each safeguard can be measured separately instead of being confounded.
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
| `run_full_ablation.sh`, `run_ablation_kaggle.ipynb` | Run the Llama-worker ablation on a GPU machine, or on Kaggle or Colab. |
| `run_reversed_pairing.sh` | Runs the same six configurations with Mistral as worker. |
| `aggregate_results.py` | Pools the reports into the ablation summary and LaTeX table. |
| `mcnemar_tests.py` | Paired significance tests. |
| `cost_breakeven.py` | Break-even ratios for the cost-weighted comparison. |
| `audit_fps.py` | Checks the same-model checker's false-positive rationales against the rows. |
| `robustness_checks.py` | Row-level channel overlap, label sensitivity, cost lower envelope and the worker-by-checker table. |
| `semantic_tier.py` | Pilot on requirements that resist compilation. |
| `early_prompt_rerun.py` | Reruns the harness's early checker prompt on archived transforms. |
| `datasets/` | The KramaBench water-quality table and rules, the authored semantic pilot items, and a script to fetch the full benchmark. |
| `figures/` | Mermaid sources and renders of the paper's figures. |
| `runs/` | Reports, raw LLM outputs and saved analysis output. |
| `requirements.txt` | Python dependencies. |

## Which file produces which result

| Paper claim / table | Where it is produced |
|---------------------|----------------------|
| Pilot worker accuracy per stage (94-100%: 100% / 94.73% / 95.47% / 94.35%) | Pilot notebook, via oracle comparison (pilot scope; see paper §5) |
| Circular validation baseline (Llama checks Llama) | `--mode circular` |
| Safeguarded results (recall/precision + 95% CI, Table 1) | `--mode hybrid` → `*_final_report.json` |
| Cross-model-only / rules-only ablation cells | `--mode cross-only` / `--mode rules-only` |
| Reflexion baseline / rules+same-model hybrid | `--mode reflexion` / `--mode hybrid-circular` |
| FP audit (finding 2, 39/39) | `python audit_fps.py runs` |
| Paired McNemar tests (Section 6) | `python mcnemar_tests.py runs` |
| Cost-weighted break-even ratios (rho* column of the ablation table, Section 6) | `python cost_breakeven.py runs` |
| Two-input comparator reference rows | By construction equal to the oracle (`cost_breakeven.py` prints them) |
| Reversed pairing (Table 2 of the paper) | `./run_reversed_pairing.sh 1,2,3 runs`, then `python robustness_checks.py runs` |
| Channel overlap, label sensitivity, cost lower envelope, bootstrap intervals | `python robustness_checks.py runs` (saved in `runs/robustness_checks_output.txt`) |
| Semantic pilot (Table 3 of the paper) | `python semantic_tier.py run`, then `python semantic_tier.py score` (saved in `runs/semantic_tier_output.txt`) |
| Early prompt rerun (Section 6 of the paper) | `python early_prompt_rerun.py run`, then `python early_prompt_rerun.py score` (saved in `runs/oldprompt_output.txt`) |
| Figures 1 and 2 | Mermaid sources in `figures/*.mmd` |

## Setup

Requires Python 3.9+ and a GPU for the LLM stages (`Llama-3.1-8B-Instruct` and `Mistral-7B-Instruct-v0.2`, both gated on Hugging Face).

```bash
pip install -r requirements.txt
export HF_TOKEN="your_token_here"   # never hardcode it
```

## Reproducibility details

| Item | Setting |
|---|---|
| Models | `meta-llama/Llama-3.1-8B-Instruct` at commit `0e9e39f249a16976918f6564b8830bc894c89659`; `mistralai/Mistral-7B-Instruct-v0.2` at commit `63a8b081895390a26e140280378bc85ec8bce07a` (`runs/model_pins.json`) |
| Quantization | bitsandbytes 4-bit NF4, double quantization, float16 compute |
| Decoding | Greedy (`do_sample=False`, temperature 0); stop at EOS, plus `<\|eot_id\|>` for Llama |
| Token caps | 2,048 new tokens per worker call; 1,024 per checker call (and per Reflexion second pass) |
| Batches | Worker: 10 rows (UCI) or 8 rows (KramaBench), as in `run_reversed_pairing.sh`. Checker: 5 rows per call |
| Worker retries | Up to 2 attempts per chunk; a truncated output splits the chunk in half instead; a single row that still fails keeps its source row (fallback rate is logged) |
| Checker parsing | No retry. A batch with no parsable JSON leaves its rows without a verdict, scored as `OK` (the "No verdict" column) |
| Samples | 100 rows per seed; 30 checked rows (UCI) or 40 (KramaBench) |
| Seeds | Seed *s* seeds Python and NumPy, draws the sample (`random_state=s`), the injected faults and the checked rows (`default_rng(s)`). Tables 1 and 2 use seeds 1-3; the full KramaBench consistency run uses seeds 1-10; in the semantic pilot seeds 1-3 only reorder rows |
| Environment | Reversed-pairing runs: RTX 4090 16 GB, torch 2.11, transformers 5.22. The original Llama-worker runs used a single 16 GB GPU; their library versions were not recorded |

Outputs are not bit-reproducible across hardware or library versions, which is why every raw LLM output is archived in `runs/`.

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

## KramaBench experiment — included & reproducible

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
(`no/yes` → `NO/YES` → `No/Yes`), so 46% of bad checked rows across the lake (4,196/9,102) carry no injected fault:
real data-quality errors. This does not extend to the single 2019 table used in the ablation
(0/796 in passthrough; 1/51 with the Llama worker).

These runs validate the **deterministic path** and dataset generality. The **LLM** worker/checker
modes were subsequently run on this same table as part of the six-configuration ablation
(see `runs/` and the paper, Section 6).

`--worker-mode` options: `llm` (default; GPU), `oracle` (deterministic reference worker, no GPU,
measures checker FPs), `passthrough` (no repair, no GPU, measures checker detection).

## False-positive audit

`audit_fps.py` regenerates the paper's FP-audit claim (finding 2) from the committed
artifacts: it reconstructs transformed rows from the raw worker outputs, certifies per-seed
transform identity against the rules+same-model reports, and mechanically checks each
circular-checker false positive's stated rationale. Expected output on this repo:
`verifiable: 39 ... 39/39`.

## Cost-weighted comparison

`cost_breakeven.py` regenerates the paper's cost-weighted comparison. With `c_FN` the cost of a
missed error and `c_FP` the cost of an unnecessary review, a configuration beats the rules+cross
hybrid exactly when `c_FN / c_FP` exceeds the printed break-even ratio `rho*`. Expected output on
this repo: circular 1.04 (UCI) / 2.65 (KramaBench), rules+same-model 0.74 / 2.94,
flag-everything 1.24 / 2.76; cross-only is dominated on both datasets.

## Paired significance tests

`mcnemar_tests.py` regenerates the paper's paired-significance paragraph: exact McNemar
tests and approximate minimum detectable effects for every comparison, computed from the
per-row verdicts on the identical row sets of the original five configurations.

## Reversed pairing

`run_reversed_pairing.sh` runs all six configurations with Mistral as worker (Llama as the cross-model checker,
Mistral checking itself), at the sample sizes, worker batch sizes, seeds and injected faults of the original runs.
Reports are the `runs/rev_*` files.

```bash
export HF_TOKEN=...            # GPU machine
./run_reversed_pairing.sh 1,2,3 runs
python robustness_checks.py runs
```

Pooled recall / share of rows flagged, single-pass LLM checker, primary labels:

| Worker -> checker | Relation | UCI Adult | KramaBench |
|---|---|---|---|
| Llama -> Llama | same | 0.925 / 84% | 0.843 / 73% |
| Mistral -> Llama | cross | 0.915 / 87% | 0.706 / 72% |
| Mistral -> Mistral | same | 0.255 / 14% | 0.101 / 9% |
| Llama -> Mistral | cross | 0.283 / 18% | 0.216 / 9% |

Workers differ between rows, so labels differ and these comparisons are unpaired. The reversed runs used different
hardware and library versions from the originals (RTX 4090 16 GB, torch 2.11, transformers 5.22).

## Label sensitivity

Injected missing values are `NaN`, but the worker sees them as empty CSV fields, so echoing `""` differs from the
oracle's `NaN` without violating a typed rule. `robustness_checks.py` also scores with "format-clean" labels, which
count such differences and same-value rewrites (for example a date in ISO 8601 form) as clean. Primary labels are
unchanged; the format-clean definition was chosen after seeing results and is reported as a sensitivity analysis.

## Semantic pilot

`datasets/semantic_tier_cases.json` holds 60 authored record pairs, each with one correct and one wrong transformation,
under a same-real-world-entity requirement. `semantic_tier.py` scores LLM checkers against deterministic baselines. The
items were authored to resist string rules, so results carry no prevalence information, and a complete lookup table
would compile the requirement away.

## Early prompt rerun

`early_prompt_rerun.py` applies the harness's early checker prompt (a vaguer rule and a single exemplar whose verdict is
`OK`, replaced in commit `443a34d`) to archived worker tables, with the ablation's rows, batches, token cap and labels:
250 rows, each checked by both models, in 14 runs. It covers all Mistral-worker seeds and one Llama-worker KramaBench seed;
the other archived Llama-worker tables do not reproduce their reports' labels. This is not the pilot notebook's prompt, which
gives a one-line rule, names both verdicts and has no exemplar.

Exact matching on `NOT_OK` gives 0 flags in all 14 runs, because the prompt never shows that token and the models wrote `NOK`,
`NG`, `Invalid`, `Warning` or `Error` instead. Counting any non-`OK` verdict as a flag, on the same rows:

| Checker | Early prompt: flagged / recall / precision | Ablation prompt: flagged / recall / precision |
|---|---|---|
| Llama | 21% / 0.22 / 0.74 | 79% / 0.79 / 0.71 |
| Mistral | 9% / 0.12 / 1.00 | 12% / 0.16 / 0.97 |

The prompts differ in rule, exemplars and vocabulary at once, so these runs cannot separate the three. Raw outputs are the
`runs/oldprompt_*` files, and the cached models are enough to run it.

## Reflexion baseline (agentic self-feedback)

`--mode reflexion` runs a same-model checker with a Reflexion-style second pass (the model is
shown its own verdicts and asked to reflect and revise). This is the agentic self-feedback
baseline: if the circular-validation hypothesis (H1) holds, self-critique alone should *not*
recover the blind-spot errors that a different checker model catches. Comparing
`circular` vs `reflexion` vs `cross-only` tests exactly that.

## One-command full evaluation

```bash
export HF_TOKEN=...            # GPU machine
./run_full_ablation.sh 1,2,3,4,5 runs
```

runs the original five configurations — {rules-only, circular, reflexion, cross-only, hybrid} × {UCI Adult,
KramaBench water-quality} × seeds — with each model loaded once per mode, then calls
`aggregate_results.py`, which writes:

- `runs/ablation_summary.md` — pooled TP/FP/FN/TN, recall/precision with exact 95% CIs
- `runs/ablation_table.tex` — a paste-ready LaTeX table for Section 6 of the paper

The rules+same-model cell (the sixth configuration) runs separately via `--mode hybrid-circular`
on the same seeds and sample sizes; the aggregator picks its reports up automatically.

No local GPU? Open **`run_ablation_kaggle.ipynb`** on Kaggle or Colab (T4 is enough), add your
`HF_TOKEN` as a secret, and run all cells; the last cell prints the LaTeX table.

Each run writes:
- `<prefix>_<mode>_seed<k>_injected_errors.json` — ground-truth fault log
- `<prefix>_<mode>_seed<k>_raw_*_batch_*.txt` — raw LLM outputs (audit trail)
- `<prefix>_<mode>_seed<k>_final_report.json` — config, ground truth, checker metrics (with CIs and per-path attribution), and row-level results

## Scope and known limitations (post-ablation)

- **Executed.** The six-configuration, three-seed checker ablation in both role assignments on UCI Adult and the KramaBench water-quality table (the paper's primary evidence; Section 6); a small authored pilot on requirements that resist compilation; and the deterministic-path consistency run over the full 320k-row KramaBench lake (Section 7). **Planned:** the CORD and NYC Open Data tiers, DS-1000.
- **Sample sizes.** The ablation checks 90 (UCI) and 120 (KramaBench) rows across 3 seeds per configuration — usable but with wide confidence intervals; the superseded single-seed pilot (N=30) is retained in the paper only as a record.
- **Fault realism.** Ablation faults are synthetic or worker-introduced. The 46% of bad rows with no injected fault applies to the full KramaBench lake, not to the single 2019 table used in the ablation (0/796 in passthrough; 1/51 with the Llama worker).
- **Prompt sensitivity.** In the semantic pilot, rewording the rule alone cut Llama's false positives from 55 to 12. All ablation findings are conditional on one fixed prompt, which does not receive the typed rules: under the early prompt, Llama flagged 21% rather than 79% of the same rows and had no parse failures.
- **Quantization & hardware.** Ablation models ran in 4-bit NF4 on a single 16 GB GPU; full-precision behavior may differ. Outputs are not bit-reproducible across hardware, which is why seeds, model commit pins (`runs/model_pins.json`), and raw outputs are archived.
- **Known engineering hazard.** Under the ablation's prompt, Llama checkers degenerated into unparsable repetition on KramaBench whichever worker they checked (85 of 1,007 Llama-checked rows across both assignments, all on KramaBench); Mistral returned a verdict for all 1,004 rows it checked.
