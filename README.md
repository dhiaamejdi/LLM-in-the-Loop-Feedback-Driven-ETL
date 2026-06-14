# LLM-in-the-Loop Feedback-Driven ETL

A modular architecture for trustworthy, self-adaptive Extract-Transform-Load (ETL) pipelines augmented by large language models (LLMs). Each ETL stage is run by a **Worker LLM** and validated by a paired **LLM-as-a-Checker**, with a feedback loop that consolidates human corrections and model-flagged discrepancies.

This repository contains the prototype and evaluation code for the paper:

> **LLM-in-the-Loop Feedback-Driven ETL: A Modular Architecture for Trustworthy, Self-Adaptive Data Pipelines**
> Dhiaa Mejdi and Adel M. Alimi. NOVAS 2026 (VLDB 2026 Workshop: Novel Optimizations for Visionary AI Systems).

The central contribution is the discovery of the **circular validation problem** — a single LLM family used as both Worker and Checker fails to flag errors characteristic of its own model — and the **Safeguarded Architecture** that mitigates it through *cross-architecture validation* (Llama Worker + Mistral Checker) and *deterministic rule hybridization* (a fast deterministic path before the expensive LLM checker).

---

## ⚠️ Please read before reviewing the code

This section exists so you don't have to reverse-engineer our design decisions. We flag every known quirk, scope limit, and intentional shortcut up front.

> **Please note (scope).** This is an **early-stage prototype**, not a production system. Only the **Low-Ambiguity tier (UCI Adult)** is implemented and evaluated. The Medium (CORD) and High (NYC Open Data) tiers described in the paper's evaluation framework are **specified but not yet run**. Every quantitative number in the paper refers to the UCI Adult prototype scope only.

> **Please note (the `resume_from_files` flag).** In `safeguarded_LLM_in_the_loop.py`, the `__main__` block calls `run_pipeline(..., resume_from_files=True)` with the value **hardcoded**, which overrides the `--resume-from-files` CLI argument. This is **intentional**: it reproduces our reported run by reconstructing the transformed data from the saved raw-output `.txt` files, rather than re-calling the Worker LLM (which requires a GPU and a valid HF token). **If you clone this repo without our intermediate `.txt` files, a resumed run will raise `FileNotFoundError`.** To run the pipeline end-to-end from scratch, set this value to `False` (see [Running](#running) below).

> **Please note (where the accuracy numbers come from).** The Worker accuracy figures in the paper (Schema Understanding 100%, Semantic Transformation 94.73%, Data Cleaning 95.47%, Feature Engineering 94.35%) are measured in the **notebook** (`LLM_in_the_Loop_Driven_ETL__3_.ipynb`), which contains deterministic **oracle** functions (`oracle_transform`, `oracle_clean`, `oracle_features`, `oracle_check_*`) used as ground truth. The standalone script `safeguarded_LLM_in_the_loop.py` implements the **Safeguarded Architecture experiment** (the 86.67% trustworthiness result and the cross-architecture validation), and does **not** recompute those per-stage accuracy numbers. The two files cover different experiments in the paper — see the [mapping table](#which-file-produces-which-result) below.

> **Please note (small N, single realization).** The Safeguarded experiment validates on a held-out sample of **N=30** transformed rows, of which **4** were flagged as True Positives. Error injection is seeded (`SEED=42`), so results are deterministic for a given seed but represent **one realization**. We do not claim statistical significance from this sample; it is an early signal. This is discussed explicitly in the paper's *Threats to Validity* section.

> **Please note (injected vs. natural errors).** Errors are **synthetically injected** (missing values, swapped columns, inconsistent casing, extraneous whitespace) via `inject_errors()`. They are not drawn from naturally occurring data faults, so the error distribution may not match real-world ETL failures.

> **Please note (the JSON-cleanup hacks).** You will see defensive code that looks unusual: a `out.replace(r'\_', '_')` line in the checker, and a hand-rolled balanced-brace parser in `extract_json_block()`. These are **not** accidental — maintaining parsable JSON output across two different model families (Llama and Mistral) was the single largest engineering hurdle, as reported in the paper's discussion. Mistral, in particular, tended to escape underscores. These hacks are the "PromptOps layer" referenced in the paper.

> **Please note (non-determinism).** LLM generation uses `temperature=0.0` for reproducibility, but LLM outputs are still not guaranteed to be bit-identical across hardware/library versions. Mean reproducibility was 0.87 cosine similarity across 10 runs.

---

## Repository contents

| File | Purpose |
|------|---------|
| `safeguarded_LLM_in_the_loop.py` | Standalone script for the **Safeguarded Architecture** experiment: cross-architecture validation (Llama Worker + Mistral Checker) and deterministic rule hybridization. Produces the 86.67% trustworthiness result. |
| `LLM_in_the_Loop_Driven_ETL__3_.ipynb` | Notebook for the **full four-stage pipeline** and the per-stage Worker/Checker accuracy measurements, including the deterministic oracle ground-truth functions. |

## Which file produces which result

| Paper claim / table | Where it is produced |
|---------------------|----------------------|
| Worker accuracy per stage (100% / 94.73% / 95.47% / 94.35%) | Notebook, via oracle comparison |
| Hallucination rate 4.2%, reproducibility 0.87 | Notebook |
| Circular validation baseline (Llama checks Llama, >60% errors missed) | Notebook baseline run |
| **Safeguarded Architecture: 4 True Positives, 86.67% (Table 4)** | `safeguarded_LLM_in_the_loop.py` |
| Cross-architecture validation (Mistral catches Llama errors) | `safeguarded_LLM_in_the_loop.py` (`hybrid_semantic_checker`) |
| Deterministic rule hybridization (fast/slow path) | `safeguarded_LLM_in_the_loop.py` (`deterministic_check_row` → LLM batch) |
| Cost analysis (Table 2) | Estimated from token counts; not auto-generated by the script |

---

## Setup

Requires Python 3.9+ and a GPU for the LLM stages (the models are `Llama-3.1-8B-Instruct` and `Mistral-7B-Instruct-v0.2`, both gated on Hugging Face).

```bash
pip install torch transformers huggingface_hub pandas numpy
```

### Hugging Face authentication

The models are gated, so you need an HF account with access granted to both models, and an access token.

> **Please note (security).** Do **not** hardcode your token in the source. Provide it via an environment variable or a Kaggle Secret. Set it like this:

```bash
export HF_TOKEN="your_token_here"
```

The script and notebook both read `os.environ.get("HF_TOKEN")`.

## Running

### Reproduce our reported Safeguarded result (resumed mode)

This requires the intermediate `results_safeguarded_raw_transform_batch_*_0.txt` files to be present in the working directory.

```bash
python safeguarded_LLM_in_the_loop.py
```

### Run the Safeguarded pipeline from scratch (fresh worker calls)

Open `safeguarded_LLM_in_the_loop.py`, find the `run_pipeline(...)` call at the bottom, and change:

```python
resume_from_files=True
```

to:

```python
resume_from_files=False
```

Then run the same command. This will call the Llama Worker live (GPU + HF access required) and regenerate the transformed data before validation.

### Run the full four-stage pipeline and accuracy measurements

Open and run `LLM_in_the_Loop_Driven_ETL__3_.ipynb` (e.g. on Kaggle with a GPU). Set `HF_TOKEN` as a Kaggle Secret first.

---

## Architecture summary

The Safeguarded pipeline runs in three phases:

1. **Semantic Transformation** — the Worker LLM (Llama) applies business rules (whitespace trimming, income-label normalization, sex capitalization) batch by batch.
2. **Hybrid Validation** — every sampled row first passes a fast deterministic check (`deterministic_check_row`); only rows that pass are batched to the Checker LLM (Mistral) for semantic validation. This is the fast-path/slow-path design.
3. **Feedback-Driven Refinement** — verdicts are aggregated; human corrections and Checker discrepancies feed a central knowledge base (the feedback loop; full closed-loop fine-tuning is future work).

## Known limitations

These are stated in full in the paper's *Architectural Limitations* and *Threats to Validity* sections, and summarized here:

- **Computational cost** — each stage is a separate LLM call; best suited to lower-volume pipelines.
- **Prompt sensitivity** — output quality depends heavily on prompt precision (hence the PromptOps layer).
- **Data privacy** — reliance on external/hosted LLMs raises GDPR/CCPA concerns without on-prem deployment.
- **Single model pair** — only Llama/Mistral tested; the cross-architecture benefit may be specific to this pairing. The "universal" character of the circular validation problem is stated as a hypothesis, not a proven result.
- **Single dataset tier** — only UCI Adult executed.

## Citation

```bibtex
@inproceedings{mejdi2026llmloop,
  title     = {LLM-in-the-Loop Feedback-Driven ETL: A Modular Architecture for Trustworthy, Self-Adaptive Data Pipelines},
  author    = {Mejdi, Dhiaa and Alimi, Adel M.},
  booktitle = {VLDB 2026 Workshop: Novel Optimizations for Visionary AI Systems (NOVAS)},
  year      = {2026}
}
```
