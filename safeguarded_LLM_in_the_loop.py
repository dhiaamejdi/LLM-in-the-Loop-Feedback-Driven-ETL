#!/usr/bin/env python3
"""
Safeguarded LLM-in-the-Loop ETL — reproducible evaluation harness.

Implements the Safeguarded Architecture from:
  "LLM-in-the-Loop Feedback-Driven ETL: Toward Trustworthy, Self-Adaptive
   Data Pipelines" (Anonymous).

Key properties (added in the post-review revision):

1. Ablation modes (--mode), so the contribution of each safeguard is
   attributable rather than confounded:
     * circular    : same-family checker (Worker model checks itself), no rules
     * cross-only  : different-family checker, no deterministic rules
     * rules-only  : deterministic fast-path checks only, no LLM checker
     * hybrid      : deterministic fast path + different-family LLM checker
                     (the full Safeguarded Architecture)

2. Ground-truth metrics. The error-injection harness logs every injected
   fault as (row, column, error_type). After the Worker transformation, a
   deterministic oracle recomputes the correct output for every row, so each
   checked row has a known ground-truth label ("residual error" vs "clean").
   Checker verdicts are scored against this label: TP / FP / FN / TN,
   recall, precision, and exact Clopper–Pearson 95% confidence intervals.

3. Per-path attribution. Every verdict is tagged checker="deterministic" or
   checker="llm", and the final report breaks results down per path.

4. Full seeding (--seed) and multi-seed support (--seeds "1,2,3").

5. No secrets in source. The HF token is read from the HF_TOKEN environment
   variable or --hf-token. NEVER hardcode tokens.

Usage examples:
  # Full safeguarded run (needs GPU + gated-model access):
  python safeguarded_LLM_in_the_loop.py --mode hybrid

  # Circular-validation baseline (Llama checks Llama):
  python safeguarded_LLM_in_the_loop.py --mode circular

  # Deterministic rules only (no LLM checker; runs without a checker model):
  python safeguarded_LLM_in_the_loop.py --mode rules-only

  # Reproduce the reported run from saved raw Worker outputs (no GPU needed
  # for the transform stage):
  python safeguarded_LLM_in_the_loop.py --mode hybrid --resume-from-files

  # 2x2 ablation across 10 seeds:
  for m in circular cross-only rules-only hybrid; do
    python safeguarded_LLM_in_the_loop.py --mode $m --seeds 1,2,3,4,5,6,7,8,9,10
  done
"""

import argparse
import glob
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# -------------------------
# Config / Constants
# -------------------------
UCI_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_HELDOUT_SIZE = 30
DEFAULT_SEED = 42
BATCH_SIZE = 20
CHECKER_BATCH_SIZE = 5

DEFAULT_WORKER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CHECKER = "mistralai/Mistral-7B-Instruct-v0.2"

# Business rules are TYPED so that the LLM prompt, the deterministic fast-path
# checker, and the ground-truth oracle are all derived from the SAME definition
# and can never drift apart. Supported types:
#   trim_whitespace                  — trim all string columns
#   strip_suffix   (column, suffix)  — remove a trailing suffix from values
#   capitalize     (column)          — 'male' -> 'Male'
#   uppercase      (column)          — 'yes' -> 'YES'
#   titlecase      (column)          — 'south boston' -> 'South Boston'
#   canonical_values (column, allowed) — value must be in the allowed set
BUSINESS_RULES: List[Dict[str, Any]] = [
    {"type": "trim_whitespace",
     "description": "Trim whitespace in all string columns"},
    {"type": "strip_suffix", "column": "income", "suffix": ".",
     "description": "Normalize income labels by removing a trailing '.' (e.g. '>50K.' becomes '>50K')"},
    {"type": "canonical_values", "column": "income", "allowed": [">50K", "<=50K"],
     "description": "Income labels must be exactly '>50K' or '<=50K'"},
    {"type": "capitalize", "column": "sex",
     "description": "Map sex to 'Male'/'Female' (capitalize)"},
]

MODES = ("hybrid", "circular", "cross-only", "rules-only", "reflexion", "hybrid-circular")
# reflexion: same-family checker with Reflexion-style self-critique (verdict pass,
# then a reflection pass asking the model to re-examine and revise its verdicts).
# This is the agentic self-feedback baseline: if H1 holds, self-reflection alone
# should NOT recover the blind-spot errors that a different-family checker catches.


# -------------------------
# Statistics helpers
# -------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    """Numerically stable binomial CDF via log-space terms (handles large n)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    lp, lq = math.log(p), math.log1p(-p)
    for i in range(0, k + 1):
        lterm = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 + i * lp + (n - i) * lq)
        total += math.exp(lterm)
    return min(total, 1.0)


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper–Pearson) two-sided CI for a binomial proportion.

    Pure-Python bisection on the binomial CDF; no scipy dependency.
    """
    if n == 0:
        return (0.0, 1.0)

    def _solve(fn, lo=0.0, hi=1.0, iters=200):
        for _ in range(iters):
            mid = (lo + hi) / 2
            if fn(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    if x == 0:
        lower = 0.0
    else:
        # largest p with P(X >= x | p) <= alpha/2  -> lower bound
        lower = _solve(lambda p: 1 - _binom_cdf(x - 1, n, p) <= alpha / 2)
    if x == n:
        upper = 1.0
    else:
        upper = 1.0 - _solve(lambda p: 1 - _binom_cdf(n - x - 1, n, p) <= alpha / 2)
    return (round(lower, 4), round(upper, 4))


# -------------------------
# Metrics
# -------------------------
@dataclass
class CheckerMetrics:
    """Checker verdicts scored against ground-truth row labels."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    per_path: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, verdict_not_ok: bool, truth_bad: bool, path: str):
        key = ("tp" if truth_bad else "fp") if verdict_not_ok else ("fn" if truth_bad else "tn")
        setattr(self, key, getattr(self, key) + 1)
        self.per_path.setdefault(path, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        self.per_path[path][key] += 1

    def to_dict(self) -> Dict[str, Any]:
        pos = self.tp + self.fn
        flagged = self.tp + self.fp
        recall = self.tp / pos if pos else None
        precision = self.tp / flagged if flagged else None
        d = {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "ground_truth_positives": pos,
            "rows_flagged": flagged,
            "recall": recall,
            "recall_ci95": clopper_pearson(self.tp, pos) if pos else None,
            "precision": precision,
            "precision_ci95": clopper_pearson(self.tp, flagged) if flagged else None,
            "per_path": self.per_path,
        }
        return d


# -------------------------
# Utility functions
# -------------------------
def is_string_col(s: pd.Series) -> bool:
    """True for text columns under both classic (object) and modern (str/string) pandas dtypes."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_uci_adult() -> pd.DataFrame:
    read_kwargs = dict(names=COLUMNS, sep=r",\s*", engine="python", na_values="?", header=None)
    train_df = pd.read_csv(UCI_TRAIN, **read_kwargs)
    test_df = pd.read_csv(UCI_TEST, skiprows=1, **read_kwargs)
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    # The test split suffixes income labels with '.', a real inconsistency we keep.
    return df


def load_dataset(data_csv: Optional[str]) -> pd.DataFrame:
    """Load either the default UCI Adult benchmark or an arbitrary CSV.

    Passing --data-csv lets the same harness run on any tabular dataset
    (e.g., NYC Open Data exports, KramaBench source files), addressing the
    single-dataset limitation of the original prototype. Note that the
    default BUSINESS_RULES (income/sex normalization) are UCI-specific;
    for other datasets, override them with --rules-json.
    """
    if data_csv:
        if any(ch in data_csv for ch in "*?["):
            files = sorted(glob.glob(data_csv))
            if not files:
                raise FileNotFoundError(f"No files match {data_csv}")
            print(f"[Pipeline] Loading {len(files)} files matching {data_csv}")
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            print(f"[Pipeline] Combined dataset: {len(df):,} rows x {df.shape[1]} columns")
            return df
        print(f"[Pipeline] Loading custom dataset: {data_csv}")
        return pd.read_csv(data_csv)
    return load_uci_adult()


KNOWN_RULE_TYPES = {"trim_whitespace", "strip_suffix", "capitalize", "uppercase",
                    "titlecase", "canonical_values"}


def load_business_rules(rules_json: Optional[str]) -> List[Dict[str, Any]]:
    """Optionally override BUSINESS_RULES with a JSON file: a list of typed rules."""
    if not rules_json:
        return BUSINESS_RULES
    with open(rules_json) as f:
        rules = json.load(f)
    assert isinstance(rules, list), "--rules-json must be a JSON list of typed rule objects"
    for r in rules:
        assert isinstance(r, dict) and r.get("type") in KNOWN_RULE_TYPES, \
            f"Unknown or missing rule type in {r!r}; supported: {sorted(KNOWN_RULE_TYPES)}"
        assert "description" in r, f"Rule {r!r} needs a natural-language 'description' for the prompt"
    print(f"[Pipeline] Loaded {len(rules)} typed business rules from {rules_json}")
    return rules


def rules_prompt_lines(rules: List[Dict[str, Any]]) -> str:
    return "\n".join(f"- {r['description']}" for r in rules)


def inject_errors(
    df: pd.DataFrame, error_config: Dict[str, float], seed: int
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Inject synthetic errors and return (corrupted_df, error_log).

    The error log records every injected fault as
    {"row": r, "column": c, "type": t} — this is the ground truth used to
    score checker verdicts.
    """
    df = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    error_log: List[Dict[str, Any]] = []
    nrows, ncols = df.shape

    # 1. Missing values
    m_count = int(nrows * ncols * error_config.get("missing_value", 0.03))
    for _ in range(m_count):
        r, c = int(rng.integers(0, nrows)), int(rng.integers(0, ncols))
        df.iat[r, c] = np.nan
        error_log.append({"row": r, "column": df.columns[c], "type": "missing_value"})

    # 2. Swapped categorical columns
    s_rows = int(nrows * error_config.get("swap_columns", 0.02))
    cat_cols = [c for c in df.columns if is_string_col(df[c])]
    for _ in range(s_rows):
        if len(cat_cols) >= 2:
            r = int(rng.integers(0, nrows))
            c1, c2 = rng.choice(cat_cols, size=2, replace=False)
            df.at[r, c1], df.at[r, c2] = df.at[r, c2], df.at[r, c1]
            error_log.append({"row": r, "column": f"{c1}<->{c2}", "type": "swap_columns"})

    # 3. Bad casing / 4. Stray whitespace — independent draws per column
    #    (the pre-revision code reused the same random_state for both, which
    #    silently corrupted the SAME rows twice; fixed by drawing from rng).
    bc_frac = error_config.get("bad_casing", 0.03)
    ws_frac = error_config.get("stray_whitespace", 0.03)
    for c in df.columns:
        if not is_string_col(df[c]):
            continue
        valid_idx = df.index[df[c].notna()].to_numpy()
        if len(valid_idx) == 0:
            continue
        bc_n = int(round(len(valid_idx) * bc_frac))
        ws_n = int(round(len(valid_idx) * ws_frac))
        bc_idx = rng.choice(valid_idx, size=min(bc_n, len(valid_idx)), replace=False)
        ws_idx = rng.choice(valid_idx, size=min(ws_n, len(valid_idx)), replace=False)
        for r in bc_idx:
            df.at[r, c] = str(df.at[r, c]).lower()
            error_log.append({"row": int(r), "column": c, "type": "bad_casing"})
        for r in ws_idx:
            df.at[r, c] = " " + str(df.at[r, c]) + " "
            error_log.append({"row": int(r), "column": c, "type": "stray_whitespace"})

    return df, error_log


# -------------------------
# Deterministic oracle (ground truth for the transformation)
# -------------------------
def oracle_transform_row(row: pd.Series, rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Deterministic reference implementation of the typed business rules.

    Driven by the same rule objects as the prompt and the fast path, so ground
    truth can never drift from what the Worker was asked to do.
    """
    rules = rules if rules is not None else BUSINESS_RULES
    out: Dict[str, Any] = {k: v for k, v in row.items()}
    for r in rules:
        rtype = r["type"]
        if rtype == "trim_whitespace":
            for col, val in out.items():
                if isinstance(val, str):
                    out[col] = val.strip()
        elif rtype == "canonical_values":
            continue  # constraint, not a transformation
        else:
            col = r["column"]
            val = out.get(col)
            if not isinstance(val, str):
                continue
            if rtype == "strip_suffix":
                v = val.strip()
                while v.endswith(r["suffix"]):
                    v = v[: -len(r["suffix"])].strip()
                out[col] = v
            elif rtype == "capitalize":
                out[col] = val.strip().capitalize()
            elif rtype == "uppercase":
                out[col] = val.strip().upper()
            elif rtype == "titlecase":
                out[col] = val.strip().title()
    return out


def row_has_residual_error(
    transformed_row: pd.Series, source_row: pd.Series,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[str]]:
    """Ground-truth label: does the Worker's output differ from the oracle?

    Returns (is_bad, list_of_reasons). Only rule-relevant fields are compared,
    so the label is independent of any LLM.
    """
    oracle = oracle_transform_row(source_row, rules)
    reasons = []
    # A row is also ground-truth bad if it violates any constraint-type rule
    # (canonical_values): e.g., a swapped-in value in a categorical column.
    # Such faults cannot be repaired by transformation rules, so the oracle
    # preserves them — but the row is still invalid and SHOULD be flagged.
    for r in (rules if rules is not None else BUSINESS_RULES):
        if r["type"] == "canonical_values":
            val = transformed_row.get(r["column"])
            if isinstance(val, str) and val not in r["allowed"] + ["Unknown"]:
                reasons.append(f"{r['column']}: {val!r} violates canonical set {r['allowed']}")
    for col, expected in oracle.items():
        got = transformed_row.get(col)
        if pd.isna(expected) and pd.isna(got):
            continue
        if isinstance(expected, str) or isinstance(got, str):
            if str(got) != str(expected):
                reasons.append(f"{col}: expected {expected!r}, got {got!r}")
        else:
            try:
                if not (pd.isna(expected) and pd.isna(got)) and float(got) != float(expected):
                    reasons.append(f"{col}: expected {expected!r}, got {got!r}")
            except (TypeError, ValueError):
                if got != expected:
                    reasons.append(f"{col}: expected {expected!r}, got {got!r}")
    return (len(reasons) > 0, reasons)


# -------------------------
# JSON extraction / serialization
# -------------------------
def make_serializable(data_dict: Dict) -> Dict:
    sanitized = {}
    for key, value in data_dict.items():
        if pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (np.integer,)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating,)):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized


def extract_json_block(text: str):
    """Finds and parses the first valid JSON object or array in a string.

    Cross-family JSON robustness (Llama vs. Mistral output quirks) is the
    'PromptOps layer' discussed in the paper.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    first = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if first == -1:
        raise ValueError("no JSON block found (no starting '{' or '[')")
    search_text = text[first:]
    try:
        return json.loads(search_text)
    except json.JSONDecodeError:
        pass
    start_char = search_text[0]
    end_char = "}" if start_char == "{" else "]"
    depth = 0
    for i, char in enumerate(search_text):
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(search_text[: i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError("no valid, balanced JSON block found")


# -------------------------
# LLM Client
# -------------------------
_CLIENT_CACHE: Dict[str, "LLMClient"] = {}


def _gpu_total_gb() -> float:
    if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def resolve_quantization(quantization: str) -> str:
    """'auto' -> 4bit on small GPUs (e.g. Kaggle/Colab T4, 16 GB), none otherwise.

    An 8B model in fp16 (~16 GB) does not fit a T4 alongside its KV cache, and
    two 8B models never fit; in 4-bit NF4 each is ~5.5 GB so worker + checker
    coexist comfortably. Requires bitsandbytes.
    """
    if quantization != "auto":
        return quantization
    if not (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()):
        return "none"
    if _gpu_total_gb() < 24:
        try:
            import bitsandbytes  # noqa: F401
            return "4bit"
        except ImportError:
            print("[LLM Client] WARNING: GPU < 24 GB and bitsandbytes not installed; "
                  "falling back to fp16 with single-model cache eviction. "
                  "`pip install bitsandbytes` is strongly recommended.")
            return "none"
    return "none"


def get_llm_client(hf_token: str, model_name: str, quantization: str = "auto") -> "LLMClient":
    """Cache clients across seeds/modes so each model loads only once per process.

    In fp16 on a small GPU, only one model fits: evict cached models before
    loading a different one. In 4/8-bit, worker and checker can coexist.
    """
    q = resolve_quantization(quantization)
    if model_name in _CLIENT_CACHE:
        return _CLIENT_CACHE[model_name]
    if q == "none" and _CLIENT_CACHE and _gpu_total_gb() and _gpu_total_gb() < 40:
        print(f"[LLM Client] Evicting cached model(s) {list(_CLIENT_CACHE)} to free GPU memory (fp16 mode).")
        import gc
        for k in list(_CLIENT_CACHE):
            del _CLIENT_CACHE[k].model
            del _CLIENT_CACHE[k]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _CLIENT_CACHE[model_name] = LLMClient(hf_token=hf_token, model_name=model_name, quantization=q)
    return _CLIENT_CACHE[model_name]


class LLMClient:
    def __init__(self, hf_token: str, model_name: str, quantization: str = "none"):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not available. pip install -r requirements.txt")
        if not hf_token:
            raise RuntimeError("HF token required for LLM mode (set HF_TOKEN env var).")
        print(f"[LLM Client] Loading tokenizer and model: {model_name} (quantization={quantization})")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: Dict[str, Any] = dict(device_map="auto", token=hf_token, low_cpu_mem_usage=True)
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
                if quantization == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def call(self, prompt: str, max_length: int = 1024, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        if "mistral" in self.model_name.lower():
            full_prompt = f"<s>[INST] {prompt} [/INST]"
            enc = self.tokenizer(full_prompt, return_tensors="pt")
        else:  # Llama-3 chat template
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        # Newer transformers return a BatchEncoding (dict-like); older return a bare
        # tensor from apply_chat_template. Normalize to (input_ids, attention_mask).
        if hasattr(enc, "keys"):  # BatchEncoding / dict
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask")
            attention_mask = attention_mask.to(self.model.device) if attention_mask is not None else None
        else:  # bare tensor
            input_ids = enc.to(self.model.device)
            attention_mask = None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        terminators = [self.tokenizer.eos_token_id]
        if "Llama-3" in self.model_name:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                eos_token_id=terminators,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)


# -------------------------
# Worker: semantic transformation stage
# -------------------------
def llm_transform_stage(
    client: LLMClient,
    df_in: pd.DataFrame,
    outfile_prefix: str,
    business_rules: List[Dict[str, Any]],
    batch_size: int = BATCH_SIZE,
    retries: int = 2,
    max_new_tokens: int = 2048,
) -> pd.DataFrame:
    """Worker transformation with truncation-proof batching.

    If a batch's output cannot be parsed as JSON (the usual cause is the
    generation hitting the token cap mid-JSON), the batch is SPLIT IN HALF and
    each half retried, recursively down to single rows. A row falls back to its
    untransformed original only if even its single-row generation is unparsable,
    and every fallback is counted and reported loudly — a silently passthrough
    "worker" would invalidate the experiment.
    """
    n = len(df_in)
    n_batches = (n + batch_size - 1) // batch_size
    fallback_rows = [0]

    def build_prompt(batch: pd.DataFrame) -> str:
        return (
            "Transform rows according to the business rules and return a JSON object like "
            "{'rows':[ {...}, {...} ]}.\n"
            "Business rules:\n" + rules_prompt_lines(business_rules) + "\n\n"
            "Context CSV:\n" + batch.to_csv(index=False) + "\n\n"
            "IMPORTANT: You MUST return only the JSON object. Do not include any introductory "
            "text or explanations. Your entire response must be the JSON itself, starting with '{'."
        )

    def transform_chunk(batch: pd.DataFrame, tag: str) -> List[Dict]:
        prompt = build_prompt(batch)
        parsed_rows = None
        for attempt in range(retries):
            out = client.call(prompt, max_length=max_new_tokens, temperature=0.0)
            with open(f"{outfile_prefix}_raw_transform_batch_{tag}_{attempt}.txt", "w", encoding="utf-8") as f:
                f.write(out)
            # Truncation heuristic: output that never closes its JSON almost always
            # means the token cap was hit. Splitting beats retrying the same prompt.
            try:
                parsed = extract_json_block(out)
                parsed_rows = parsed.get("rows") if isinstance(parsed, dict) else (
                    parsed if isinstance(parsed, list) else None
                )
                if parsed_rows is not None:
                    break
            except Exception as e:
                print(f"  chunk {tag} attempt {attempt + 1} unparsable ({e})", flush=True)
                parsed_rows = None
                if len(batch) > 1:
                    break  # don't waste a retry on a likely-truncated prompt; split instead
        if parsed_rows is None:
            if len(batch) > 1:
                mid = len(batch) // 2
                print(f"  splitting chunk {tag} ({len(batch)} rows) -> halves", flush=True)
                return (transform_chunk(batch.iloc[:mid], f"{tag}a")
                        + transform_chunk(batch.iloc[mid:], f"{tag}b"))
            fallback_rows[0] += 1
            print(f"  WARNING: single-row chunk {tag} unparsable after {retries} attempts; "
                  f"falling back to original row.", flush=True)
            return [batch.iloc[0].to_dict()]
        out_rows = []
        for i in range(len(batch)):
            if i < len(parsed_rows) and isinstance(parsed_rows[i], dict):
                out_rows.append(parsed_rows[i])
            else:
                fallback_rows[0] += 1
                out_rows.append(batch.iloc[i].to_dict())
        return out_rows

    rows_out: List[Dict] = []
    for start in range(0, n, batch_size):
        batch = df_in.iloc[start:start + batch_size]
        t0 = time.time()
        print(f"[Worker LLM: {client.model_name}] Transforming rows {start}..{start + len(batch) - 1} "
              f"(batch {start // batch_size + 1}/{n_batches})", flush=True)
        rows_out.extend(transform_chunk(batch, str(start)))
        print(f"  batch done in {time.time() - t0:.1f}s", flush=True)
    if fallback_rows[0]:
        pct = 100.0 * fallback_rows[0] / n
        print(f"[Worker LLM] WARNING: {fallback_rows[0]}/{n} rows ({pct:.1f}%) fell back to originals. "
              f"If this exceeds a few percent, results reflect a partially-passthrough worker — "
              f"reduce --batch-size or raise --max-new-tokens and rerun.", flush=True)
    else:
        print(f"[Worker LLM] All {n} rows transformed by the model (0 fallbacks).", flush=True)
    return pd.DataFrame(rows_out)


def reconstruct_transformed_df_from_files(outfile_prefix: str) -> pd.DataFrame:
    """Rebuild the transformed DataFrame from saved raw Worker outputs.

    Enables re-running validation/ablation without a GPU, from the archived
    `<prefix>_raw_transform_batch_*_0.txt` files shipped with the repo.
    """
    print(f"[Resume] Reconstructing transformed DataFrame from files: {outfile_prefix}_raw_transform_batch_*_0.txt")
    transform_files = glob.glob(f"{outfile_prefix}_raw_transform_batch_*_0.txt")
    if not transform_files:
        raise FileNotFoundError(
            f"No transform files found matching {outfile_prefix}_raw_transform_batch_*_0.txt. "
            "Run once without --resume-from-files first (GPU required), or download the "
            "archived outputs from the repository release."
        )
    def _batch_key(f):
        mm = re.search(r"_batch_([0-9]+)([ab]*)_", f)
        return (int(mm.group(1)), mm.group(2)) if mm else (10**9, "")
    transform_files.sort(key=_batch_key)
    all_rows: List[Dict] = []
    for file_path in transform_files:
        parsed_rows = None
        for candidate in (file_path, file_path.replace("_0.txt", "_1.txt")):
            if not os.path.exists(candidate):
                continue
            with open(candidate, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                parsed = extract_json_block(content)
                cand_rows = parsed.get("rows") if isinstance(parsed, dict) else parsed
                if isinstance(cand_rows, list):
                    parsed_rows = cand_rows
                    break
            except Exception:
                continue
        if parsed_rows is not None:
            all_rows.extend(parsed_rows)
        else:
            print(f"  WARNING: no parsable attempt for {file_path}; rows from this chunk are lost — "
                  f"prefer the consolidated *_transformed.csv cache.")
    if not all_rows:
        raise ValueError("Could not reconstruct any rows from the saved files.")
    print(f"[Resume] Reconstructed {len(all_rows)} rows.")
    return pd.DataFrame(all_rows)


# -------------------------
# Checkers
# -------------------------
def deterministic_check_row(row: pd.Series, rules: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, str]:
    """Fast path: compiled, rule-based checks (Safeguard 2), derived from the
    same typed rule objects as the prompt and the oracle."""
    rules = rules if rules is not None else BUSINESS_RULES
    for r in rules:
        rtype = r["type"]
        if rtype == "trim_whitespace":
            for col, val in row.items():
                if isinstance(val, str) and (val != val.strip()):
                    return False, f"DETERMINISTIC_FAIL: Whitespace not trimmed in column '{col}'"
        elif rtype == "canonical_values":
            col, val = r["column"], row.get(r["column"])
            if isinstance(val, str) and val not in r["allowed"] + ["Unknown"]:
                return False, f"DETERMINISTIC_FAIL: {col} '{val}' not in allowed set {r['allowed']}"
        else:
            col = r["column"]
            val = row.get(col)
            if not isinstance(val, str):
                continue
            if rtype == "strip_suffix" and val.strip().endswith(r["suffix"]):
                return False, f"DETERMINISTIC_FAIL: {col} '{val}' retains suffix '{r['suffix']}'"
            if rtype == "capitalize" and val != val.capitalize():
                return False, f"DETERMINISTIC_FAIL: {col} '{val}' not capitalized"
            if rtype == "uppercase" and val != val.upper():
                return False, f"DETERMINISTIC_FAIL: {col} '{val}' not uppercase"
            if rtype == "titlecase" and val != val.title():
                return False, f"DETERMINISTIC_FAIL: {col} '{val}' not title-case"
    return True, "DETERMINISTIC_OK"


def llm_check_batches(
    checker_client: LLMClient,
    original_rows: pd.DataFrame,
    transformed_rows: pd.DataFrame,
    idxs: List[int],
    outfile_prefix: str,
    reflexion: bool = False,
) -> List[Dict]:
    """Slow path: batched semantic validation by the Checker LLM.

    With reflexion=True, a second Reflexion-style pass shows the model its own
    verdicts and asks it to reflect on likely mistakes and output revised
    verdicts. This implements the self-feedback baseline (Shinn et al. 2023
    style) for comparison against cross-architecture validation.
    """
    verdicts: List[Dict] = []
    for start in range(0, len(idxs), CHECKER_BATCH_SIZE):
        batch_idxs = idxs[start:start + CHECKER_BATCH_SIZE]
        cases = [
            {
                "row_idx": int(idx),
                "original": make_serializable(original_rows.iloc[idx].to_dict()),
                "transformed": make_serializable(transformed_rows.iloc[idx].to_dict()),
            }
            for idx in batch_idxs
        ]
        prompt = (
            "You are a silent, precise JSON-generating API. You will be given a rule and a JSON "
            "array of data cases. For each case, decide whether the 'transformed' data is a valid "
            "transformation of the 'original' data according to the rule. Your entire response "
            "MUST be a single JSON array, one object per case, no other text or markdown.\n\n"
            "## RULE:\n"
            "Compare every field of 'transformed' against 'original' character by character. "
            "Return verdict NOT_OK if ANY field has: leading/trailing whitespace that was not "
            "trimmed; a label that was not normalized (e.g. a trailing period, or wrong casing "
            "such as lowercase where canonical casing is expected); or a value that does not "
            "belong in its column. Return OK only if EVERY field is correct.\n\n"
            "## EXAMPLE:\n### INPUT:\n"
            "[\n  {\n"
            '    "row_idx": 998,\n'
            '    "original": {"name": " john ", "income": "<=50K."},\n'
            '    "transformed": {"name": "john", "income": "<=50K"}\n'
            "  },\n  {\n"
            '    "row_idx": 999,\n'
            '    "original": {"name": " mary ", "income": ">50K."},\n'
            '    "transformed": {"name": " mary", "income": ">50K."}\n'
            "  }\n]\n"
            "### CORRECT OUTPUT:\n"
            "[\n  {\n"
            '    "row_idx": 998,\n'
            '    "verdict": "OK",\n'
            '    "notes": "Whitespace trimmed and income label normalized correctly."\n'
            "  },\n  {\n"
            '    "row_idx": 999,\n'
            '    "verdict": "NOT_OK",\n'
            '    "notes": "Leading whitespace in name was not trimmed; income label retains trailing period."\n'
            "  }\n]\n\n"
            "## ACTUAL TASK:\n### INPUT:\n"
            f"{json.dumps(cases, indent=2)}\n"
            "### CORRECT OUTPUT:"
        )
        print(f"[Checker LLM: {checker_client.model_name}] Validating batch of {len(cases)} rows...")
        t0 = time.time()
        out = checker_client.call(prompt, max_length=1024, temperature=0.0)
        # PromptOps cleanup: Mistral escapes underscores in JSON keys.
        out = out.replace(r"\_", "_")
        print(f"  checker batch done in {time.time() - t0:.1f}s", flush=True)
        with open(f"{outfile_prefix}_raw_llm_checks_batch_{start}.txt", "w", encoding="utf-8") as f:
            f.write(out)
        try:
            batch_verdicts = extract_json_block(out)
            if not isinstance(batch_verdicts, list):
                print(f"  WARNING: LLM output for batch {start} was not a list; skipping.")
                continue
        except Exception as e:
            print(f"  WARNING: failed to parse LLM checker output for batch {start}: {e}")
            continue

        if reflexion:
            reflect_prompt = (
                "You are a meticulous data-validation agent reviewing YOUR OWN previous verdicts. "
                "Below are data cases and the verdicts you produced. Reflect carefully: for each case, "
                "re-examine the 'transformed' values character by character for untrimmed whitespace, "
                "non-normalized labels, wrong casing, or values that do not belong in their column. "
                "If your previous verdict was wrong, correct it. "
                "Your entire response MUST be a single JSON array with one object per case, each with "
                "keys 'row_idx', 'verdict' ('OK' or 'NOT_OK'), and 'notes'. No other text.\n\n"
                "## CASES:\n" + json.dumps(cases, indent=2) + "\n\n"
                "## YOUR PREVIOUS VERDICTS:\n" + json.dumps(batch_verdicts, indent=2) + "\n\n"
                "## REVISED VERDICTS:"
            )
            out2 = checker_client.call(reflect_prompt, max_length=1024, temperature=0.0)
            out2 = out2.replace(r"\_", "_")
            with open(f"{outfile_prefix}_raw_reflexion_batch_{start}.txt", "w", encoding="utf-8") as f:
                f.write(out2)
            try:
                revised = extract_json_block(out2)
                if isinstance(revised, list) and revised:
                    batch_verdicts = revised
            except Exception as e:
                print(f"  WARNING: reflexion pass unparsable for batch {start} ({e}); keeping first-pass verdicts.")

        for v in batch_verdicts:
            v["checker"] = "llm-reflexion" if reflexion else "llm"
        verdicts.extend(batch_verdicts)
    return verdicts


def run_checker(
    mode: str,
    checker_client: Optional[LLMClient],
    original_rows: pd.DataFrame,
    transformed_rows: pd.DataFrame,
    sample_idxs: List[int],
    outfile_prefix: str,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    """Dispatch validation according to the ablation mode."""
    use_rules = mode in ("hybrid", "rules-only", "hybrid-circular")
    use_llm = mode in ("hybrid", "cross-only", "circular", "reflexion", "hybrid-circular")

    verdicts: List[Dict] = []
    rows_for_llm = []
    if use_rules:
        print(f"[Checker] Deterministic fast path on {len(sample_idxs)} rows...")
        for idx in sample_idxs:
            ok, reason = deterministic_check_row(transformed_rows.iloc[idx], rules)
            if not ok:
                verdicts.append({"row_idx": int(idx), "verdict": "NOT_OK", "notes": reason, "checker": "deterministic"})
            else:
                rows_for_llm.append(idx)
        print(f"[Checker] {len(sample_idxs) - len(rows_for_llm)} rows failed deterministic checks.")
    else:
        rows_for_llm = list(sample_idxs)

    if use_llm and rows_for_llm:
        verdicts.extend(
            llm_check_batches(checker_client, original_rows, transformed_rows, rows_for_llm,
                              outfile_prefix, reflexion=(mode == "reflexion"))
        )
    elif not use_llm:
        # rules-only mode: rows passing deterministic checks are marked OK.
        for idx in rows_for_llm:
            verdicts.append({"row_idx": int(idx), "verdict": "OK", "notes": "rules-only mode", "checker": "deterministic"})
    return verdicts


# -------------------------
# Orchestration
# -------------------------
def run_pipeline(
    hf_token: str,
    worker_model: str,
    checker_model: str,
    outfile_prefix: str,
    mode: str,
    seed: int,
    sample_size: int,
    heldout_size: int,
    resume_from_files: bool,
    data_csv: Optional[str] = None,
    business_rules: Optional[List[Dict[str, Any]]] = None,
    worker_mode: str = "llm",
    quantization: str = "auto",
    max_new_tokens: int = 2048,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    assert mode in MODES, f"mode must be one of {MODES}"
    seed_everything(seed)
    run_prefix = f"{outfile_prefix}_{mode}_seed{seed}"

    print(f"\n=== Safeguarded pipeline | mode={mode} | seed={seed} ===")
    business_rules = business_rules or BUSINESS_RULES
    print("[Pipeline] Loading and preparing dataset...")
    df_full = load_dataset(data_csv)
    df_sample = df_full.sample(min(sample_size, len(df_full)), random_state=seed).reset_index(drop=True)
    df_with_errors, error_log = inject_errors(
        df_sample,
        {"missing_value": 0.03, "swap_columns": 0.02, "bad_casing": 0.03, "stray_whitespace": 0.03},
        seed=seed,
    )
    with open(f"{run_prefix}_injected_errors.json", "w") as f:
        json.dump(error_log, f, indent=2)
    print(f"[Pipeline] Injected {len(error_log)} faults (ground-truth log saved).")

    # ---- Stage: Worker semantic transformation ----
    # The transform depends only on (dataset, rules, seed) — NOT on the checker
    # mode — so raw worker outputs are written under a mode-independent prefix
    # and automatically reused by every other mode with the same seed. This cuts
    # the dominant GPU cost of an ablation matrix by ~the number of modes.
    transform_prefix = f"{outfile_prefix}_seed{seed}"
    print(f"\n--- [Stage] Semantic Transformation (Worker: {worker_mode}) ---")

    def _try_reuse_cached_transform() -> Optional[pd.DataFrame]:
        json_path = f"{transform_prefix}_transformed.json"
        if os.path.exists(json_path):
            with open(json_path) as jf:
                cached = pd.DataFrame(json.load(jf))
            if len(cached) == len(df_with_errors):
                print(f"[Transform cache] Reusing exact transformed table for seed {seed} "
                      f"({len(cached)} rows) — no worker LLM call needed.")
                return cached.reindex(columns=df_with_errors.columns)
            print(f"[Transform cache] Cached table row count mismatch ({len(cached)} vs {len(df_with_errors)}); "
                  f"falling back to raw-file reconstruction.")
        if not glob.glob(f"{transform_prefix}_raw_transform_batch_*_0.txt"):
            return None
        try:
            cached = reconstruct_transformed_df_from_files(transform_prefix)
        except Exception as e:
            print(f"[Transform cache] Unusable ({e}); regenerating.")
            return None
        if len(cached) != len(df_with_errors):
            print(f"[Transform cache] Row count mismatch ({len(cached)} vs {len(df_with_errors)}); regenerating.")
            return None
        print(f"[Transform cache] Reusing worker output for seed {seed} ({len(cached)} rows) — no worker LLM call needed.")
        return cached.reindex(columns=df_with_errors.columns)

    cached_transform = _try_reuse_cached_transform() if worker_mode == "llm" else None
    if resume_from_files:
        df_transformed = reconstruct_transformed_df_from_files(outfile_prefix)
        df_transformed = df_transformed.reindex(columns=df_with_errors.columns)
    elif cached_transform is not None:
        df_transformed = cached_transform
    elif worker_mode == "oracle":
        # Deterministic reference worker: useful for measuring checker FP rate
        # (all rows are clean by construction). No GPU required.
        df_transformed = pd.DataFrame(
            [oracle_transform_row(df_with_errors.iloc[i], business_rules) for i in range(len(df_with_errors))]
        )
    elif worker_mode == "passthrough":
        # No transformation: every injected rule violation remains in the data,
        # so this directly measures checker DETECTION of known faults. No GPU required.
        df_transformed = df_with_errors.copy(deep=True)
    else:  # "llm"
        worker_client = get_llm_client(hf_token, worker_model, quantization)
        df_transformed = llm_transform_stage(worker_client, df_with_errors, transform_prefix, business_rules,
                                             batch_size=batch_size, max_new_tokens=max_new_tokens)
        # Persist the exact transformed table so cache reuse across modes is
        # byte-exact. JSON (not CSV) is used deliberately: CSV round-trips
        # coerce whitespace-padded numerics back to numbers, silently erasing
        # injected faults and changing ground truth between modes.
        df_transformed.to_json(f"{transform_prefix}_transformed.json", orient="records")
        # client stays cached for reuse across seeds/modes

    # ---- Ground truth: which transformed rows still violate the rules? ----
    truth: Dict[int, Tuple[bool, List[str]]] = {}
    for i in range(len(df_transformed)):
        truth[i] = row_has_residual_error(df_transformed.iloc[i], df_with_errors.iloc[i], business_rules)
    n_bad = sum(1 for b, _ in truth.values() if b)
    print(f"[Ground truth] {n_bad}/{len(df_transformed)} transformed rows contain residual rule violations.")

    # ---- Stage: Hybrid / ablation checker ----
    print(f"\n--- [Stage] Validation (mode={mode}) ---")
    checker_client = None
    if mode in ("hybrid", "cross-only", "circular", "reflexion", "hybrid-circular"):
        model_for_checker = worker_model if mode in ("circular", "reflexion", "hybrid-circular") else checker_model
        checker_client = get_llm_client(hf_token, model_for_checker, quantization)

    rng = np.random.default_rng(seed)
    sample_idxs = sorted(int(i) for i in rng.choice(
        len(df_transformed), size=min(heldout_size, len(df_transformed)), replace=False
    ))
    verdicts = run_checker(mode, checker_client, df_with_errors, df_transformed, sample_idxs, run_prefix, business_rules)

    # ---- Score verdicts against ground truth ----
    metrics = CheckerMetrics()
    verdict_by_idx = {v.get("row_idx"): v for v in verdicts}
    scored_rows = []
    for idx in sample_idxs:
        v = verdict_by_idx.get(idx, {"verdict": "OK", "notes": "NO VERDICT RETURNED (parse failure or row skipped)", "checker": "missing"})
        not_ok = v.get("verdict") == "NOT_OK"
        truth_bad, reasons = truth[idx]
        metrics.record(not_ok, truth_bad, v.get("checker", "none"))
        scored_rows.append({
            "row_idx": idx, "verdict": v.get("verdict"), "checker_path": v.get("checker"),
            "checker_notes": v.get("notes"), "ground_truth_bad": truth_bad, "ground_truth_reasons": reasons,
        })

    final_report = {
        "config": {
            "mode": mode,
            "worker_mode": worker_mode,
            "data_csv": data_csv or "UCI Adult (default)",
            "worker_model": worker_model if worker_mode == "llm" else None,
            "checker_model": (worker_model if mode in ("circular", "reflexion", "hybrid-circular") else checker_model)
            if mode != "rules-only" else None,
            "seed": seed,
            "sample_size": sample_size,
            "heldout_size": len(sample_idxs),
            "resumed_from_files": resume_from_files,
        },
        "ground_truth": {
            "injected_faults": len(error_log),
            "residual_bad_rows_full_sample": n_bad,
            "residual_bad_rows_in_heldout": sum(1 for i in sample_idxs if truth[i][0]),
        },
        "verdict_coverage": {
            "rows_sampled": len(sample_idxs),
            "rows_with_verdict": sum(1 for i in sample_idxs if i in verdict_by_idx),
            "rows_missing_verdict": sum(1 for i in sample_idxs if i not in verdict_by_idx),
        },
        "checker_metrics": metrics.to_dict(),
        "row_level_results": scored_rows,
    }
    report_path = f"{run_prefix}_final_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Done] Report saved to {report_path}")
    print(json.dumps({k: final_report[k] for k in ("config", "ground_truth", "checker_metrics")}, indent=2))
    return final_report


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Safeguarded LLM-in-the-loop ETL evaluation harness.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face API token (default: HF_TOKEN env var). Never hardcode this.")
    parser.add_argument("--worker-model", type=str, default=DEFAULT_WORKER)
    parser.add_argument("--checker-model", type=str, default=DEFAULT_CHECKER)
    parser.add_argument("--mode", type=str, choices=MODES, default="hybrid",
                        help="Ablation mode: circular | cross-only | rules-only | hybrid")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds for a multi-seed run (overrides --seed).")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--heldout-size", type=int, default=DEFAULT_HELDOUT_SIZE)
    parser.add_argument("--outfile-prefix", type=str, default="results_safeguarded")
    parser.add_argument("--data-csv", type=str, default=None,
                        help="Path to a custom tabular CSV. Default: UCI Adult (downloaded).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Rows per worker transform call. Use smaller values for wide datasets "
                             "so the JSON output fits --max-new-tokens (e.g. 8 for KramaBench).")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Generation cap for the worker transform (default 2048; lower = faster).")
    parser.add_argument("--quantization", type=str, choices=("auto", "4bit", "8bit", "none"), default="auto",
                        help="Model quantization. auto = 4-bit on GPUs under 24 GB (Kaggle/Colab T4) "
                             "when bitsandbytes is installed, fp16 otherwise.")
    parser.add_argument("--worker-mode", type=str, choices=("llm", "oracle", "passthrough"), default="llm",
                        help="llm: call the Worker LLM (GPU). oracle: deterministic reference worker "
                             "(no GPU; measures checker false positives). passthrough: no transformation "
                             "(no GPU; measures checker detection of injected faults).")
    parser.add_argument("--rules-json", type=str, default=None,
                        help="JSON file mapping rule names to natural-language descriptions, "
                             "overriding the UCI-specific defaults for custom datasets.")
    parser.add_argument("--resume-from-files", action="store_true",
                        help="Reconstruct the transformed data from saved raw Worker outputs "
                             "instead of calling the Worker LLM (no GPU needed for the transform stage).")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.outfile_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checker_needs_llm = args.mode in ("hybrid", "cross-only", "circular")
    worker_needs_llm = args.worker_mode == "llm" and not args.resume_from_files
    if (checker_needs_llm or worker_needs_llm) and not args.hf_token:
        raise SystemExit("HF token required for LLM worker/checker. Set the HF_TOKEN environment variable, "
                         "or use --worker-mode oracle/passthrough with --mode rules-only for a fully "
                         "deterministic, GPU-free run.")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    start_time = time.time()
    for seed in seeds:
        run_pipeline(
            hf_token=args.hf_token,
            worker_model=args.worker_model,
            checker_model=args.checker_model,
            outfile_prefix=args.outfile_prefix,
            mode=args.mode,
            seed=seed,
            sample_size=args.sample_size,
            heldout_size=args.heldout_size,
            resume_from_files=args.resume_from_files,
            data_csv=args.data_csv,
            business_rules=load_business_rules(args.rules_json),
            worker_mode=args.worker_mode,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
