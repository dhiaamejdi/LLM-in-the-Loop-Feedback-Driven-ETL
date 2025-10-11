import os; os.environ["HF_TOKEN"] ="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"   
hf_token = os.environ.get("HF_TOKEN")
from huggingface_hub import HfApi

# Use the secret you created
import os
hf_token = os.environ.get("HF_TOKEN")

# Authenticate
if hf_token:
    api = HfApi()
    api.whoami(token=hf_token)
    print("Successfully logged into Hugging Face.")
else:
    print("Hugging Face token not found. Please add it as a Kaggle Secret with the label 'HF_TOKEN'.")

"""
Enhanced LLM-driven ETL Pipeline with Architectural Safeguards
- Cross-Architecture Validation (Llama worker + Mistral checker)
- Deterministic Rule Hybridization
- Stochasticity Modeling
- Enhanced Human-in-the-Loop
"""
# safeguarded_pipeline.py
"""
LLM-driven ETL Full Pipeline with Architectural Safeguards

This script implements the "safeguarded" architecture described in the research paper.
Key features include:
1.  Cross-Architecture Validation: Uses a different LLM for checking (e.g., Mistral)
    than for the primary work (e.g., Llama 3), preventing the model from being
    blind to its own family-specific errors.
2.  Deterministic Rule Hybridization: Implements a hybrid checker that first
    applies fast, reliable, deterministic rules for simple validation tasks.
    Only the rows that pass these basic checks are sent to the more expensive
    and stochastic LLM checker for complex semantic validation.
3.  Robust Data Handling: Includes a utility to correctly serialize Pandas/NumPy
    data types into JSON, preventing common runtime errors.
"""

import os
import argparse
import json
import random
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Quiet transformers messages & deprecation warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional HF imports for the LLM
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
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
    "hours-per-week", "native-country", "income"
]
SAMPLE_SIZE = 100
SEED = 42
BATCH_SIZE = 20
CHECKER_BATCH_SIZE = 5

BUSINESS_RULES = {
    "trim_whitespace": "Trim whitespace in all string columns",
    "income_map": "Normalize income labels to '>50K' and '<=50K' (e.g. '>50K.' becomes '>50K')",
    "sex_map": "Map sex to 'Male'/'Female' (capitalize)",
}

# -------------------------
# Data classes for metrics
# -------------------------
@dataclass
class StageMetrics:
    name: str
    samples_checked: int = 0
    success_count: int = 0
    failure_count: int = 0
    notes: List[str] = field(default_factory=list)

    def record_success(self, n=1):
        self.success_count += n
        self.samples_checked += n

    def record_failure(self, n=1, note=None):
        self.failure_count += n
        self.samples_checked += n
        if note:
            self.notes.append(note)

    def to_dict(self):
        return {
            "stage": self.name,
            "samples_checked": int(self.samples_checked),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "success_rate": float(self.success_count / self.samples_checked) if self.samples_checked else 0.0,
            "notes": self.notes
        }

# -------------------------
# Utility functions
# -------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def load_uci_adult():
    train_df = pd.read_csv(UCI_TRAIN, names=COLUMNS, sep=r",\s", engine="python", na_values="?", header=None)
    test_df = pd.read_csv(UCI_TEST, names=COLUMNS, sep=r",\s", engine="python", skiprows=1, na_values="?", header=None)
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    return df

def inject_errors(df: pd.DataFrame, error_config: Dict[str, float], seed=SEED) -> pd.DataFrame:
    df = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    nrows, ncols = df.shape
    total_cells = nrows * ncols
    m_frac = error_config.get("missing_value", 0.03)
    m_count = int(total_cells * m_frac)
    for _ in range(m_count):
        r = rng.integers(0, nrows)
        c = rng.integers(0, ncols)
        df.iat[r, c] = np.nan
    s_frac = error_config.get("swap_columns", 0.02)
    s_rows = int(nrows * s_frac)
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for _ in range(s_rows):
        if len(cat_cols) >= 2:
            r = rng.integers(0, nrows)
            c1, c2 = rng.choice(cat_cols, size=2, replace=False)
            df.at[r, c1], df.at[r, c2] = df.at[r, c2], df.at[r, c1]
    bc_frac = error_config.get("bad_casing", 0.03)
    ws_frac = error_config.get("stray_whitespace", 0.03)
    for c in df.columns:
        if df[c].dtype == object:
            str_mask = df[c].notna()
            bc_indices = df[str_mask].sample(frac=bc_frac, random_state=seed).index
            df.loc[bc_indices, c] = df.loc[bc_indices, c].str.lower()
            ws_indices = df[str_mask].sample(frac=ws_frac, random_state=seed).index
            df.loc[ws_indices, c] = " " + df.loc[ws_indices, c].astype(str) + " "
    return df

def make_serializable(data_dict: Dict) -> Dict:
    """Converts numpy types in a dictionary to native Python types for JSON serialization."""
    sanitized = {}
    for key, value in data_dict.items():
        if pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.int64)):      # <-- CORRECTED LINE 1
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):    # <-- CORRECTED LINE 2
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized

def extract_json_block(text: str):
    """Finds and parses the first valid JSON object or array in a string."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    first_char_pos = -1
    for i, char in enumerate(text):
        if char in ['{', '[']:
            first_char_pos = i
            break
    if first_char_pos == -1:
        raise ValueError("no JSON block found (no starting '{' or '[')")
    search_text = text[first_char_pos:]
    try:
        return json.loads(search_text)
    except json.JSONDecodeError:
        pass
    start_char = search_text[0]
    end_char = '}' if start_char == '{' else ']'
    depth = 0
    for i, char in enumerate(search_text):
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                block = search_text[:i+1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    pass
    raise ValueError("no valid, balanced JSON block found")

# -------------------------
# LLM Client
# -------------------------
class LLMClient:
    def __init__(self, hf_token: str, model_name: str):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not available. Please install them.")
        if hf_token is None:
            raise RuntimeError("HF token required for LLM mode.")

        print(f"[LLM Client] Loading tokenizer and model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token
        )
        self.model.eval()

    def call(self, prompt: str, max_length: int = 1024, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]

        if "mistral" in self.model_name.lower():
             full_prompt = f"<s>[INST] {prompt} [/INST]"
             input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.model.device)
        else: # Default to Llama-3 style
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if "Llama-3" in self.model_name else self.tokenizer.eos_token_id
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, max_new_tokens=max_length, eos_token_id=terminators,
                do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

# -------------------------
# LLM-driven Stage Helpers
# -------------------------
def llm_transform_stage(client: LLMClient, df_in: pd.DataFrame, outfile_prefix: str, business_rules: Dict[str,str], batch_size:int=BATCH_SIZE, retries:int=2):
    rows_out = []
    n = len(df_in)
    for start in range(0, n, batch_size):
        batch = df_in.iloc[start:start+batch_size]
        print(f"[Worker LLM: {client.model_name}] Transforming batch rows {start}..{start+len(batch)-1}")
        prompt = (
            "Transform rows according to the business rules and return a JSON object like {'rows':[ {...}, {...} ]}.\n"
            "Business rules:\n" + "\n".join([f"- {k}: {v}" for k,v in business_rules.items()]) + "\n\n"
            "Context CSV:\n" + batch.to_csv(index=False) + "\n\n"
            "IMPORTANT: You MUST return only the JSON object. Do not include any introductory text or explanations. Your entire response must be the JSON itself, starting with '{'."
        )
        parsed_rows = None
        for attempt in range(retries):
            out = client.call(prompt, max_length=4096, temperature=0.0)
            with open(f"{outfile_prefix}_raw_transform_batch_{start}_{attempt}.txt", "w", encoding="utf-8") as f: f.write(out)
            try:
                parsed = extract_json_block(out)
                parsed_rows = parsed.get("rows") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
                if parsed_rows is not None: break
            except Exception as e:
                print(f"Transform batch {start} attempt {attempt+1} failed to parse: {e}")
                parsed_rows = None
        if parsed_rows is None: parsed_rows = []
        for i in range(len(batch)):
            rows_out.append(parsed_rows[i] if i < len(parsed_rows) and isinstance(parsed_rows[i], dict) else batch.iloc[i].to_dict())
    df_transformed = pd.DataFrame(rows_out)
    return df_transformed

# ----------------------------------------------------
# Safeguarded Checker Implementations
# ----------------------------------------------------
def deterministic_check_row(row: pd.Series) -> Tuple[bool, str]:
    """Safeguard 2: Deterministic Rule Hybridization."""
    for col, val in row.items():
        if isinstance(val, str) and (val.startswith(' ') or val.endswith(' ')):
            return False, f"DETERMINISTIC_FAIL: Whitespace not trimmed in column '{col}'"
    if 'income' in row and isinstance(row['income'], str) and row['income'] not in ['>50K', '<=50K', 'Unknown']:
        return False, f"DETERMINISTIC_FAIL: Income '{row['income']}' not normalized"
    return True, "DETERMINISTIC_OK"

def hybrid_semantic_checker(
    checker_client: LLMClient,
    original_rows: pd.DataFrame,
    transformed_rows: pd.DataFrame,
    sample_idxs: List[int],
    outfile_prefix: str
) -> List[Dict]:
    """
    Implements Safeguards 1 & 2 with batching, a robust one-shot prompt,
    and a cleaning step to remove rogue backslashes from the model's output.
    """
    # This print confirms you are running the latest version of the function
    print("\n--- [CONFIRMATION] EXECUTING FINAL BATCHED CHECKER WITH CLEANUP ---\n")
    
    all_verdicts = []
    deterministic_fails = []
    rows_for_llm_check = []

    print(f"\n[Hybrid Checker] Running deterministic checks on {len(sample_idxs)} samples...")
    for idx in sample_idxs:
        is_ok, reason = deterministic_check_row(transformed_rows.iloc[idx])
        if not is_ok:
            verdict = {"row_idx": int(idx), "verdict": "NOT_OK", "notes": reason, "checker": "deterministic"}
            all_verdicts.append(verdict)
            deterministic_fails.append(verdict)
        else:
            rows_for_llm_check.append(idx)

    print(f"[Hybrid Checker] {len(deterministic_fails)} rows failed deterministic checks.")
    
    if not rows_for_llm_check:
        return all_verdicts

    print(f"[Hybrid Checker] Sending {len(rows_for_llm_check)} rows to LLM for semantic validation in batches of {CHECKER_BATCH_SIZE}.")
    
    for start in range(0, len(rows_for_llm_check), CHECKER_BATCH_SIZE):
        batch_idxs = rows_for_llm_check[start : start + CHECKER_BATCH_SIZE]
        
        cases = [
            {
                "row_idx": int(idx),
                "original": make_serializable(original_rows.iloc[idx].to_dict()),
                "transformed": make_serializable(transformed_rows.iloc[idx].to_dict())
            }
            for idx in batch_idxs
        ]
        
        # This is the full, robust prompt from before
        prompt = (
            "You are a silent, precise JSON-generating API. You will be given a rule and a JSON array of data cases. "
            "For each case, you will check if the 'transformed' data is a valid transformation of the 'original' data according to the rule. "
            "Your entire response MUST be a single JSON array, with one object per case. Do not include any text, explanations, or markdown before or after the JSON array.\n\n"
            "## RULE:\n"
            "The 'transformed' row should be a semantically correct version of the 'original' row, with whitespace trimmed and labels normalized.\n\n"
            "## EXAMPLE:\n"
            "### INPUT:\n"
            "[\n"
            "  {\n"
            '    "row_idx": 999,\n'
            '    "original": {"name": " john ", "income": "<=50K."},\n'
            '    "transformed": {"name": "john", "income": "<=50K"}\n'
            "  }\n"
            "]\n"
            "### CORRECT OUTPUT:\n"
            "[\n"
            "  {\n"
            '    "row_idx": 999,\n'
            '    "verdict": "OK",\n'
            '    "notes": "Whitespace was trimmed and income label was normalized correctly."\n'
            "  }\n"
            "]\n\n"
            "## ACTUAL TASK:\n"
            "### INPUT:\n"
            f"{json.dumps(cases, indent=2)}\n"
            "### CORRECT OUTPUT:"
        )
        
        print(f"[Checker LLM: {checker_client.model_name}] Validating batch of {len(cases)} rows (indices {batch_idxs[0]}..{batch_idxs[-1]})...")
        
        out = checker_client.call(prompt, max_length=8192, temperature=0.0)
        
        # --- THIS IS THE CRITICAL FIX ---
        # The model incorrectly adds a backslash (\) before underscores (_). This line removes it.
        out = out.replace(r'\_', '_')
        # --------------------------------
        
        with open(f"{outfile_prefix}_raw_hybrid_semantic_checks_batch_{start}.txt", "w", encoding="utf-8") as f:
            f.write(out)

        try:
            llm_verdicts_batch = extract_json_block(out)
            if isinstance(llm_verdicts_batch, list):
                for v in llm_verdicts_batch:
                    v['checker'] = 'llm'
                all_verdicts.extend(llm_verdicts_batch)
            else:
                print(f"    - WARNING: LLM output for batch {start} was not a list. Skipping.")
        except Exception as e:
            print(f"    - WARNING: Hybrid checker failed to parse LLM output for batch {start}. Error: {e}")
            
    return all_verdicts


# -------------------------
# Orchestration
# -------------------------
def run_pipeline(hf_token: str, worker_model: str, checker_model: str, outfile_prefix: str, resume_from_files: bool): # <-- Added resume argument
    seed_everything(SEED)
    
    # This part MUST still run to get the original data for comparison
    print("[Pipeline] Loading and preparing dataset...")
    df_sample = load_uci_adult().sample(SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    df_with_errors = inject_errors(df_sample.copy(), {"missing_value":0.03, "swap_columns":0.02, "bad_casing":0.03, "stray_whitespace":0.03})

    print("\n--- [Stage 2] Semantic Transformation ---")

    # --- MODIFICATION START ---
    if resume_from_files:
        print("resummeeed")
        # If resuming, load from files instead of calling the LLM
        df_transformed = reconstruct_transformed_df_from_files(outfile_prefix)
        # Ensure column order matches the original for consistency
        df_transformed = df_transformed.reindex(columns=df_with_errors.columns)
    else:
        # If not resuming, run the original expensive function
        print("--- Initializing LLM Clients for Safeguarded Architecture ---")
        worker_client = LLMClient(hf_token=hf_token, model_name=worker_model)
        df_transformed = llm_transform_stage(worker_client, df_with_errors, outfile_prefix, BUSINESS_RULES, BATCH_SIZE)
    # --- MODIFICATION END ---
    
    # The rest of the pipeline continues from here
    print("\n--- [Stage 3] Hybrid Checker (Deterministic + Checker) ---")
    print("--- Initializing LLM Clients for Safeguarded Architecture ---")
    checker_client = LLMClient(hf_token=hf_token, model_name=checker_model)

    sample_idxs = list(np.random.choice(len(df_transformed), size=min(30, len(df_transformed)), replace=False))
    transform_verdicts = hybrid_semantic_checker(checker_client, df_with_errors, df_transformed, sample_idxs, outfile_prefix)
    
    transform_checker_metrics = StageMetrics("Transformation Hybrid Checker")
    transform_checker_metrics.samples_checked = len(sample_idxs)
    transform_checker_metrics.success_count = sum(1 for v in transform_verdicts if v.get("verdict") == "OK")
    transform_checker_metrics.failure_count = transform_checker_metrics.samples_checked - transform_checker_metrics.success_count

    # Final Report
    final_report = {
        "config": {"worker_model": worker_model, "checker_model": checker_model, "sample_size": SAMPLE_SIZE, "resumed_from_files": resume_from_files},
        "results": {
            "semantic_transformation_hybrid_checker": transform_checker_metrics.to_dict()
        }
    }
    report_path = f"{outfile_prefix}_final_report.json"
    with open(report_path, "w") as f: json.dump(final_report, f, indent=4)
    print(f"\n✅ Safeguarded pipeline finished. Final report saved to {report_path}")
    print(json.dumps(final_report, indent=4))

import glob
import re

def reconstruct_transformed_df_from_files(outfile_prefix: str) -> pd.DataFrame:
    """
    Finds all raw transform output files, parses them, and reconstructs the transformed DataFrame.
    This allows resuming the pipeline without re-running the expensive LLM transform stage.
    """
    print(f"\n[Pipeline Resumed] Reconstructing transformed DataFrame from files with prefix: {outfile_prefix}")
    
    # Find all the output files from the first attempt (attempt 0)
    file_pattern = f"{outfile_prefix}_raw_transform_batch_*_0.txt"
    transform_files = glob.glob(file_pattern)

    if not transform_files:
        raise FileNotFoundError(f"No transform files found matching the pattern: {file_pattern}. Cannot resume.")

    # Sort files numerically by batch number to ensure correct order
    def get_batch_num(filename):
        match = re.search(r'_batch_(\d+)_', filename)
        return int(match.group(1)) if match else -1
        
    transform_files.sort(key=get_batch_num)
    
    all_rows = []
    for file_path in transform_files:
        print(f"  - Reading and parsing {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        try:
            parsed = extract_json_block(content)
            # The output can be a dict {'rows': [...]} or just a list [...]
            parsed_rows = parsed.get("rows") if isinstance(parsed, dict) else parsed
            if isinstance(parsed_rows, list):
                all_rows.extend(parsed_rows)
            else:
                print(f"    - WARNING: Could not find a list of rows in {file_path}. Skipping.")
        except Exception as e:
            print(f"    - WARNING: Failed to parse JSON from {file_path}. Error: {e}. Skipping.")
            
    if not all_rows:
        raise ValueError("Could not reconstruct any rows from the saved files. Please check the file contents.")
        
    print(f"[Pipeline Resumed] Successfully reconstructed {len(all_rows)} rows.")
    return pd.DataFrame(all_rows)


# -------------------------
# Main Execution Block
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full LLM-driven ETL pipeline with architectural safeguards.")
    
    hf_token = os.environ.get("HF_TOKEN")

    parser.add_argument("--hf-token", type=str, default=hf_token, help="Hugging Face API token.")
    parser.add_argument("--worker-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model for worker tasks.")
    parser.add_argument("--checker-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Hugging Face model for checker tasks.")
    parser.add_argument("--outfile-prefix", type=str, default="results_safeguarded", help="Prefix for all output files.")
    # --- NEW ARGUMENT ---
    parser.add_argument("--resume-from-files", action="store_true", help="Resume pipeline by loading transformed data from files instead of calling the worker LLM.")
    
    args, _ = parser.parse_known_args()
    
    output_dir = os.path.dirname(args.outfile_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set it via --hf-token or the HF_TOKEN environment variable.")

    start_time = time.time()
    # Corrected the original error here as well
    run_pipeline(
        hf_token=args.hf_token,
        worker_model=args.worker_model,
        checker_model=args.checker_model,
        outfile_prefix=args.outfile_prefix,
        resume_from_files= True # <-- Pass the new argument
    )
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
