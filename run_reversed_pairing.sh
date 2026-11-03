#!/usr/bin/env bash
# Reversed worker/checker pairing (paper, Section 8 "Internal validity: family
# versus model" and roadmap item 2). Separates the family relationship from
# model-specific checker behavior by running the same ablation cells with the
# roles swapped: Mistral-7B-Instruct-v0.2 as Worker, Llama-3.1-8B-Instruct as
# Checker, plus the Mistral self-pairing.
#
# Sample sizes, worker batch sizes and seeds match the reported ablation exactly
# (UCI: 100-row samples, 30 checked rows/seed, 10 rows/worker call; KramaBench:
# 100 rows, 40 checked, 8 rows/worker call, as in the archived runs/*.log), so the resulting *_final_report.json files are directly comparable to runs/.
#
# Requires: a 16 GB GPU, HF_TOKEN with access to both gated models.
# The worker transform runs once per (dataset, seed); every later mode reuses it.
#
#   ./run_reversed_pairing.sh [SEEDS] [OUTDIR]        (defaults: 1,2,3  runs)
#
# Output prefixes:  <OUT>/rev_uci_<mode>_seed<k>_*   and  <OUT>/rev_kb_<mode>_seed<k>_*
#   cross-only / hybrid : Mistral worker, Llama checker   (reversed cross pairing)
#   circular / reflexion / hybrid-circular : Mistral checks Mistral (self-pairing)
# rules-only needs no checker model, but it scores the Mistral worker's transforms,
# so it is re-run here instead of being reused from the original Llama-worker runs.
set -e
SEEDS="${1:-1,2,3}"
OUT="${2:-runs}"
mkdir -p "$OUT"

WORKER="mistralai/Mistral-7B-Instruct-v0.2"
CHECKER="meta-llama/Llama-3.1-8B-Instruct"

echo "=== UCI Adult, reversed pairing (seeds: $SEEDS) ==="
for MODE in circular reflexion rules-only cross-only hybrid hybrid-circular; do
  python safeguarded_LLM_in_the_loop.py --mode "$MODE" --seeds "$SEEDS" \
    --worker-model "$WORKER" --checker-model "$CHECKER" \
    --sample-size 100 --heldout-size 30 --batch-size 10 \
    --outfile-prefix "$OUT/rev_uci"
done

echo "=== KramaBench water-quality, reversed pairing ==="
for MODE in circular reflexion rules-only cross-only hybrid hybrid-circular; do
  python safeguarded_LLM_in_the_loop.py --mode "$MODE" --seeds "$SEEDS" \
    --worker-model "$WORKER" --checker-model "$CHECKER" \
    --data-csv datasets/water-body-testing-2019.csv \
    --rules-json datasets/kramabench_water_rules.json \
    --sample-size 100 --heldout-size 40 --batch-size 8 \
    --outfile-prefix "$OUT/rev_kb"
done

echo "Done. Compare rev_* reports against the original runs/ reports:"
echo "  if the 'same-family over-flags / cross is conservative' pattern follows the"
echo "  CHECKER model (Llama still over-flags as cross checker; Mistral is still"
echo "  conservative when checking itself), the family interpretation is ruled out."
