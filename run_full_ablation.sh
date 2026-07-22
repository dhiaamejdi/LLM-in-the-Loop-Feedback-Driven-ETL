#!/usr/bin/env bash
# Full evaluation matrix for the resubmission. Requires: GPU + HF_TOKEN with
# access to Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2.
#
#   ./run_full_ablation.sh [SEEDS] [OUTDIR]
#
# Defaults: SEEDS="1,2,3,4,5" OUTDIR="runs"
# Models are cached in-process, so each loads once per mode.
set -e
SEEDS="${1:-1,2,3,4,5}"
OUT="${2:-runs}"
mkdir -p "$OUT"

echo "=== UCI Adult: 2x2 ablation + Reflexion baseline (seeds: $SEEDS) ==="
for MODE in rules-only circular reflexion cross-only hybrid; do
  python safeguarded_LLM_in_the_loop.py --mode "$MODE" --seeds "$SEEDS" \
    --outfile-prefix "$OUT/uci"
done

echo "=== KramaBench water-quality: same matrix ==="
for MODE in rules-only circular reflexion cross-only hybrid; do
  python safeguarded_LLM_in_the_loop.py --mode "$MODE" --seeds "$SEEDS" \
    --data-csv datasets/water-body-testing-2019.csv \
    --rules-json datasets/kramabench_water_rules.json \
    --sample-size 1000 --heldout-size 100 \
    --outfile-prefix "$OUT/kb"
done

echo "=== Aggregating -> LaTeX table ==="
python aggregate_results.py "$OUT"
