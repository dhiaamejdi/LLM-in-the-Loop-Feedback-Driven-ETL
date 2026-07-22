#!/usr/bin/env bash
# Fetch KramaBench (Lai et al., 2025) — reviewer-recommended benchmark of real
# data-lake files. https://github.com/mitdbg/KramaBench
set -e
curl -sL -o kramabench.zip https://github.com/mitdbg/KramaBench/archive/refs/heads/main.zip
unzip -q kramabench.zip
echo "Data lakes under Kramabench-main/data/{archeology,astronomy,biomedical,environment,legal,wildfire}"
