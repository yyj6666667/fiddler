#!/usr/bin/env bash
set -e
cd /home/yyj/fiddler/benchmarks
FIDDLER_MODEL="${1:-mistralai/Mixtral-8x7B-v0.1}"
echo "=== Fiddler baseline (latency benchmark，带 PyTorch Profiler) ==="
(
  python latency.py \
    --model "${FIDDLER_MODEL}" \
    --cpu-offload 1 \
    --batch_size 1 \
    --beam_width 1 \
    --profile \
    --output-dir vs_output
)