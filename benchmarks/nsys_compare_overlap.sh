#!/usr/bin/env bash
# nsys 采集 overlap=0 与 overlap=1 时间线，对比不并行 vs 并行。用法: bash nsys_compare_overlap.sh [latency.py 参数...]
# 输出: benchmarks/nsys_overlap_reports/ 下 .nsys-rep 与 overlap_performance_comparison.md

set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${BENCHMARKS_DIR}/nsys_overlap_reports"

mkdir -p "$OUTPUT_DIR"
cd "$BENCHMARKS_DIR"

nsys profile \
  --output="${OUTPUT_DIR}/nsys_compare_overlap" \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --stats=true \
  python latency.py --compare-overlap --output-dir "${OUTPUT_DIR}" "$@"

echo "完成: ${OUTPUT_DIR}/nsys_compare_overlap.nsys-rep 与 overlap_performance_comparison.md"
