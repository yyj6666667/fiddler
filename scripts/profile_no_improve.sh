#!/usr/bin/env bash
# No improve 时的 Profiler：yyj_improve_loop=0、yyj_improve_cost=0，cpu_offload=1，生成 Chrome trace。
# 用法: bash scripts/profile_no_improve.sh [latency.py 的其他参数...]
# 例:   bash scripts/profile_no_improve.sh
#       bash scripts/profile_no_improve.sh --output-dir benchmarks/vs_output_no_improve
#       bash scripts/profile_no_improve.sh --input-token 64 --output-token 64
# 生成的 trace 可用 chrome://tracing 打开（默认文件名含 _no_improve）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/benchmarks/vs_output_no_improve}"

cd "$REPO_ROOT/benchmarks"
exec python latency.py \
  --profile \
  --cpu-offload 1 \
  --yyj-improve-loop 0 \
  --yyj-improve-cost 0 \
  --output-dir "$OUTPUT_DIR" \
  "$@"
