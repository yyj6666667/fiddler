#!/usr/bin/env bash
# Profile cpu_offload=1，生成 profile trace（overlap 已移除，仅单路 offload）。
# 用法: bash fiddler_cmp.sh [latency.py 的其他参数...]
# 例:   bash fiddler_cmp.sh
#       bash fiddler_cmp.sh --output-dir benchmarks/vs_output

set -euo pipefail


cd "/home/yyj/fiddler/benchmarks"
exec python latency.py --profile --cpu-offload 1 --output-dir "/home/yyj/fiddler/benchmarks/vs_output" "$@"
