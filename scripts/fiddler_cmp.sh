#!/usr/bin/env bash
# 仅对比 cpu_offload=1 下 不并行 vs 并行，并生成 profile trace 与性能对比文件。
# 所有 profiling 通过 latency.py 完成。
# 用法: bash fiddler_cmp.sh [latency.py 的其他参数...]
# 例:   bash fiddler_cmp.sh
#       bash fiddler_cmp.sh --output-dir benchmarks/vs_output

set -euo pipefail


cd "/home/yyj/fiddler/benchmarks"
exec python latency.py  --profile --output-dir "/home/yyj/fiddler/benchmarks/vs_output" "$@"
