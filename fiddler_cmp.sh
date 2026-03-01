#!/usr/bin/env bash
# 运行 Fiddler 的 cpu_offload=0 与 1 对比，并生成 profile trace 与性能对比文件。
# 所有 profiling 通过 latency.py 完成。
# 用法: bash fiddler_cmp.sh [latency.py 的其他参数...]
# 例:   bash fiddler_cmp.sh
#       bash fiddler_cmp.sh --output-dir benchmarks/vs_output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT/benchmarks"
exec python latency.py --compare-cpu-offload --profile --output-dir "${REPO_ROOT}/benchmarks/vs_output" "$@"
