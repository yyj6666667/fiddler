#!/usr/bin/env bash

set -euo pipefail

# 用法：
#   bash benchmarks/Vs.sh [MODEL_ID_OR_PATH]
# 默认 MODEL 为 mistralai/Mixtral-8x7B-v0.1（与 Fiddler 共用）
# 脚本会显式进入仓库根目录，再在 benchmarks/ 下运行，避免路径冲突。

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-mistralai/Mixtral-8x7B-v0.1}"

echo "=== Mixtral-offloading baseline (带 PyTorch Profiler) ==="
(
  cd benchmarks
  python eval-baseline.py \
    --framework mixtral-offloading \
    --model "${MODEL}" \
    --quantized False \
    --profile
)

echo
echo "=== Fiddler baseline (latency benchmark，带 PyTorch Profiler) ==="
(
  cd benchmarks
  python latency.py \
    --model "${MODEL}" \
    --cpu-offload 1 \
    --batch_size 1 \
    --beam_num 1 \
    --profile
)

echo
echo "=== 完成 ==="
echo "Profiler trace 文件："
echo "  - mixtral-offloading_profiler_trace.json"
echo "  - fiddler_profiler_trace.json"
