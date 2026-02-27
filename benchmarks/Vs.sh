#!/usr/bin/env bash

set -euo pipefail

# 用法：
#   bash benchmarks/Vs.sh [FIDDLER_MODEL_ID_OR_PATH] [OFFLOADING_MODEL_ID_OR_PATH]
# 默认 FIDDLER_MODEL 为 mistralai/Mixtral-8x7B-v0.1
# 默认 OFFLOADING_MODEL 为 lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo（HQQ 量化，含 W_q）
# 脚本会显式进入仓库根目录，再在 benchmarks/ 下运行，避免路径冲突。
#
# 运行前需安装 mixtral-offloading 依赖（含 hqq）：
#   pip install -r benchmarks/mixtral_offloading/requirements.txt

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FIDDLER_MODEL="${1:-mistralai/Mixtral-8x7B-v0.1}"
OFFLOADING_MODEL="${2:-lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo}"

# 检查 mixtral-offloading 依赖，缺则提示并退出
if ! python -c "import hqq" 2>/dev/null; then
  echo "错误: 未安装 mixtral-offloading 依赖（缺少 hqq）。"
  echo "请先执行: pip install -r benchmarks/mixtral_offloading/requirements.txt"
  exit 1
fi

echo "=== Mixtral-offloading baseline (带 PyTorch Profiler) ==="
(
  cd benchmarks
  python eval-baseline.py \
    --framework mixtral-offloading \
    --model "${OFFLOADING_MODEL}" \
    --quantized true \
    --profile
)

echo
echo "=== Fiddler baseline (latency benchmark，带 PyTorch Profiler) ==="
(
  cd benchmarks
  python latency.py \
    --model "${FIDDLER_MODEL}" \
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
