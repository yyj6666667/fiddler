#!/usr/bin/env bash

set -euo pipefail

# 用法：
#   bash benchmarks/Vs.sh [选项] [FIDDLER_MODEL_ID_OR_PATH] [OFFLOADING_MODEL_ID_OR_PATH]
#   bash benchmarks/Vs.sh --fiddler-only [FIDDLER_MODEL_ID_OR_PATH]   # 仅跑 Fiddler，不跑 mixtral-offloading
# 默认 FIDDLER_MODEL 为 mistralai/Mixtral-8x7B-v0.1
# 默认 OFFLOADING_MODEL 为 lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo（HQQ 量化，含 W_q）
# 脚本会显式进入仓库根目录，再在 benchmarks/ 下运行，避免路径冲突。
#
# 运行前需安装 mixtral-offloading 依赖（含 hqq）（仅在全量对比时需要）：
#   pip install -r benchmarks/mixtral_offloading/requirements.txt

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Vs.sh 生成的所有结果（profiler trace、latency.txt 等）统一放到此目录
VS_OUTPUT_DIR="benchmarks/vs_output"
mkdir -p "$VS_OUTPUT_DIR"

FIDDLER_ONLY=0
if [[ "${1:-}" == "--fiddler-only" || "${1:-}" == "-f" ]]; then
  FIDDLER_ONLY=1
  shift
fi

FIDDLER_MODEL="${1:-mistralai/Mixtral-8x7B-v0.1}"
OFFLOADING_MODEL="${2:-lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo}"

# 仅在全量对比时检查 mixtral-offloading 依赖
if [[ "$FIDDLER_ONLY" -eq 0 ]]; then
  if ! python -c "import hqq" 2>/dev/null; then
    echo "错误: 未安装 mixtral-offloading 依赖（缺少 hqq）。"
    echo "请先执行: pip install -r benchmarks/mixtral_offloading/requirements.txt"
    echo "或仅跑 Fiddler: bash benchmarks/Vs.sh --fiddler-only"
    exit 1
  fi

  echo "=== Mixtral-offloading baseline (带 PyTorch Profiler) ==="
  (
    cd benchmarks
    python eval-baseline.py \
      --framework mixtral-offloading \
      --model "${OFFLOADING_MODEL}" \
      --quantized true \
      --profile \
      --output-dir vs_output
  )
  echo
fi

echo "=== Fiddler baseline (latency benchmark，带 PyTorch Profiler) ==="
(
  cd benchmarks
  python latency.py \
    --model "${FIDDLER_MODEL}" \
    --cpu-offload 1 \
    --batch_size 1 \
    --beam_width 1 \
    --profile \
    --output-dir vs_output
)

echo
echo "=== 完成 ==="
echo "输出目录: ${VS_OUTPUT_DIR}/"
if [[ "$FIDDLER_ONLY" -eq 1 ]]; then
  echo "  - fiddler_profiler_trace_no_stack.json"
  echo "  - fiddler_memory_timeline.html"
  echo "  - latency.txt"
else
  echo "  - mixtral-offloading_profiler_trace_no_stack.json"
  echo "  - mixtral-offloading_memory_timeline.html"
  echo "  - fiddler_profiler_trace_no_stack.json"
  echo "  - fiddler_memory_timeline.html"
  echo "  - latency.txt"
fi
