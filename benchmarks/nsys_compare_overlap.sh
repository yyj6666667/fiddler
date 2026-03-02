#!/usr/bin/env bash
# 使用 NVIDIA Nsight Systems (nsys) 采集 overlap=0 与 overlap=1 的 GPU/CPU 时间线，
# 用于分析「不并行」与「并行」的执行差异。不修改 latency.py，不使用 --profile。
#
# 用法:
#   bash nsys_compare_overlap.sh [latency.py 的其他参数...]
# 例:
#   bash nsys_compare_overlap.sh
#   bash nsys_compare_overlap.sh --output-dir /tmp/nsys_out
#
# 依赖: 已安装 Nsight Systems (nsys 在 PATH 中)
# 输出: --output-dir 下生成 nsys_compare_overlap.nsys-rep（及 .nsys-rep 同名的导出文件），
#       以及 latency.py 写入的 overlap_performance_comparison.md。
# 查看: nsys-ui nsys_compare_overlap.nsys-rep 或 nsys export 导出为其他格式。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCHMARKS_DIR="${REPO_ROOT}/benchmarks"
# 默认输出目录（可与 latency.py 的 --output-dir 一致，便于一起查看）
OUTPUT_DIR="${1:-}"
if [[ -n "${OUTPUT_DIR:-}" && "${OUTPUT_DIR:-}" != --* ]]; then
  # 第一个参数是目录则当作 output-dir
  shift
else
  OUTPUT_DIR="${BENCHMARKS_DIR}/nsys_overlap_reports"
fi

mkdir -p "$OUTPUT_DIR"
cd "$BENCHMARKS_DIR"

# nsys: -t cuda,nvtx,osrt 采集 CUDA 与 OS 运行时；--force-overwrite 覆盖旧报告
# -c nvtx -p train_loop: 精确打击，仅当执行到代码里 NVTX 标记的 train_loop 区域时采集
#   若未在 Python 里打 NVTX 的 train_loop 标记，可去掉 -c nvtx 与 -p train_loop 改为全程采集
# latency.py: --compare-overlap 依次跑 overlap=0 与 overlap=1，且不传 --profile
echo "=== Nsight Systems 采集 overlap 对比 (不并行 -> 并行) ==="
echo "输出目录: $OUTPUT_DIR"
echo ""

nsys profile \
  -o "${OUTPUT_DIR}/nsys_compare_overlap" \
  -t cuda,nvtx,osrt \
  -c nvtx \
  -p train_loop \
  --force-overwrite=true \
  --stats=true \
  -- \
  python latency.py \
  --compare-overlap \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo ""
echo "=== 采集完成 ==="
echo "  - 时间线报告: ${OUTPUT_DIR}/nsys_compare_overlap.nsys-rep"
echo "  - 性能对比:   ${OUTPUT_DIR}/overlap_performance_comparison.md"
echo "  - 查看 nsys:  nsys-ui ${OUTPUT_DIR}/nsys_compare_overlap.nsys-rep"
echo "  - 时间线中前半段为 不并行(overlap=0)，后半段为 并行(overlap=1)。"
