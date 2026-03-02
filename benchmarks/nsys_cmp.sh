#!/usr/bin/env bash
# nsys 采集 overlap=0 与 overlap=1 时间线，对比不并行 vs 并行。用法: bash nsys_compare_overlap.sh [latency.py 参数...]
# 输出: benchmarks/nsys_overlap_reports/ 下 .nsys-rep 与 overlap_performance_comparison.md

set -euo pipefail
# 1. 创建配置文件，允许非 root 用户进行 GPU Profiling
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-profiler.conf

# 2. 解除 Linux 内核的 perf 限制
echo "kernel.perf_event_paranoid = -1" | sudo tee /etc/sysctl.d/nsys-perf.conf

mkdir -p /home/yyj/fiddler/benchmarks/nsys_overlap_reports

sudo sysctl -w kernel.perf_event_paranoid=1

cd /home/yyj/fiddler/benchmarks
sudo nsys profile \
  --output="/home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_compare_overlap" \
  --trace=cuda,nvtx\
  --sample=cpu \
  --gpu-metrics-device=none \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --stats=true \
  python latency.py --compare-overlap --output-dir "/home/yyj/fiddler/benchmarks/nsys_overlap_reports"

echo "完成: /home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_compare_overlap.nsys-rep 与 overlap_performance_comparison.md"
