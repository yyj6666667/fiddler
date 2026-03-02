#!/usr/bin/env bash
# nsys 采集 overlap=0 与 overlap=1 时间线，对比不并行 vs 并行。用法: bash nsys_compare_overlap.sh [latency.py 参数...]
# 输出: benchmarks/nsys_overlap_reports/ 下 .nsys-rep 与 overlap_performance_comparison.md

set -euo pipefail


mkdir -p /home/yyj/fiddler/benchmarks/nsys_overlap_reports

sudo sysctl -w kernel.perf_event_paranoid=1
sudo -i

cd /home/yyj/fiddler/benchmarks
nsys profile \
  --output="/home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_compare_overlap" \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --gpu-metrics-device=all \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --stats=true \
  python latency.py --compare-overlap --output-dir "/home/yyj/fiddler/benchmarks/nsys_overlap_reports" 

echo "完成: /home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_compare_overlap.nsys-rep 与 overlap_performance_comparison.md"
