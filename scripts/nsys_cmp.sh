#!/usr/bin/env bash
# nsys 分别采集 overlap=0 与 overlap=1 两段时间线，生成两份 .nsys-rep（well named）。用法: bash nsys_cmp.sh
# 输出: benchmarks/nsys_overlap_reports/ 下 nsys_no_overlap.nsys-rep、nsys_overlap.nsys-rep 与 overlap_performance_comparison.md

set -euo pipefail
# 1. 创建配置文件，允许非 root 用户进行 GPU Profiling
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-profiler.conf

# 2. 解除 Linux 内核的 perf 限制
echo "kernel.perf_event_paranoid = -1" | sudo tee /etc/sysctl.d/nsys-perf.conf

mkdir -p /home/yyj/fiddler/benchmarks/nsys_overlap_reports

cd /home/yyj/fiddler/benchmarks

export HF_HOME="/home/yyj/.cache/huggingface"

# 第一段：不并行 (overlap=0)，cudaProfilerStart/Stop 只包这一段
sudo -E /usr/local/cuda/bin/nsys profile \
  --output="/home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_no_overlap" \
  --trace=cuda,nvtx \
  --sample=cpu \
  --gpu-metrics-device=none \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --stats=true \
  /opt/conda/bin/python latency.py --compare-overlap --nsys-overlap-run=0 --output-dir "/home/yyj/fiddler/benchmarks/nsys_overlap_reports"

# 第二段：并行 (overlap=1)，cudaProfilerStart/Stop 只包这一段
sudo -E /usr/local/cuda/bin/nsys profile \
  --output="/home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_overlap" \
  --trace=cuda,nvtx \
  --sample=cpu \
  --gpu-metrics-device=none \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --stats=true \
  /opt/conda/bin/python latency.py --compare-overlap --nsys-overlap-run=1 --output-dir "/home/yyj/fiddler/benchmarks/nsys_overlap_reports"


echo "完成: /home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_no_overlap.nsys-rep、/home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_overlap.nsys-rep 与 /home/yyj/fiddler/benchmarks/nsys_overlap_reports/overlap_performance_comparison.md"
