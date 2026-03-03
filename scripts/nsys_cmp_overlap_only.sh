#!/usr/bin/env bash
# nsys 仅采集 overlap=1（并行）时间线。用法: bash nsys_cmp_overlap_only.sh
# 输出: benchmarks/nsys_overlap_reports/ 下 nsys_overlap_only.nsys-rep

set -euo pipefail
# 1. 创建配置文件，允许非 root 用户进行 GPU Profiling
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-profiler.conf

# 2. 解除 Linux 内核的 perf 限制
echo "kernel.perf_event_paranoid = -1" | sudo tee /etc/sysctl.d/nsys-perf.conf

mkdir -p /home/yyj/fiddler/benchmarks/nsys_overlap_reports

cd /home/yyj/fiddler/benchmarks

# 在运行脚本前设置环境变量
export HF_HOME="/home/yyj/.cache/huggingface"

# 单跑 overlap 时 latency.py 不调用 cudaProfilerStart/Stop，故不用 cudaProfilerApi，采集整次运行
sudo -E /usr/local/cuda/bin/nsys profile \
  --output="/home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_overlap_only" \
  --trace=cuda,nvtx \
  --sample=cpu \
  --gpu-metrics-device=none \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --stats=true \
  /opt/conda/bin/python latency.py --cpu-offload 1 --overlap --output-dir "/home/yyj/fiddler/benchmarks/nsys_overlap_reports"

echo "完成: /home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_overlap_only.nsys-rep"
