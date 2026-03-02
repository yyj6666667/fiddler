####
# without stack
####
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/mixtral-offloading_profiler_trace_no_stack.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_no_stack.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/latency.txt gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/cpu_offload_performance_comparison.md gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/overlap_performance_comparison.md gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_no_stack_cpu_offload_0.json gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_no_stack_cpu_offload_1.json gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_cpu_offload_1_overlap_0.json gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_cpu_offload_1_overlap_1.json gs://856356105879-us-central1-blueprint-config/

####
# nsys overlap 时间线（nsys_compare_overlap.sh 默认输出目录）
####
gcloud storage cp /home/yyj/fiddler/benchmarks/nsys_overlap_reports/nsys_compare_overlap.nsys-rep gs://856356105879-us-central1-blueprint-config/

# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/mixtral-offloading_profiler_trace_no_stack.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_no_stack.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/latency.txt ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/cpu_offload_performance_comparison.md ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/overlap_performance_comparison.md ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_no_stack_cpu_offload_0.json ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_no_stack_cpu_offload_1.json ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_cpu_offload_1_overlap_0.json ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_cpu_offload_1_overlap_1.json ./
# nsys overlap 时间线
gcloud storage cp gs://856356105879-us-central1-blueprint-config/nsys_compare_overlap.nsys-rep ./

####
# with stack
####
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/mixtral-offloading_profiler_trace.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/latency.txt gs://856356105879-us-central1-blueprint-config/


# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/mixtral-offloading_profiler_trace.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/latency.txt ./