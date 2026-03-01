####
# without stack
####
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/mixtral-offloading_profiler_trace_no_stack.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_no_stack.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/mixtral-offloading_memory_snapshot.pickle gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_memory_snapshot.pickle gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/latency.txt gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/cpu_offload_performance_comparison.md gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_no_stack_cpu_offload_0.json gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace_no_stack_cpu_offload_1.json gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_memory_snapshot_cpu_offload_0.pickle gs://856356105879-us-central1-blueprint-config/
gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_memory_snapshot_cpu_offload_1.pickle gs://856356105879-us-central1-blueprint-config/


# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/mixtral-offloading_profiler_trace_no_stack.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/mixtral-offloading_memory_snapshot.pickle ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_no_stack.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_memory_snapshot.pickle ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/latency.txt ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/cpu_offload_performance_comparison.md ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_no_stack_cpu_offload_0.json ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace_no_stack_cpu_offload_1.json ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_memory_snapshot_cpu_offload_0.pickle ./
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_memory_snapshot_cpu_offload_1.pickle ./


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