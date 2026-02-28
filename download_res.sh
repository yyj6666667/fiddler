gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/mixtral-offloading_profiler_trace.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/fiddler_profiler_trace.json gs://856356105879-us-central1-blueprint-config/

gcloud storage cp /home/yyj/fiddler/benchmarks/vs_output/latency.txt gs://856356105879-us-central1-blueprint-config/


# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/mixtral-offloading_profiler_trace.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/fiddler_profiler_trace.json ./
# on local
gcloud storage cp gs://856356105879-us-central1-blueprint-config/latency.txt ./