#!/usr/bin/env bash

# 硬编码仓库根目录
REPO_ROOT="/home/yyj/fiddler"


cd "${REPO_ROOT}"
echo "[run_improve_loop] running bench_improve_loop.py ..."
python improve/bench_improve_loop.py


echo "[run_improve_loop] uploading log to GCS ..."
gcloud storage cp "${REPO_ROOT}/improve/improve_loop.log" \
  "gs://856356105879-us-central1-blueprint-config/improve_loop.log"

