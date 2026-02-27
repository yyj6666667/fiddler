#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

python src/fiddler/infer.py \
--input "${PROMPT:-写一首关于春天的诗}" \
--n-token 32 \
--profile
