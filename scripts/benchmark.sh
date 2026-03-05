cd fiddler/improve
git stash
git pull
python bench_improve_loop.py
gcloud storage cp ./improve_both.log gs://856356105879-us-central1-blueprint-config/improve_both.log

gcloud storage cp gs://856356105879-us-central1-blueprint-config/improve_both.log ./