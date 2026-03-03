cd /home/yyj/fiddler/benchmarks
git stash
git pull
bash nsys_cmp.sh

cd /home/yyj/fiddler/benchmarks
git stash
git pull
bash run_overlap_bench.sh

cd /home/yyj/fiddler
git stash
git pull
chmod +x fiddler_cmp.sh && ./fiddler_cmp.sh

