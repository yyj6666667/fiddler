import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(log_path):
    results = {}  # (yyj_improve_loop, input_token, output_token) -> tokens/s
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            yyj = int(row[0])
            in_tok = int(row[1])
            out_tok = int(row[2])
            tps = float(row[6])
            results[(yyj, in_tok, out_tok)] = tps
    return results


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(repo_root, "improve", "improve_both.log")

    results = load_results(log_path)

    input_tokens = sorted({k[1] for k in results.keys()})
    output_tokens = sorted({k[2] for k in results.keys()})
    yyj_options = sorted({k[0] for k in results.keys()})

    # 平均加速比：improve 全开(1) / 全关(0) 的 tps 之比，对所有 (in_tok, out_tok) 取平均
    speedups = []
    speedup_by_key = {}  # (in_tok, out_tok) -> speedup
    for in_tok in input_tokens:
        for out_tok in output_tokens:
            tps0 = results.get((0, in_tok, out_tok), 0.0)
            tps1 = results.get((1, in_tok, out_tok), 0.0)
            if tps0 > 0 and tps1 >= 0:
                sp = tps1 / tps0
                speedups.append(sp)
                speedup_by_key[(in_tok, out_tok)] = sp
            else:
                speedup_by_key[(in_tok, out_tok)] = 0.0
    avg_speedup = np.mean(speedups) if speedups else 0.0

    # 分组柱状图：每个 (input, output) 一组，组内两根柱子对应 yyj_improve_loop=0 和 1
    categories = [
        f"in{in_tok}\nout{out_tok}\n{speedup_by_key[(in_tok, out_tok)]:.3f}x"
        for in_tok in input_tokens
        for out_tok in output_tokens
    ]
    n_cats = len(categories)
    n_bars = len(yyj_options)
    bar_width = 0.35
    x = np.arange(n_cats)
    offsets = np.linspace(-bar_width / 2 * (n_bars - 1), bar_width / 2 * (n_bars - 1), n_bars)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "origin fiddler", 1: "yyj_improve"}

    for i_opt, opt in enumerate(yyj_options):
        heights = []
        for in_tok in input_tokens:
            for out_tok in output_tokens:
                heights.append(results.get((opt, in_tok, out_tok), 0.0))
        pos = x + offsets[i_opt]
        ax.bar(pos, heights, bar_width, label=labels.get(opt, f"opt={opt}"), color=colors.get(opt, "#333333"))

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_xlabel("(input_tokens, output_tokens, speedup)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    title = f"Impact of yyj_improve on Throughput (avg speedup = {avg_speedup:.3f}x)"
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig_path = os.path.join(repo_root, "improve", "improve_both.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Average speedup (improve on / off): {avg_speedup:.3f}x")


if __name__ == "__main__":
    main()

