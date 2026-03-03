import csv
import os

import matplotlib.pyplot as plt


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
    log_path = os.path.join(repo_root, "improve", "improve_loop.log")

    results = load_results(log_path)

    input_tokens = sorted({k[1] for k in results.keys()})
    output_tokens = sorted({k[2] for k in results.keys()})
    yyj_options = sorted({k[0] for k in results.keys()})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    markers = {16: "o", 32: "s", 64: "^"}
    colors = {0: "#1f77b4", 1: "#d62728"}

    for opt in yyj_options:
        for in_tok in input_tokens:
            xs = output_tokens
            ys = [results[(opt, in_tok, out_tok)] for out_tok in output_tokens]
            label = f"yyj_improve_loop={opt}, input={in_tok}"
            ax.plot(
                xs,
                ys,
                marker=markers.get(in_tok, "o"),
                color=colors.get(opt, "#333333"),
                linestyle="-" if opt == 1 else "--",
                linewidth=1.5,
                markersize=5,
                label=label,
            )

    ax.set_xlabel("Output tokens", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_title("Impact of yyj_improve_loop on Inference Throughput", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig_path = os.path.join(repo_root, "improve", "improve_loop.png")
    fig.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    main()

