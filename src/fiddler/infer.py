import argparse
import os

from mixtral import FiddlerMixtral


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="University of Washington is",
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=20,
        help="Number of tokens to generate.",
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")
    parser.add_argument(
        "--no-hot",
        action="store_true",
        help="专家放置随机化：不按 profile 热点顺序，随机选择哪些专家常驻 GPU（用于公平对比）。",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="feat: GPU 与 CPU expert 计算并行（仅当 cpu_offload=1 时生效）。",
    )
    parser.add_argument(
        "--yyj-improve-cost",
        dest="yyj_improve_cost",
        action="store_true",
        help="合并 cost 与 decide best：在算每个 expert cost 时顺带贪心决定 CPU/GPU，省略 256 次枚举。",
    )

    args = parser.parse_args()
    model = FiddlerMixtral(args)
    prefill_time, decode_time, hit_rate = model.generate(
        args.input, output_token=args.n_token
    )
    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    )
