import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function

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
        "--profile",
        action="store_true",
        help="启用 PyTorch Profiler：Chrome trace 保存到 asset/fiddler_profiler_trace_no_stack.json，内存时间线 HTML 保存到 asset/fiddler_memory_timeline.html。",
    )

    args = parser.parse_args()
    model = FiddlerMixtral(args)

    if args.profile:
        activities = (
            [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else [ProfilerActivity.CPU]
        )
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        ) as prof:
            with record_function("fiddler_generate"):
                prefill_time, decode_time, hit_rate = model.generate(
                    args.input, output_token=args.n_token
                )

        print("PyTorch profiler summary for Fiddler (top 30 by time):")
        sort_key = (
            "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        )
        print(prof.key_averages().table(sort_by=sort_key, row_limit=30))

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        asset_dir = os.path.join(project_root, "asset")
        os.makedirs(asset_dir, exist_ok=True)

        trace_path = os.path.join(asset_dir, "fiddler_profiler_trace_no_stack.json")
        prof.export_chrome_trace(trace_path)
        print(
            f"Chrome trace 已保存到 {trace_path}，可用 chrome://tracing 打开查看。"
        )

        memory_html_path = os.path.join(asset_dir, "fiddler_memory_timeline.html")
        try:
            prof.export_memory_timeline(memory_html_path)
            print(
                f"内存时间线 HTML 已保存到 {memory_html_path}，用浏览器打开可查看内存变化。"
            )
        except Exception as e:
            print(
                f"导出内存时间线时出错（可能在不支持的环境下）：{e}；Chrome trace 仍可用。"
            )
    else:
        prefill_time, decode_time, hit_rate = model.generate(
            args.input, output_token=args.n_token
        )

    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    )
