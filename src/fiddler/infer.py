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
        help="启用 PyTorch Profiler：Chrome trace 保存到 asset/fiddler_profiler_trace_no_stack.json，内存快照保存到 asset/fiddler_memory_snapshot.pickle（可拖入 https://pytorch.org/memory_viz 查看）。",
    )

    args = parser.parse_args()
    model = FiddlerMixtral(args)

    if args.profile:
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history()
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
        if torch.cuda.is_available():
            memory_snapshot_path = os.path.join(asset_dir, "fiddler_memory_snapshot.pickle")
            try:
                torch.cuda.memory._dump_snapshot(memory_snapshot_path)
                print(
                    f"内存快照已保存到 {memory_snapshot_path}，可拖入 https://pytorch.org/memory_viz 查看。"
                )
            except Exception as e:
                print(
                    f"导出内存快照时出错：{e}；Chrome trace 仍可用。"
                )
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
    else:
        prefill_time, decode_time, hit_rate = model.generate(
            args.input, output_token=args.n_token
        )

    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    )
