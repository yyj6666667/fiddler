"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append("../src")
from fiddler import FiddlerMixtral


def write_comparison(output_dir, results, output_token):
    """将 cpu_offload=0 与 1 的关键性能写入 output_dir/cpu_offload_performance_comparison.md"""
    total_time_0 = results[0][1] + results[0][2]
    total_time_1 = results[1][1] + results[1][2]
    tokens_per_sec_0 = output_token / total_time_0 if total_time_0 > 0 else 0
    tokens_per_sec_1 = output_token / total_time_1 if total_time_1 > 0 else 0
    path = os.path.join(output_dir, "cpu_offload_performance_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fiddler cpu_offload 关键性能对比\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| 指标 | cpu_offload=0 (GPU baseline) | cpu_offload=1 (offload) |\n")
        f.write("|------|----------------------------|------------------------|\n")
        f.write(f"| prefill_time (s) | {results[0][1]:.4f} | {results[1][1]:.4f} |\n")
        f.write(f"| decode_time (s) | {results[0][2]:.4f} | {results[1][2]:.4f} |\n")
        f.write(f"| hit_rate | {results[0][3]:.4f} | {results[1][3]:.4f} |\n")
        f.write(f"| tokens/s | {tokens_per_sec_0:.2f} | {tokens_per_sec_1:.2f} |\n")
        f.write(
            "\n说明: cpu_offload=0 为以 GPU 执行为主的 baseline；cpu_offload=1 为 prefill 阶段 CPU offload 调度。\n"
        )
    print(f"关键性能对比已写入 {path}")


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
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference.",
    )
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用 PyTorch Profiler：trace 与内存快照文件名会包含 cpu_offload，写入 --output-dir（可拖入 chrome://tracing 与 https://pytorch.org/memory_viz）。",
    )
    parser.add_argument(
        "--compare-cpu-offload",
        action="store_true",
        help="依次以 cpu_offload=0 与 1 各跑一次（input/output_token 同当前循环），将关键性能对比写入 output-dir/cpu_offload_performance_comparison.md。可与 --profile 同时使用。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="目录：profiler trace 与 latency.txt 的写入路径，默认当前目录。",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    path_json = "./ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)

    if args.compare_cpu_offload:
        input_token, output_token = 16, 16
        idx_text = 0
        while idx_text < len(texts) and len(texts[idx_text].split()) < input_token:
            idx_text += 1
        if idx_text >= len(texts):
            raise SystemExit("compare-cpu-offload: 未找到足够长的输入文本。")
        text = texts[idx_text]
        results = []
        for offload in [0, 1]:
            args.cpu_offload = offload
            print(f"\n=== cpu_offload={offload} ===")
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
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    with record_function("fiddler_generate"):
                        prefill_time, decode_time, hit_rate = model.generate(
                            [text],
                            output_token=output_token,
                            input_token=input_token,
                        )
                trace_basename = f"fiddler_profiler_trace_no_stack_cpu_offload_{offload}.json"
                trace_path = os.path.join(args.output_dir, trace_basename)
                prof.export_chrome_trace(trace_path)
                print(f"Chrome trace 已保存到 {trace_path}，可用 chrome://tracing 打开查看。")
                if torch.cuda.is_available():
                    snapshot_basename = f"fiddler_memory_snapshot_cpu_offload_{offload}.pickle"
                    memory_snapshot_path = os.path.join(args.output_dir, snapshot_basename)
                    try:
                        torch.cuda.memory._dump_snapshot(memory_snapshot_path)
                        print(
                            f"内存快照已保存到 {memory_snapshot_path}，可拖入 https://pytorch.org/memory_viz 查看。"
                        )
                    except Exception as e:
                        print(f"导出内存快照时出错：{e}；Chrome trace 仍可用。")
                    finally:
                        torch.cuda.memory._record_memory_history(enabled=None)
            else:
                prefill_time, decode_time, hit_rate = model.generate(
                    [text], output_token=output_token, input_token=input_token
                )
            results.append((offload, prefill_time, decode_time, hit_rate))
            print(
                f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_comparison(args.output_dir, results, output_token)
        print(
            f"\n汇总: prefill_time (0/1) = {results[0][1]:.4f}/{results[1][1]:.4f}s, "
            f"decode_time = {results[0][2]:.4f}/{results[1][2]:.4f}s, "
            f"hit_rate = {results[0][3]:.4f}/{results[1][3]:.4f}"
        )
        sys.exit(0)

    model = FiddlerMixtral(args)
    n_sample = 1

    did_profile = False

    for input_token in [16]:
        for output_token in [16]:
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break

                if args.profile and (not did_profile):
                    # 内存快照仅记录本段（第一次 generate：prefill+decode），不包含模型加载等
                    if torch.cuda.is_available():
                        torch.cuda.memory._record_memory_history()
                    activities = (
                        [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        if torch.cuda.is_available()
                        else [ProfilerActivity.CPU]
                    )
                    with profile(
                        activities=activities,
                        profile_memory=True,
                        with_stack=False,
                    ) as prof:
                        with record_function("fiddler_generate"):
                            prefill_time, decode_time, hit_rate = model.generate(
                                [text],
                                output_token=output_token,
                                input_token=input_token,
                            )

                    did_profile = True

                    trace_basename = f"fiddler_profiler_trace_no_stack_cpu_offload_{args.cpu_offload}.json"
                    trace_path = os.path.join(args.output_dir, trace_basename)
                    prof.export_chrome_trace(trace_path)
                    print(
                        f"Chrome trace 已保存到 {trace_path}，可用 chrome://tracing 打开查看。"
                    )
                    if torch.cuda.is_available():
                        memory_snapshot_basename = f"fiddler_memory_snapshot_cpu_offload_{args.cpu_offload}.pickle"
                        memory_snapshot_path = os.path.join(args.output_dir, memory_snapshot_basename)
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
                        [text], output_token=output_token, input_token=input_token
                    )

                prefill_time_sum += prefill_time
                decode_time_sum += decode_time
                hit_rate_sum += hit_rate

            # write to file
            latency_path = os.path.join(args.output_dir, "latency.txt")
            with open(latency_path, "a") as f:
                f.write(
                    f"input_token: {input_token}, output_token: {output_token}, "
                    f"prefill_time: {prefill_time_sum / n_sample}, "
                    f"decode_time: {decode_time_sum / n_sample}, "
                    f"hit_rate: {hit_rate_sum / n_sample},"
                    f"{output_token * n_sample / (prefill_time_sum + decode_time_sum):.2f}token/s\n"
                )
