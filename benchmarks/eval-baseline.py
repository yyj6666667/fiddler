# fix numpy in colab
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy
import os
import sys
import argparse
import logging

# 所有路径基于本文件所在目录，与当前工作目录无关，避免与 Vs.sh 等调用方 cwd 冲突
BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARKS_DIR)


def main():
    os.chdir(os.path.join(BENCHMARKS_DIR, "mixtral_offloading"))

    if args.framework == 'mixtral-offloading':
        logging.info('Using mixtral-offloading')
        model = init_mixtral_offload()
    elif args.framework == 'deepspeed-mii':
        logging.info('Using deepspeed-mii')
        model = init_deepspeed_mii()
    else:
        raise ValueError(f'Unknown framework: {args.framework}')

    eval(model)


def init_deepspeed_mii():
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig

    model_id = "mistralai/Mixtral-8x7B-v0.1"
    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            }
        },
        "train_micro_batch_size_per_gpu": 1,
    }

    hfdsc = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16)

    deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model


def init_mixtral_offload():
    from hqq.core.quantize import BaseQuantizeConfig
    from mixtral_offloading.src.build_model import OffloadConfig, QuantConfig, build_model

    quantized = args.quantized

    # Use the same model identifier/path as fiddler if provided.
    # If args.model is a local directory, use it as state_path.
    # If args.model is a HF repo id:
    #   - quantized=True: allow downloading missing quantized files (W_q, etc.)
    #   - quantized=False: use local HF cache path only (no download)
    model_name = args.model
    if os.path.isdir(model_name):
        state_path = model_name
    else:
        if quantized:
            from huggingface_hub import snapshot_download
            state_path = snapshot_download(repo_id=model_name)
        else:
            cache_root = os.path.expanduser("~/.cache/huggingface/hub")
            blob_dir = "models--" + "--".join(model_name.split("/"))
            snapshots_dir = os.path.join(cache_root, blob_dir, "snapshots")
            if not os.path.isdir(snapshots_dir):
                raise FileNotFoundError(
                    f"未找到 HF 缓存目录 {snapshots_dir}，请先下载模型或指定本地 --model 路径。"
                )
            revisions = os.listdir(snapshots_dir)
            if not revisions:
                raise FileNotFoundError(f"HF 缓存目录为空: {snapshots_dir}")
            # 选择最近写入的 snapshot 目录（不触发下载/解析）
            revisions = sorted(
                revisions,
                key=lambda r: os.path.getmtime(os.path.join(snapshots_dir, r)),
                reverse=True,
            )
            state_path = os.path.join(snapshots_dir, revisions[0])

    config = AutoConfig.from_pretrained(model_name)

    device = torch.device("cuda:0")

    #as the offload_per_layer goes up, for sure the vram usage goes down
    offload_per_layer = 5
    ###############################################################

    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )

    if quantized:
        quant_config = QuantConfig(
            ffn_config=ffn_config,
            attn_config=attn_config)
    else:
        quant_config = None

    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
        model_name=model_name,
    )
    return model


def eval(model):
    import random
    import json
    import time

    device = torch.device("cuda:0")

    # Use dataset colocated with state_path if args.model is a local directory,
    # otherwise use benchmarks/ShareGPT (same as latency.py / Fiddler).
    if os.path.isdir(args.model):
        path_json = os.path.join(args.model, 'ShareGPT_V3_unfiltered_cleaned_split.json')
    else:
        path_json = os.path.join(BENCHMARKS_DIR, 'ShareGPT_V3_unfiltered_cleaned_split.json')
    with open(path_json, 'r') as f:
        data = json.load(f)
    texts = []
    for d in data:
        if len(d['conversations']) == 0:
            continue
        # the input of the first round
        texts.append(' '.join(d['conversations'][0]['value'].split()))

    logging.info(f'n of input {len(texts)}')
    random.seed(0)
    random.shuffle(texts)

    n_sample = 3

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    did_profile = False

    for input_token in [16]:
        for output_token in [16]:
            idx_text = 0
            time_sum = 0
            num_tokens = 0
            logging.info(
                f'evaluating -- input_token: {input_token}, output_token: {output_token}')
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                # print(f'input text: {text.split()[:input_token]}')
                input_ids = tokenizer.encode(
                    text, return_tensors='pt').to(device)
                start_time = time.time()

                # 只对第一次推理做一次完整的 PyTorch Profiler，避免开销太大
                if args.profile and (not did_profile):
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        if torch.cuda.is_available()
                        else [ProfilerActivity.CPU],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        with_flops=True,
                    ) as prof:
                        with record_function("model_generate"):
                            result = model.generate(
                                input_ids=input_ids[:, :input_token],
                                max_new_tokens=output_token,
                                min_new_tokens=output_token,
                                do_sample=True,
                                temperature=0.9,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=True,
                            )

                    did_profile = True

                    logging.info("PyTorch profiler summary (top 30 by CUDA time):")
                    logging.info(
                        "\n"
                        + prof.key_averages().table(
                            sort_by="cuda_time_total"
                            if torch.cuda.is_available()
                            else "cpu_time_total",
                            row_limit=30,
                        )
                    )

                    # 根据 framework 命名 trace 文件，方便区分不同 inference engine
                    trace_path = f"{args.framework}_profiler_trace.json"
                    prof.export_chrome_trace(trace_path)
                    logging.info(
                        f"Chrome trace 已保存到 {trace_path}，可用 chrome://tracing 打开查看。"
                    )
                else:
                    result = model.generate(
                        input_ids=input_ids[:, :input_token],
                        max_new_tokens=output_token,
                        min_new_tokens=output_token,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )

                end_time = time.time()
                time_sum += end_time - start_time
                # count the number of tokens in the output
                num_tokens += result["sequences"].shape[1]
                # print(f'output text: {tokenizer.decode(result["sequences"][0])}')

            logging.info(
                f'*******************\n'
                f'input_token: {input_token}, output_token: {output_token}, '
                f'time: {time_sum / n_sample:.2f}, '
                f'token/s: {output_token / (time_sum / n_sample):.2f}\n'
                f'*******************\n')


def _parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"Expected true/false, got {s!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quantized',
        type=_parse_bool,
        default=False,
        help='Whether to use quantized model in mixtral-offloading (true/false).',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mixtral-8x7B-v0.1',
        help='Model path or HF repo id to use for both fiddler and mixtral-offloading baselines.',
    )
    parser.add_argument(
        '--framework',
        type=str,
        default='mixtral-offloading',
        choices=[
            'mixtral-offloading',
            'deepspeed-mii'],
        help='Which framework to use for evaluation.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='启用 PyTorch Profiler 对第一次推理进行性能分析（同时导出 Chrome trace）。',
    )

    args = parser.parse_args()

    # save log to file
    logging.basicConfig(filename='eval.log', level=logging.INFO)
    main()
