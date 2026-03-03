import os
import sys
from types import SimpleNamespace

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_root = os.path.join(repo_root, "src")
sys.path.append(src_root)
from fiddler import FiddlerMixtral  # type: ignore


def build_args(model_name: str, yyj_improve_loop: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name,
        cpu_offload=1,
        batch_size=1,
        beam_width=1,
        no_hot=False,
        yyj_improve_loop=bool(yyj_improve_loop),
    )


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    input_tokens = [16, 32, 64]
    output_tokens = [16, 32, 64]
    yyj_options = [0, 1]
    model_name = "mistralai/Mixtral-8x7B-v0.1"

    os.makedirs(os.path.join(repo_root, "improve"), exist_ok=True)
    log_path = os.path.join(repo_root, "improve", "improve_loop.log")

    # 构造一个足够长的 dummy 文本，依靠 input_token 截断控制长度
    base_text = "hello world " * 256

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "# yyj_improve_loop 对 inference 性能影响评测\n"
            "# columns: yyj_improve_loop,input_token,output_token,prefill_time,decode_time,hit_rate,tokens_per_second\n"
        )

    for opt in yyj_options:
        args = build_args(model_name, opt)
        model = FiddlerMixtral(args)

        for in_tok in input_tokens:
            for out_tok in output_tokens:
                prefill_time, decode_time, hit_rate = model.generate(
                    [base_text], output_token=out_tok, input_token=in_tok
                )
                total_time = prefill_time + decode_time
                tps = out_tok / total_time if total_time > 0 else 0.0

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{opt},{in_tok},{out_tok},"
                        f"{prefill_time:.6f},{decode_time:.6f},{hit_rate:.6f},{tps:.3f}\n"
                    )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


