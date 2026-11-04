"""Small installed-package model anchor; prints routing metadata, never timings."""

import argparse
import json
import os

from qwen_infra.config import decode_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--profile", choices=("native", "decode-bucket", "decode-fused"), default="decode-fused"
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--long-length", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.long_length or args.long_length + args.max_tokens > args.max_model_len:
        parser.error("Long prompt plus output must fit --max-model-len")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MAX_JOBS", "2")
    from qwen_infra.runtime_paths import prepare_jit_environment

    prepare_jit_environment()
    from vllm import LLM, SamplingParams

    options = decode_options(args.profile)
    options.update(max_num_seqs=32, max_num_batched_tokens=2048)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **options,
    )
    try:
        token = llm.get_tokenizer().encode("Hello", add_special_tokens=False)[0]
        banks = [("short", [32]), ("skew", [args.long_length] + [64] * 8), ("uniform", [4096] * 32)]
        for name, lengths in banks:
            prompts = [{"prompt_token_ids": [token] * length} for length in lengths]
            outputs = llm.generate(
                prompts, SamplingParams(max_tokens=args.max_tokens, temperature=0, ignore_eos=True)
            )
            row = dict(
                bank=name,
                lengths=lengths,
                completed=len(outputs),
                finished=all(output.finished for output in outputs),
                output_tokens=[len(output.outputs[0].token_ids) for output in outputs],
            )
            if args.profile != "native":
                row["workers"] = llm.collective_rpc("qwen_infra_status", timeout=60)
            print(json.dumps(row), flush=True)
            if not row["finished"] or len(outputs) != len(lengths) or not all(row["output_tokens"]):
                raise RuntimeError("Incomplete model smoke outputs")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
