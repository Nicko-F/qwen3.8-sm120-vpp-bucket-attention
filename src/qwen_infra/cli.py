"""Native serving plus a bounded P+1 example, with CPU-only launch previews."""

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import shlex
import sys

from qwen_infra.config import PREFILL_PROFILES, SERVE_PROFILES, PrefillConfig, decode_options


def serve_command(args, passthrough):
    options = decode_options(args.profile)
    if args.tensor_parallel_size is not None:
        options["tensor_parallel_size"] = args.tensor_parallel_size
    elif args.profile != "native":
        options["tensor_parallel_size"] = 4
    argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            argv.extend(
                [flag, json.dumps(value, separators=(",", ":")) if isinstance(value, dict) else str(value)]
            )
    return argv + passthrough


def doctor(check_cuda=False):
    result = dict(python=sys.version.split()[0], platform=sys.platform, packages={})
    for distribution in ("qwen-infra", "vllm", "torch", "flashinfer-python", "nvidia-cutlass-dsl"):
        try:
            result["packages"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][distribution] = None
    from qwen_infra.cache import cache_root

    result["cache"] = str(cache_root())
    result["cuda_home"] = os.environ.get("CUDA_HOME")
    result["native_runtime_present"] = importlib.util.find_spec("vllm") is not None
    if check_cuda:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["devices"] = [
            dict(
                index=i,
                name=torch.cuda.get_device_name(i),
                capability=list(torch.cuda.get_device_capability(i)),
            )
            for i in range(torch.cuda.device_count())
        ]
    return result


def _prefill(args):
    config = PrefillConfig(
        args.model,
        args.profile,
        args.chunk_size,
        args.max_model_len,
        max(8, args.requests),
        args.gpu_memory_utilization,
        args.transport,
    )
    options = config.engine_options()
    if args.requests < 1 or args.prompt_tokens is not None and args.prompt_tokens < 1:
        raise ValueError("Requests and prompt token counts must be positive")
    if args.prompt_tokens is not None and args.prompt_tokens + 1 > args.max_model_len:
        raise ValueError("Prompt plus one output token must fit --max-model-len")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    engine=options,
                    sampling={"max_tokens": 1, "temperature": 0},
                    requests=args.requests,
                    prompt_tokens=args.prompt_tokens,
                ),
                indent=2,
            )
        )
        return 0
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MAX_JOBS", "2")
    from qwen_infra.runtime_paths import prepare_jit_environment

    prepare_jit_environment()
    if args.profile == "vpp" and args.transport == "segment-dma":
        os.environ.setdefault("NCCL_COMM_BLOCKING", "1")
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig

    text = AutoConfig.from_pretrained(args.model).get_text_config()
    if args.profile == "vpp" and (
        text.num_hidden_layers != 64 or text.hidden_size != 5120 or "moe" in text.model_type
    ):
        raise ValueError("VPP currently supports dense 64-layer H5120 Qwen models at TP1/PP4")
    # Resolve the class explicitly: the native config accepts a class for custom executors.
    if args.profile == "vpp":
        from qwen_infra.vpp.executor import FoldedExecutor

        options["distributed_executor_backend"] = FoldedExecutor
    llm = LLM(**options)
    try:
        if args.prompt_tokens is not None:
            tokenizer = llm.get_tokenizer()
            token = tokenizer.encode("Hello", add_special_tokens=False)
            if not token:
                raise ValueError("Tokenizer returned an empty seed token sequence")
            prompt = {"prompt_token_ids": [token[0]] * args.prompt_tokens}
        elif args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        else:
            prompt = args.prompt or "Explain why the sky is blue."
        outputs = llm.generate(
            [prompt for _ in range(args.requests)], SamplingParams(max_tokens=1, temperature=0)
        )
        print(
            json.dumps(
                [
                    dict(
                        request_id=o.request_id,
                        text=o.outputs[0].text,
                        token_ids=o.outputs[0].token_ids,
                        finished=o.finished,
                    )
                    for o in outputs
                ],
                indent=2,
            )
        )
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            if args.profile == "vpp":
                llm.collective_rpc("qwen_infra_vpp_close_transport", timeout=60)
        except Exception:
            if not failed:
                raise
            logging.getLogger(__name__).exception("VPP transport cleanup failed after inference failure")
        finally:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                if not failed:
                    raise
                logging.getLogger(__name__).exception("Native engine cleanup failed after inference failure")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qwen-infra")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("doctor", help="Show installed packages without allocating a GPU")
    inspect.add_argument(
        "--cuda", action="store_true", help="Also initialize CUDA and inspect visible devices"
    )
    serve = sub.add_parser("serve", help="Launch the native vLLM OpenAI-compatible service")
    serve.add_argument("--model", required=True)
    serve.add_argument("--profile", choices=SERVE_PROFILES, default="native")
    serve.add_argument("--tensor-parallel-size", type=int)
    serve.add_argument("--dry-run", action="store_true")
    prefill = sub.add_parser("prefill", help="Run a bounded four-GPU prefill plus one output token")
    prefill.add_argument("--model", required=True)
    prefill.add_argument("--profile", choices=PREFILL_PROFILES, default="vpp")
    prefill.add_argument("--transport", choices=("segment-dma", "nccl"), default="segment-dma")
    prefill.add_argument("--chunk-size", type=int, default=2048)
    prefill.add_argument("--max-model-len", type=int, default=32768)
    prefill.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    prefill.add_argument("--requests", type=int, default=1)
    inputs = prefill.add_mutually_exclusive_group()
    inputs.add_argument("--prompt")
    inputs.add_argument("--prompt-file")
    inputs.add_argument("--prompt-tokens", type=int)
    prefill.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if args.command != "serve" and extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    if args.command == "doctor":
        print(json.dumps(doctor(args.cuda), indent=2))
        return 0
    if args.command == "prefill":
        return _prefill(args)
    command = serve_command(args, extra[1:] if extra[:1] == ["--"] else extra)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MAX_JOBS", "2")
    if args.profile != "native":
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        from qwen_infra.runtime_paths import prepare_jit_environment

        prepare_jit_environment()
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
