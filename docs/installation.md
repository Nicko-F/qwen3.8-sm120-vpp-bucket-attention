# Installation and usage

qwen-infra adds folded VPP prefill and NVFP4 bucket decode to the native vLLM
runtime. The examples below use `nvidia/Qwen3.8-27B-NVFP4`; a Hugging Face model
identifier or local model directory can be passed to every model command.

## Requirements

- Linux x86-64 and Python 3.10 or later.
- An NVIDIA driver compatible with the CUDA 13 runtime.
- SM120 GPUs for the custom kernels. The tested model uses four RTX 5090s:
  TP4 for decode and TP1/PP4 for prefill.
- A host C++ compiler, Ninja and a CUDA 13 toolkit containing `nvcc` for JIT
  compilation. The bootstrap can install the toolkit inside its virtual environment.

The installation reference in [requirements-tested.txt](../requirements-tested.txt)
uses vLLM `0.32.0rc1.dev196+ga9fe30e9f`, PyTorch `2.14.0+cu130`,
FlashInfer `0.7.1`, Transformers `5.19.0` and CUTLASS DSL `4.8.0`.
The adapters use vLLM's internal worker and attention interfaces; test those
interfaces when upgrading the runtime.

## Install

From the repository root:

```bash
bash scripts/bootstrap.sh --with-toolkit -- doctor --cuda
source .venv/bin/activate
```

The bootstrap creates `.venv`, installs the tested runtime and qwen-infra,
and runs the command after `--`. With no command, it runs `doctor`.
The toolkit is installed under the selected environment's `cuda-toolkit`
directory. CUDA/C++ extensions compile on first use and reuse their build cache
on subsequent launches.

| Bootstrap option | Use |
| --- | --- |
| `--venv PATH` | Choose the virtual environment directory; default: `.venv`. |
| `--python PYTHON` | Choose the Python executable used to create the environment. |
| `--vllm-wheel URL` | Install a different vLLM wheel in place of the tested requirements file. |
| `--with-toolkit` | Install CUDA compiler/runtime packages into the selected environment. |
| `--no-prefetch` | Skip downloads of FlashInfer's prebuilt kernel providers. |
| `--package-only` | Install qwen-infra alone, useful for CPU tools or an already provisioned environment. |

Runtime installation prefetches FlashInfer's matching cubin wheel and minimal
SM120f JIT-cache wheel. For an offline host, provision those wheels and the
runtime dependencies in advance, then use `--no-prefetch`. When prebuilt
kernels are unavailable, FlashInfer builds them from source at startup; allow
additional time for this first launch.

To install into an existing compatible environment:

```bash
python -m pip install -e .
qwen-infra doctor --cuda
```

`doctor` reports package versions, the cache directory and `CUDA_HOME`.
Adding `--cuda` also reports visible GPUs and their compute capabilities.

## Serve

Launch the native OpenAI-compatible service:

```bash
qwen-infra serve --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile native --tensor-parallel-size 4 --host 127.0.0.1 --port 8000
```

To enable bucket decode, select `decode-bucket`:

```bash
qwen-infra serve --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-bucket --tensor-parallel-size 4 \
  --host 127.0.0.1 --port 8000
```

`decode-fused` adds the attention producer and consumer fusions. Both optimized
profiles configure BF16 activations, NVFP4 KV, FlashInfer attention, a 32-sequence
budget, synchronous scheduling and CUDA Graph capture. Additional serving
arguments are forwarded to vLLM. Append `--dry-run` to preview the launch command.

Send a request with [examples/openai_client.py](../examples/openai_client.py), or
use any OpenAI-compatible client pointed at `http://127.0.0.1:8000/v1`.
Model loading and downloads use the normal vLLM/Hugging Face mechanisms.

## Run prefill

The `prefill` command runs a request bank through prefill and generates one token
per request, abbreviated P+1. Compare conventional pipeline parallelism
(`cpp`) with the folded pipeline (`vpp`):

```bash
qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile cpp --requests 4 --prompt-tokens 8192 \
  --chunk-size 2048 --max-model-len 8193

qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile vpp --requests 4 --prompt-tokens 8192 \
  --chunk-size 2048 --max-model-len 8193
```

Use `--prompt "text"` or `--prompt-file PATH` for text input.
`--prompt-tokens N` creates a repeated-token input of exactly N tokens.
Set `--max-model-len` to cover the prompt plus its output token, and keep it
within the model's position range. `--chunk-size` controls the token budget per
prefill chunk; accepted values are 1–16384. Use `--gpu-memory-utilization` to
adjust the native memory budget, and `--dry-run` to inspect the engine options.

VPP uses rank-shared segmented DMA by default. `--transport nccl` selects the
native tensor-transport compatibility path. `--resource-layout edge` selects
the edge-owned buffer layout for resource comparisons.

## Supported configurations

This table describes the implemented optimization paths. The tested checkpoint
uses TP4 for its six-query-head/one-KV-head local decode geometry.

| Profile | Model and tensor contract | Execution |
| --- | --- | --- |
| `serve --profile native` | Models, precision and devices supported by the installed vLLM. | Native serving and parallelism. |
| `serve --profile decode-bucket` | SM120; BF16 queries/output; native NHD NVFP4 KV with 32-token pages; six Q heads, one KV head and head dimension 256 per rank; full attention with scale 1/16. | One pipeline stage; sequential V2 runner; one query token per request; up to 32 requests; optimized graph rows 1, 8, 16, 24 and 32. |
| `serve --profile decode-fused` | Bucket contract plus hidden size 5120, Gemma RMSNorm with epsilon 1e-6, gated attention and a static per-tensor FP8 output-projection input. | Bucket execution plus fused attention producers and consumers. |
| `prefill --profile cpp` | BF16 activations and NVFP4 KV through native FlashInfer. | Native TP1/PP4 prefill and one output token. |
| `prefill --profile vpp` | Dense 64-layer, hidden-size-5120 Qwen backbone. | TP1/PP4; two folded traversals; P+1 request banks; same-host Linux transport. |

The optimized decode worker checks the device, runner, model geometry and native
KV layout before installing its graphs. Unsupported configurations use native
attention, and startup logs report the reason. Unselected graph shapes and
non-decode work retain the native route. The task planner sizes its work from
the device's discovered SM count within its supported 32–340 range.

The optimized profiles use synchronous scheduling with prefix caching disabled.
Decode integration also requires speculation, LoRA, dual-batch overlap, context
parallelism, sleep mode and weight transfer to be disabled. VPP currently
supports P+1 banks with prefix caching, asynchronous scheduling, LoRA and
speculative decoding disabled. Contexts through 128K have been measured;
capacity is configured from the model, index bounds and available memory.

## Toolchains, caches and memory

The package discovers CUDA, NCCL, cuBLAS and FlashInfer from the active
environment. Use `CUDA_HOME` for an existing toolkit, `NCCL_HOME` or
`CUBLAS_HOME` for alternative dependency locations, and `QWEN_INFRA_CCCL_INCLUDE`
for a CCCL include directory containing `nv/target`.
Compiler errors retain their build logs for inspection.

The package cache follows `XDG_CACHE_HOME` or `~/.cache/qwen-infra`;
`QWEN_INFRA_CACHE_DIR` selects an explicit location. The bootstrap places
temporary, pip and XDG caches under the checkout's `.cache` directory unless
those environment variables are already set. Hugging Face, vLLM and PyTorch
also support their usual cache settings. `MAX_JOBS` controls build parallelism
and defaults to two; `OMP_NUM_THREADS` also defaults to two.

Select GPUs with `CUDA_VISIBLE_DEVICES`. For four-GPU measurements, use the same
ordered devices for both profiles and place the group within one NUMA domain.

Rank-shared VPP communication uses two source slots and three receive slots per
rank. Each slot holds BF16 hidden and residual planes of width 5120:

| Chunk capacity | Bytes per slot | Device communication payload per rank |
| --- | --- | --- |
| 2048 tokens | 40 MiB | 200 MiB |
| 8192 tokens | 160 MiB | 800 MiB |

Model weights, KV cache, recurrent state, CUDA Graphs and build workspace add
to this budget. The segmented transport also uses eight 16 MiB host segments
per edge. Reduce the chunk or native memory budget to fit the target machine.
`QWEN_INFRA_HOST_RESERVE_GIB` optionally reserves host-memory headroom for VPP
shared-buffer allocations; its default is zero.

See [VPP resources](vpp-resources.md) for buffer lifetimes and measurements,
[validation](validation.md) for tests, and [benchmarking](../benchmarks/README.md)
for comparison recipes.
