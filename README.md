# Qwen3.8 SM120: VPP & Bucket Attention

[![CI](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml)
[![GPU: SM120](https://img.shields.io/badge/GPU-SM120-76B900)](docs/installation.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Faster Qwen3.8 inference on NVIDIA Blackwell GPUs, from pipeline scheduling to CUDA kernels.**
Built on vLLM and FlashInfer, this project combines folded VPP for long-context
prefill, load-balanced bucket attention for decode, and fused low-precision
operators around attention and MLP inputs.

[中文](README.zh-CN.md) · [Features](#features) · [Performance](#performance) ·
[Quick start](#quick-start) · [Architecture](docs/architecture.md) ·
[Benchmarks](benchmarks/README.md)

## Features

- **Folded VPP prefill.** Eight virtual stages run across four GPUs in a folded
  pipeline, reducing fill/drain bubbles as attention cost grows with context.
  Segmented CUDA DMA overlaps boundary transfers with computation through copy
  engines and a compact, rank-shared buffer pool.
- **KV-length-aware decode.** A shared task planner balances KV work across
  CTAs for mixed-length requests. Automatic native, aligned and persistent paths
  adapt execution to the workload; compatible attention layers reuse the plan.
- **Fused CUDA operators.** Residual/RMSNorm → NVFP4 packing, QK Norm/RoPE/KV
  append, and attention-output/gate → FP8 quantization reduce launches and
  intermediate memory traffic.
- **vLLM integration.** Serving profiles keep vLLM's continuous batching,
  sampling and OpenAI-compatible API. Prefill and decode have separate execution
  profiles for development and benchmarking.

| Component | Workload | Implementation | Entry point |
| --- | --- | --- | --- |
| Folded VPP | Long-context prefill, TP1/PP4 | Eight virtual stages; segmented DMA or NCCL transport | `qwen-infra prefill --profile vpp` |
| Bucket attention | Decode with uneven KV lengths | NVFP4 paged KV; shared planning; FlashInfer XQA task execution | `qwen-infra serve --profile decode-bucket` |
| Kernel fusion | Decode attention and MLP boundaries | CUDA C++ producers for packed NVFP4 and FP8 consumers | `qwen-infra serve --profile decode-fused` |
| Native execution | Reference comparisons | Original vLLM serving and CPP prefill | `--profile native` / `--profile cpp` |

## Performance

Representative results on **four NUMA-local RTX 5090 GPUs**, using
`nvidia/Qwen3.8-27B-NVFP4` with NVFP4 KV cache:

| Optimization | Benchmark scope | Baseline | Result |
| --- | --- | --- | --- |
| VPP prefill | Four requests, 32K–128K inputs; selected chunk/graph configurations | CPP | **5.19–8.90% lower bank latency** |
| Bucket attention | Full-model decode, severely uneven KV lengths, B9 / B32 | Native vLLM | **28.28% / 36.55% lower step latency** |
| Bucket serving | Mixed-length HTTP arrivals | Native vLLM | **10–11% higher output throughput** |
| Three kernel fusions | Full GPU decode call, B1/8/16/32, 1K/2K contexts | Same bucket attention path | **2.04–7.18% lower latency** |

See [performance results](docs/results.md) for absolute timings and workload
definitions, and [benchmarking](benchmarks/README.md) to run comparisons.

## Quick start

### Install and serve

On Linux with SM120 GPUs, bootstrap creates `.venv`, installs the runtime and
starts the OpenAI-compatible server:

```bash
git clone https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention.git
cd qwen3.8-sm120-vpp-bucket-attention
bash scripts/bootstrap.sh --with-toolkit -- serve \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-bucket --tensor-parallel-size 4 \
  --max-model-len 32768 --host 127.0.0.1 --port 8000
```

Use a local checkpoint directory with `--model`, select devices through
`CUDA_VISIBLE_DEVICES`, and omit `--with-toolkit` when CUDA is already installed.
See [installation and supported configurations](docs/installation.md).

### Enable fused operators

```bash
.venv/bin/qwen-infra serve --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-fused --tensor-parallel-size 4 \
  --max-model-len 32768 --host 127.0.0.1 --port 8000
```

### Run prefill

Run four 8K prompts through VPP, generating one output token per request:

```bash
.venv/bin/qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile vpp --chunk-size 2048 --prompt-tokens 8192 --requests 4
```

Use `--profile cpp` for the native pipeline comparison. The VPP DMA path shares
two output slots and three input slots per rank; the
[communication design](docs/vpp-resources.md) explains streams, prefetch and
buffer lifetimes.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Runtime setup, launch options and supported configurations |
| [Architecture](docs/architecture.md) | VPP, bucket attention, fusion and precision |
| [Integration design](docs/design.md) | vLLM extension points and source map |
| [VPP communication](docs/vpp-resources.md) | Copy pipeline, events, credits and memory budget |
| [Performance](docs/results.md) | Prefill, decode, operator and HTTP results |
| [Benchmarking](benchmarks/README.md) | Comparison commands and data tables |
| [Testing](docs/validation.md) | CPU tests, CUDA checks and model integration |

## Acknowledgements

This project builds on [vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[Tencent HPC-Ops](https://github.com/Tencent/hpc-ops), and the
[VPP paper and implementation](https://arxiv.org/abs/2608.26523).
See [references](docs/references.md) for the algorithms and source adaptations.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
Project code is licensed under [Apache-2.0](LICENSE), with third-party attribution
in [NOTICE](NOTICE) and [licenses/](licenses/).
