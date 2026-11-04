# Qwen3.8 SM120: VPP & Bucket Attention

[![CI](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[中文](README.zh-CN.md) · [Architecture](docs/architecture.md) ·
[Results](docs/results.md) · [Installation](docs/installation.md) ·
[Contributing](CONTRIBUTING.md)

Inference optimizations for Qwen3.8 on NVIDIA SM120 GPUs, built on **vLLM and
FlashInfer**:

- **Prefill:** folded virtual pipeline parallelism (VPP), with segmented CUDA
  DMA or NCCL boundary transport.
- **Decode:** KV-length-aware bucket attention, shared planning across compatible
  layers, and native fallback for unsuitable workloads.
- **Kernels:** optional residual/RMSNorm → NVFP4 packing, QK norm/RoPE/KV append,
  and attention-output/FP8 quantization fusion.

This repository contains the deployable source package. Model weights and the
research workspace's environments, caches and experiment orchestration are
separate. The new package's installation validation is tracked in
[installation](docs/installation.md); the [performance tables](docs/results.md)
come from the completed research experiments.

## Quick start

Use Linux with NVIDIA GPUs. After cloning, bootstrap creates a private virtual
environment and starts vLLM's OpenAI-compatible server:

```bash
git clone https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention.git
cd qwen3.8-sm120-vpp-bucket-attention
bash scripts/bootstrap.sh --with-toolkit -- serve \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-bucket \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --host 127.0.0.1 --port 8000
```

`--model` accepts your own local directory or Hugging Face model identifier.
The tested checkpoint is [NVIDIA's mixed NVFP4
release](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4); use a local copy to
skip downloading it again.
Select GPUs with the standard `CUDA_VISIBLE_DEVICES` environment variable.
The four-GPU setting above is the tested model deployment, not a requirement of
native mode. See [supported configurations](docs/installation.md#support).
Omit `--with-toolkit` when a compatible CUDA toolkit is already available.

```bash
# Inspect dependencies and capabilities without loading model weights.
.venv/bin/qwen-infra doctor

# Keep the original vLLM path, or explicitly enable the fusion candidate.
.venv/bin/qwen-infra serve --model /path/to/model --profile native
.venv/bin/qwen-infra serve --model /path/to/model --profile decode-fused \
  --tensor-parallel-size 4

# Run a P+1 prefill bank with four folded VPP ranks.
.venv/bin/qwen-infra prefill --model /path/to/Qwen-NVFP4 --profile vpp \
  --chunk-size 2048 --prompt-tokens 8192 --requests 1
```

Native vLLM arguments remain available. Use `--dry-run` to inspect the resolved
launch before allocating a model. No historical run, source hash, private
directory or author-specific environment is required.

## Measured results

The completed experiments used four same-NUMA RTX 5090 GPUs, mixed NVFP4 weights,
NVFP4 KV and native model arithmetic. Metrics below have different scopes and
must not be added together.

| Scope | Observation |
| --- | --- |
| Static prefill, four requests | Best VPP configurations reduced bank latency by **4.47–8.55%** against the best measured CPP configurations over 8K–128K inputs. |
| Full-model decode, severe KV skew | Bucket routing reduced measured step latency by **28.28% / 36.55%** in the declared B9 / B32 windows. |
| Finite mixed HTTP banks | Bucket attention improved output throughput by about **10–11%** in the measured mixed-arrival banks; uniform throughput was near parity. |
| Fusion, controlled GPU calls | All three fusions reduced latency by **2.04–7.18%** against the same bucket path over the tested 1K/2K contexts and batch sizes. |
| Fusion, HTTP | Additional throughput changes ranged from **−1.46% to +1.76%**; there is no universal serving gain. |

See [results and limitations](docs/results.md) for baselines, absolute timings,
negative results and measurement boundaries. These are historical measurements,
not a claim that every fresh installation has been benchmarked.

## Project layout

```text
src/qwen_infra/   Python integration and packaged CUDA/C++ sources
scripts/          Private-environment bootstrap
examples/         Small usage examples
tests/            Configuration, packaging and integration checks
benchmarks/       Curated tables and benchmark methodology
docs/             Architecture, installation, results and source references
```

## License and acknowledgements

Project code is under [Apache-2.0](LICENSE), with third-party notices preserved in
[NOTICE](NOTICE) and [licenses/](licenses/). Key references are
[VPP](https://arxiv.org/abs/2608.26523),
[Tencent HPC-Ops](https://github.com/Tencent/hpc-ops),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer), and
[vLLM](https://github.com/vllm-project/vllm).
