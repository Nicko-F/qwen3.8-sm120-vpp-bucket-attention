# Source references

## Serving and packaging

- [vLLM README](https://github.com/vllm-project/vllm/blob/main/README.md),
  [Python packaging](https://github.com/vllm-project/vllm/blob/main/pyproject.toml)
  and [quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html):
  normal installation, a small CLI and the existing OpenAI-compatible server.
- [vLLM plugin design](https://github.com/vllm-project/vllm/blob/main/docs/design/plugin_system.md):
  extension boundaries. This package also uses version-sensitive internal
  worker/model interfaces, which are documented rather than presented as stable
  public vLLM APIs.

## Prefill VPP

- [VPP paper, arXiv:2608.26523](https://arxiv.org/abs/2608.26523): folded virtual
  stages for chunked long-context prefill.
- [Author implementation](https://github.com/RookieCoder-Camera/vllm-ascend/tree/c409752993b80524e4af6b98ef3b69a8f9c13f2c),
  especially
  [layer placement](https://github.com/RookieCoder-Camera/vllm-ascend/blob/c409752993b80524e4af6b98ef3b69a8f9c13f2c/vllm_ascend/distributed/vpp_utils.py):
  the CUDA adaptation retains folded placement and continuation ordering while
  providing its own bounded same-host DMA/NCCL transport.
- [NCCL shared-memory transport](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/transport/shm.cc):
  a design reference for bounded copy/event progress, not a vendored NCCL plugin.

## Decode bucket attention

- [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops/tree/2a2e26562433a8ba4b504858f1c938eb7612c901),
  including
  [task assignment](https://github.com/Tencent/hpc-ops/blob/2a2e26562433a8ba4b504858f1c938eb7612c901/src/attention/decode/sm90/assign_task.cu)
  and [benchmarks](https://github.com/Tencent/hpc-ops/tree/2a2e26562433a8ba4b504858f1c938eb7612c901/benchmark):
  global KV-work partitioning and length distributions. Its Hopper BF16/FP8
  implementation is not described as an unchanged SM120 NVFP4 kernel.
- [FlashInfer XQA](https://github.com/flashinfer-ai/flashinfer/tree/0179e5b312d6aa0c790e8e571b9037bb277f4dbd/csrc/xqa):
  native cache loading, attention mathematics, partial-output and merge semantics.
  The maintained task adaptation preserves the NVIDIA Apache license header.
- [Separate FlashInfer contribution](https://github.com/Nicko-F/flashinfer/tree/feat/sm120-nvfp4-decode-bucketing):
  broader head/page/query support under development. Its fixed-mode integration
  and measurements are separate from this release's automatic-routing/fusion
  deployment path.

## Precision and build interfaces

- [CUTLASS SM120 NVFP4 example](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu):
  NVFP4 consumer layout and hardware capability reference.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and
  [PyTorch extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html):
  installed-package header discovery and ordinary CUDA/C++ extension building.

Revisions identify sources studied and adapted. They are not runtime admission
checks. License texts and modification notices are collected in [NOTICE](../NOTICE).
