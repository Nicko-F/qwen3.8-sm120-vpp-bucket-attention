# References

## Prefill and communication

- [VPP paper, arXiv:2608.26523](https://arxiv.org/abs/2608.26523): folded virtual
  stages for chunked long-context prefill.
- [VPP author implementation](https://github.com/RookieCoder-Camera/vllm-ascend/tree/c409752993b80524e4af6b98ef3b69a8f9c13f2c),
  especially [layer placement](https://github.com/RookieCoder-Camera/vllm-ascend/blob/c409752993b80524e4af6b98ef3b69a8f9c13f2c/vllm_ascend/distributed/vpp_utils.py):
  source for folded placement and continuation ordering. This project adapts
  those ideas to CUDA workers and same-host DMA/NCCL transport.
- [NCCL shared-memory transport](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/transport/shm.cc):
  reference for bounded host-staged copy and event progress.

## Bucket attention

- [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops): dynamic decode work
  scheduling for variable KV lengths. The studied
  [assignment implementation](https://github.com/Tencent/hpc-ops/blob/2a2e26562433a8ba4b504858f1c938eb7612c901/src/attention/decode/sm90/assign_task.cu)
  and [benchmarks](https://github.com/Tencent/hpc-ops/tree/2a2e26562433a8ba4b504858f1c938eb7612c901/benchmark)
  inform global KV-work partitioning and length-distribution comparisons.
- [FlashInfer XQA source](https://github.com/flashinfer-ai/flashinfer/tree/0179e5b312d6aa0c790e8e571b9037bb277f4dbd/csrc/xqa):
  cache loading, attention math, partial outputs and merge semantics. The SM120
  adaptation combines this task body with NVFP4 bucket scheduling.
- [Multi-shape FlashInfer development branch](https://github.com/Nicko-F/flashinfer/tree/feat/sm120-nvfp4-decode-bucketing):
  related work on additional head ratios, page sizes and query counts.

## Runtime and kernels

- [vLLM](https://github.com/vllm-project/vllm),
  [quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) and
  [benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/): serving,
  scheduling, graph execution and performance measurement.
- [vLLM plugin design](https://github.com/vllm-project/vllm/blob/main/docs/design/plugin_system.md):
  extension architecture. The source map in [integration design](design.md)
  identifies this project's worker and model-runner hooks.
- [CUTLASS SM120 NVFP4 example](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu):
  NVFP4 consumer layouts and hardware interfaces.
- [PyTorch CUDA/C++ extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html):
  runtime compilation and installed-header discovery.

Source revisions identify the implementations studied and adapted. Copyright,
license texts and modification notices are in [NOTICE](../NOTICE) and
[licenses/](../licenses/).
