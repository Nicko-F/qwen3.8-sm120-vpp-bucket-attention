# Integration design

The package keeps model execution and serving inside vLLM. Workers install
prefill or decode extensions during model initialization, before profiling and
CUDA Graph capture. CUDA/C++ modules build lazily against the active runtime.

## Execution profiles

| Profile | vLLM integration | Execution |
| --- | --- | --- |
| `native` | Standard serving entry point | Native model, scheduler and attention |
| `decode-bucket` | `DecodeWorker` | Native/aligned/persistent attention routing with shared planning |
| `decode-fused` | `DecodeWorker` plus fusion hooks | Bucket attention with fused input/output producers |
| `cpp` | Standard prefill runner | Chunked prefill at TP1/PP4 |
| `vpp` | Folded executor, worker and model runner | Folded TP1/PP4 prefill with segmented DMA or NCCL |

The CLI accepts model IDs or paths and forwards serving arguments to vLLM.
Configuration and device selection use ordinary command-line arguments and
runtime environment variables; see [installation](installation.md).

## Prefill dataflow

The folded executor broadcasts admission and continuation messages. Rank-local
queues choose forward or return work while maintaining recurrent-state order.
The worker binds the selected virtual stage, acquires an input lease, runs its
layers and publishes hidden/residual outputs into a source slot.

Segmented DMA transfers the output through a registered host ring into a shared
receive slot on the next rank. CUDA events track each tensor plane. Source
credits return after D2H completion; receive credits return after the consuming
stage's completion event. Per-edge sequence and batch identities accompany
rank-shared physical storage.

| Source | Responsibility |
| --- | --- |
| [`vpp/layout.py`](../src/qwen_infra/vpp/layout.py) | Layer placement, rank-local order and window accounting |
| [`vpp/executor.py`](../src/qwen_infra/vpp/executor.py) | Admission, continuation and result collection |
| [`vpp/worker.py`](../src/qwen_infra/vpp/worker.py), [`runner.py`](../src/qwen_infra/vpp/runner.py) | Stage execution and input/output binding |
| [`vpp/model.py`](../src/qwen_infra/vpp/model.py) | Folded model adapter |
| [`vpp/producer.py`](../src/qwen_infra/vpp/producer.py) | Direct boundary output publication |
| [`vpp/native/`](../src/qwen_infra/vpp/native/) | Segmented DMA, host rings, events, credits and shared device pools |

The [communication design](vpp-resources.md) gives the complete stream and
buffer lifecycle.

## Decode graph lifecycle

`DecodeWorker` discovers eligible NVFP4 attention layers, creates shared planning
workspaces and installs execution adapters before graph capture. Layers with
compatible request metadata share the task map; layer-specific pointers and
partial-output scratch remain local to each attention call.

At the forward boundary the runtime refreshes lengths and selects a route.
Captured graphs use stable workspace addresses. Native graph entries remain
available for the native route, and the lifecycle restores those entries during
shutdown. Runtime capability and tensor checks select the native path for
configurations outside the extension's supported geometry.

| Source | Responsibility |
| --- | --- |
| [`worker.py`](../src/qwen_infra/worker.py) | Model-load hooks, lifecycle setup and status RPC |
| [`kernels/nvfp4_kv_bucket_worker.py`](../src/qwen_infra/kernels/nvfp4_kv_bucket_worker.py) | Shared workspaces, route selection and graph integration |
| [`kernels/nvfp4_bucket_xqa.py`](../src/qwen_infra/kernels/nvfp4_bucket_xqa.py) | Task planning, binding and CUDA launches |
| [`kernels/nvfp4_bucket_xqa.cu`](../src/qwen_infra/kernels/nvfp4_bucket_xqa.cu) | CUDA assignment and attention entry points |
| [`kernels/xqa_task_impl.cuh`](../src/qwen_infra/kernels/xqa_task_impl.cuh) | Adapted XQA task execution |

## Fusion integration

Fusion producers return the formats already expected by native GEMMs: packed
NVFP4 values/scales for the MLP and FP8 values for the attention output
projection. The residual stream stays BF16. Named graph owners hold captured
buffers throughout the graph lifecycle.

| Source | Responsibility |
| --- | --- |
| [`kernels/service_fusions.py`](../src/qwen_infra/kernels/service_fusions.py) | Install model and compiler hooks |
| [`kernels/nvfp4_norm_pack.cu`](../src/qwen_infra/kernels/nvfp4_norm_pack.cu) | Residual, Gemma RMSNorm and NVFP4 packing |
| [`kernels/nvfp4_attention_front.cu`](../src/qwen_infra/kernels/nvfp4_attention_front.cu) | QK norm, RoPE and KV append |
| [`kernels/nvfp4_attention_fusion.py`](../src/qwen_infra/kernels/nvfp4_attention_fusion.py) | Attention boundary adapters and output fusion |

See [architecture](architecture.md#precision) for precision, [testing](validation.md)
for checks and [references](references.md) for upstream sources.
