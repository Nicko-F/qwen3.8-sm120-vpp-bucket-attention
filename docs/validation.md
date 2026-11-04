# Package validation

This record validates extraction and installation separately from the historical
[performance results](results.md). No new performance gain is claimed here.

| Check | Outcome |
| --- | --- |
| Local CPU suite | 30 passed; three opt-in GPU cases and two Linux-only cases skipped on Windows. |
| Ruff | Passed. |
| C++ slot ownership/generation | Passed. |
| Wheel source assets | All 19 CUDA/C++ JIT assets included. |
| Isolated lightweight wheel install | Doctor and native/decode/VPP launch previews passed without GPU dependencies. |
| Public Linux CI | [Passed for `48dfad0`](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/runs/41867204118), including bootstrap fixtures. |
| Linux wheel with reused runtime | Package import closure passed from the installed virtual environment. |
| Installed Decode/fusion model | Passed: 42/42 requests, eight output tokens each, exit 0. |
| Installed VPP model | Passed: four 8K-token requests, one output token each, exit 0. |

The Linux runtime uses a new private virtual environment with read-only access
to the existing tested PyTorch/vLLM/FlashInfer dependencies and local model
weights. Package source and caches are separate from the research workspace.
This is not a fresh internet bootstrap or a new full performance matrix.

The first GPU startup was deliberately stopped after the validation wrapper
omitted the reused environment's Ninja executable from PATH. Its owned process
session was cleaned and the GPUs released. The wrapper's tool discovery was
corrected before retrying; no kernel arithmetic or package dependency was changed
to address this operational issue.

The next run reached its 1200-second limit while native FlashInfer kernels were
still building: `gemm_sm120` completed in about 12 minutes, `gemm` in about 44
seconds, and `fp4_gemm_cutlass_sm120` remained in compilation. There were no
request-bank outputs, Python tracebacks or optimized-route receipts. Its owned
session was cleaned. The completed Decode smoke then used the official matching
SM120 prebuilt kernel provider. Failed runs and the native cache were retained;
attention mathematics did not change.

GPU checks use four selected idle GPUs in one NUMA domain, bounded threads,
unique logs, a finite timeout and an external 128 GiB available-memory floor.
Only the validation's owned process session may be stopped. Those shared-server
rules are not startup requirements imposed on other users by the library.

## Decode model anchor

The installed kernel implementation is revision `085d4e6`. Later CLI preview,
installer and documentation changes do not alter those kernels. The official
providers installed successfully: `flashinfer-cubin` 0.7.1,
`flashinfer-jit-cache` 0.7.1+cu130, and `flashinfer-jit-cache-sm120f` 0.7.1+cu130.

The completed model smoke used banks with 1, 9 and 32 requests. All 42 requests
finished with 8 output tokens each (336 total). Each of four ranks reported 16
eligible FA layers, bucket attention installed, fusions enabled, all 10 extra
captures (rows 1/8/16/24/32, aligned/persistent), and preserved native graphs.
No worker reported a fallback reason. Per-rank scratch was 6,180,864 bytes.

Final route receipts were aligned. Persistent capture readiness is verified;
transient selection during earlier mixed-bank steps was not observed by the
final-status RPC and is not claimed. This smoke establishes installed-package
execution, not a new performance or semantic-quality result.

The model run exited 0 and its owned process session was empty after cleanup.
Native vLLM shutdown nevertheless reported forced termination of four workers
and semaphore/shared-memory cleanup warnings; it is not described as a fully
graceful shutdown.

## VPP model anchor

The same installed kernel revision completed TP1/PP4 folded VPP with segmented
DMA, a 2048-token chunk capacity, four 8192-token prompts and one generated token
per request. The installed package built its native C++ transport and bound
stage graphs; all four requests reported `finished=true`.

The CLI completed its mandatory all-rank transport-close RPC before requesting
native engine shutdown. The close implementation rejects live contexts or
leases and drains slots. Successful closure follows from that guarded RPC
returning; numeric lease/slot receipts were not printed and are not claimed as
directly observed counters.

The run exited 0, its owned process session was empty after cleanup, and the
selected GPUs were released. Native vLLM reported all four workers exited
gracefully, followed by forced termination of the remaining EngineCore process.
FlashInfer also selected default tactics for untuned shapes. These warnings are
retained; this smoke is not a throughput benchmark or a fully graceful-shutdown
claim. The optional NCCL VPP transport was not separately executed in this
package-validation run.
