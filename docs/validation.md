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
| Public Linux CI | [Passed for `085d4e6`](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/runs/41860375203). |
| Linux wheel with reused runtime | Package import closure passed from the installed virtual environment. |
| Installed Decode/fusion model | Initial bounded run reached the native dependency compilation timeout; a prebuilt-kernel retry is being prepared. |
| Installed VPP model | Pending the Decode stage. |

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
still building: `gemm_sm120` completed in about12 minutes, `gemm` in about44
seconds, and `fp4_gemm_cutlass_sm120` remained in compilation. There were no
request-bank outputs, Python tracebacks or optimized-route receipts. Its owned
session was cleaned. The next step uses the official matching SM120 prebuilt
kernel provider, preserving this cache and the failed run rather than changing
attention mathematics or restarting the same cold build from scratch.

GPU checks use four selected idle GPUs in one NUMA domain, bounded threads,
unique logs, a finite timeout and an external128 GiB available-memory floor.
Only the validation's owned process session may be stopped. Those shared-server
rules are not startup requirements imposed on other users by the library.
