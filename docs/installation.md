# Installation

## Bootstrap

The bootstrap script creates `.venv` in the checkout, installs the tested native
runtime and this package, then forwards the command after `--` to `qwen-infra`:

```bash
bash scripts/bootstrap.sh --with-toolkit -- serve --model /path/to/model \
  --profile decode-bucket --tensor-parallel-size 4
```

Use `--venv /your/private/environment` to select another environment location.
Use `--vllm-wheel URL` to test another official wheel. Dependency constraints are
an installation reference; runtime checks inspect APIs, device capabilities and
tensor layouts, rather than requiring an exact Git revision.

After installation, use `.venv/bin/qwen-infra` directly. Model download/loading
is delegated to native vLLM/Hugging Face. A local model directory avoids another
download. No model weights are bundled.

## Existing environment

To reuse an environment containing compatible PyTorch, vLLM and FlashInfer:

```bash
python -m pip install -e . --no-deps
qwen-infra doctor
qwen-infra serve --model /path/to/model --profile native
```

The Python package itself is lightweight. CUDA/C++ extensions build lazily only
when the corresponding optimization is selected. The original vLLM wheel is
used; there is no mandatory full vLLM source rebuild.

The tested runtime is vLLM `0.32.0rc1.dev196+ga9fe30e9f`, PyTorch `2.14.0`,
FlashInfer `0.7.1`, Transformers `5.19.0`, and CUTLASS DSL `4.8.0` with CUDA 13.
These adapters use internal vLLM interfaces, so newer versions need compatibility
testing even if installation succeeds.

## Support

| Path | Current implementation contract |
| --- | --- |
| `native` serving | The installed vLLM's supported devices, models and parallelism. |
| `decode-bucket` | SM120; native V2 runner; BF16 Q/O; NVFP4 KV; D256; six local Q heads and one KV head; page32; one query token/request; bounded graph rows. |
| `decode-fused` | Bucket contract plus the implemented H5120 Gemma RMSNorm and static-FP8 output-projection consumer. |
| `prefill --profile cpp` | Native CPP comparison using the same configured TP1/PP4 prefill bank. |
| `prefill --profile vpp` | Implemented dense64-layer H5120 Qwen backbone, TP1/PP4, two folded traversals and one generated token per request. |

The current automatic Decode/fusion path is not the separate upstream
multi-head/multi-query development port. Unsupported Decode configurations
select native attention before capture when they can be identified safely.
Compilation errors are reported with their retained build log; they are not
silently described as successful acceleration.

VPP currently excludes prefix caching, asynchronous scheduling, LoRA and
speculative decoding. Its segmented-DMA backend is a same-host Linux transport.
The NCCL compatibility option uses native tensor transport and does not imply
the same performance as the segmented direct-boundary implementation.

Measured context coverage extends to 128K. Actual table/index arithmetic and
allocation limits remain checked; the code does not require a model's configured
maximum context to equal the author's historical 131072 setting.

## CUDA, NCCL and caches

Custom kernels require an installed CUDA toolkit with `nvcc`, CUDA 13 NVFP4
support, a working host C++ compiler and the relevant dependency headers.
`--with-toolkit` installs the matching compiler/runtime packages privately inside
the selected virtual environment. It does not modify the driver or host toolkit.
`CUDA_HOME` selects a toolkit. NCCL and FlashInfer headers/libraries are
discovered from the active environment; missing components are reported by the
selected extension rather than hidden behind an experiment directory check.

The default package cache follows `XDG_CACHE_HOME` or `~/.cache/qwen-infra`.
Set `QWEN_INFRA_CACHE_DIR` to choose another location. Native vLLM, Hugging Face
and PyTorch retain their standard cache controls. Build keys reflect source,
compiler and flags; they do not grant or deny permission to run the package.
Build parallelism respects `MAX_JOBS` and defaults to two jobs.

Use `CUDA_VISIBLE_DEVICES` for GPU selection. Four-GPU comparisons should use
the same ordered devices and preferably one NUMA domain. There is no hard-coded
physical GPU list, fixed SM count, required username, private model path or
global environment modification. GPU P2P policy remains a deployment choice.

## Memory planning

The supported VPP boundary contains two BF16 planes of width 5120. One seven-slot
pool requires `7 * 2 * token_capacity * 5120 * 2` bytes: 560 MiB at capacity4096.
An interior rank can own four pools, plus registered host transport buffers.
Model weights, native KV, recurrent state, graphs and build workspace are
additional allocations. Reduce chunk capacity or memory budgets for your GPU.

The research server used a 128 GiB minimum available-host-memory policy and
same-NUMA selected-idle-device admission. Those are shared-server operating
rules, not universal RAM/GPU-count requirements embedded in the library.

## Validation status

The research implementations completed the experiments in [results](results.md).
This newly extracted package is being checked separately for wheel contents,
configuration, imports and bounded model integration. Until the final smoke
record is added here, a fresh-package GPU deployment is not claimed as validated.
No new performance matrix is required merely to change packaging.
