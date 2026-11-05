# Testing and validation

Use the CPU checks for package changes, the model tests for runtime integration,
and the transport checks for VPP ownership and buffer reuse. Supported models
and execution settings are listed in
[Supported configurations](installation.md#supported-configurations).

## CPU and package checks

Install the development tools in the active environment and run the same
checks as the Linux CI job:

```bash
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m pytest -q -m 'not gpu'
python -m build
python scripts/check_wheel.py dist/*.whl
```

The suite covers configuration and fallback decisions, task planning, graph
ownership, package import closure, runtime-path discovery, bootstrap forwarding,
and VPP slot/control behavior. Linux bootstrap fixtures exercise installer
commands without downloading the runtime. C++ ownership tests run when a host
compiler is available; checks needing optional dependencies are skipped when
those dependencies are absent.

To run the C++ slot and generation test directly:

```bash
c++ -std=c++17 -Isrc/qwen_infra/vpp/native \
  tests/slot_pool.cpp src/qwen_infra/vpp/native/interface.cpp \
  src/qwen_infra/vpp/native/slot_pool.cpp -o /tmp/qwen-slots
/tmp/qwen-slots
```

The wheel check verifies that every CUDA/C++ JIT source asset is packaged.
The [Linux CI workflow](../.github/workflows/ci.yml) runs lint, CPU tests,
wheel construction, source-asset verification and the standalone C++ test.

## Inspect an installation

```bash
qwen-infra doctor
qwen-infra doctor --cuda
qwen-infra serve --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-bucket --tensor-parallel-size 4 --dry-run
qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile vpp --requests 4 --prompt-tokens 8192 \
  --chunk-size 2048 --max-model-len 8193 --dry-run
```

The plain doctor and launch previews run without loading a model.
`doctor --cuda` initializes CUDA and reports visible devices.

## Decode model tests

Expose exactly four compatible GPUs and set a model identifier or local path:

```bash
QWEN_INFRA_TEST_MODEL=nvidia/Qwen3.8-27B-NVFP4 \
  python -m pytest -m gpu -s tests/test_gpu_model.py
```

The test runs `native`, `decode-bucket` and `decode-fused` in sequence.
Each profile submits eight short requests and checks completion. Optimized
profiles also query the worker's `qwen_infra_status` RPC to check bucket
installation, fallback status, the captured eight-row aligned graph and the
selected route. The model environment variable opts into these GPU tests.

For applications using vLLM's Python `LLM` API, inspect the same metadata with:

```python
status = llm.collective_rpc("qwen_infra_status", timeout=60)
```

The response includes per-rank eligible layer counts, captured row/route pairs,
scratch bytes, native-graph preservation and fusion status.

## VPP transport and model checks

Run the four-GPU transport check before a model comparison:

```bash
python benchmarks/vpp_resources.py \
  --mode module --capacity 8192 --layout rank
```

The default workload sends 20 messages over each of six boundaries. It varies
valid rows, delays plane publication, wraps logical slots, verifies payload
identity and rejects retired tickets. It also checks the two-source/three-receive
slot budget and confirms zero live ownership after transport close.
`--messages` accepts 8–64.

For a complete prefill path, run the CPP and VPP examples in the
[installation guide](installation.md#run-prefill). Each command returns an array
of request records; check `finished=true` and one output token per request.

The focused model check adds warmup, repeated bank timing and resource receipts:

```bash
python benchmarks/vpp_resources.py --model nvidia/Qwen3.8-27B-NVFP4 \
  --capacity 2048 --layout rank --warmup 1 --repeats 3
```

Its final JSON includes completed banks, per-rank resource data and the
transport-close result. See [benchmarking](../benchmarks/README.md) for matched
edge/rank comparisons and longer contexts.

## Completed checks

| Area | Recorded coverage |
| --- | --- |
| CPU/package | Configuration, imports, planning, bootstrap fixtures, C++ ownership, lint and wheel source assets; [public Linux CI](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/runs/41867204118). |
| Installed decode/fusion | 42 completed requests across banks of 1, 9 and 32; eight output tokens per request; all four ranks installed bucket attention and fusions, captured both routes at all five supported row sizes, and preserved native graphs. |
| Installed VPP | Four 8K-token P+1 requests completed through the installed package and segmented transport. |
| Rank-shared transport | 120 boundary receives with payload checks, slot reuse, retired-ticket rejection and fully drained ownership. |
| Rank-shared model/resource | Four-request P+1 banks at 8K, 32K, 64K and 128K input lengths; matched edge/rank layouts and drained transport receipts. |

The installed-model checks used the tested runtime dependencies and model
weights. Some tested vLLM runs emitted worker or EngineCore termination warnings
during shutdown after requests completed. When investigating shutdown, retain
the native logs and check transport closure separately from inference results.

Timing tables and measurement settings are in [Results](results.md).
The [VPP resource design](vpp-resources.md) contains transport occupancy,
allocation and long-context measurements.
