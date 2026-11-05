# Benchmarking

This directory contains measured comparisons and a focused VPP resource
benchmark. [Results](../docs/results.md) describes the workloads, timing scopes
and performance tables; [VPP resources](../docs/vpp-resources.md) explains the
buffer design and resource measurements.

| File | Contents |
| --- | --- |
| [prefill.csv](prefill.csv) | Best measured CPP/VPP P+1 bank latency at four input lengths, with the selected VPP chunk and graph settings. |
| [decode.csv](decode.csv) | Full-model native/bucket decode step latency for short, uniform and skewed context banks. |
| [http_fusion.csv](http_fusion.csv) | Native, bucket and fused HTTP throughput across six workload banks. |
| [vpp_resources.csv](vpp_resources.csv) | Matched edge/rank layouts at 8K–128K inputs, including individual bank times, memory allocation and credit waits. |
| [vpp_resources.py](vpp_resources.py) | Four-GPU transport checks and repeated P+1 model/resource measurements. |

Install the tested runtime using the [installation guide](../docs/installation.md)
and check the [supported configurations](../docs/installation.md#supported-configurations)
before running a comparison.

## HTTP serving

Launch one serving profile with a common configuration for all three arms:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
COMMON=(
  --dtype bfloat16 --kv-cache-dtype nvfp4
  --attention-config '{"backend":"FLASHINFER"}'
  --max-num-seqs 32 --max-num-batched-tokens 2048
  --no-async-scheduling --no-enable-prefix-caching
  --max-model-len 131072 --kv-cache-memory-bytes 8589934592
  --compilation-config '{
    "mode": 3,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "compile_ranges_endpoints": [32, 2048],
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32]
  }'
)
qwen-infra serve --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile native --tensor-parallel-size 4 \
  --host 127.0.0.1 --port 8000 "${COMMON[@]}"
```

In a second terminal using the same environment, run the vLLM client:

```bash
vllm bench serve \
  --backend vllm --base-url http://127.0.0.1:8000 \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --max-concurrency 32 \
  --seed 123 --save-result
```

Repeat with `--profile decode-bucket` and `--profile decode-fused`, retaining
the same `COMMON` arguments. This fixes the KV allocation at 8 GiB per rank.
Keep the model, ordered devices, CPU/NUMA placement, graph settings, KV budget,
inputs and sampling fixed. Use separate priming requests before timing each
profile, and record the native vLLM command printed at startup.

The random dataset above provides a uniform-length starting point. To evaluate
bucket scheduling, also include an explicit mixed-length dataset and retain its
length distribution: the amount of work per request is the feature being
tested. Report generated tokens/s, TTFT, inter-token latency or TPOT, request
completion latency and request failures together. Include both cohort-level
and overall results for mixed banks.

The client options follow the
[vLLM benchmarking CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/).
Tencent's [HPC-Ops benchmarks](https://github.com/Tencent/hpc-ops/tree/main/benchmark)
provide additional examples of attention length distributions.

## Prefill P+1

The package CLI runs the same input bank through native CPP or folded VPP:

```bash
qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile cpp --requests 4 --prompt-tokens 8192 \
  --chunk-size 2048 --max-model-len 8193

qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile vpp --requests 4 --prompt-tokens 8192 \
  --chunk-size 2048 --max-model-len 8193
```

Both commands use TP1/PP4, BF16 activations, NVFP4 KV, piecewise graphs and one
generated token per request. Measure complete request banks, keeping model
loading and compilation outside steady-runtime timing. The resource script
below provides warmup and timed repetitions for VPP layout comparisons.

For a chunk-size comparison, match capacities and graph settings between
profiles. For a tuning envelope, report every configuration considered and the
best result from each profile. The prefill table in [Results](../docs/results.md)
uses the latter scope. Its one-token output defines P+1 bank latency; HTTP
long-decode throughput is measured separately.

## VPP transport and resource comparison

Use four visible GPUs in one NUMA domain. The transport check exercises buffer
ownership, variable valid rows and slot reuse without loading model weights:

```bash
python benchmarks/vpp_resources.py \
  --mode module --capacity 8192 --layout rank
```

For model/resource measurements, run each configuration in a separate process:

```bash
python benchmarks/vpp_resources.py --model nvidia/Qwen3.8-27B-NVFP4 \
  --capacity 2048 --layout edge --warmup 1 --repeats 3
python benchmarks/vpp_resources.py --model nvidia/Qwen3.8-27B-NVFP4 \
  --capacity 2048 --layout rank --warmup 1 --repeats 3
```

Repeat with `--capacity 8192`. The script accepts capacities 2048 and 8192;
by default, each request contains four times that many input tokens: 8K or 32K.
Every bank contains four requests and one output token per request. Each process
loads one engine, runs one warm bank, and times three banks.

Both layouts use a fixed 6 GiB native KV budget per rank and eight maximum
sequences. This keeps communication-buffer savings visible rather than
automatically allocating the saved capacity to a larger KV cache.

For longer inputs, set the input length while holding capacity at 8192:

```bash
python benchmarks/vpp_resources.py --model nvidia/Qwen3.8-27B-NVFP4 \
  --capacity 8192 --prompt-tokens 65536 --layout edge --warmup 1 --repeats 3
python benchmarks/vpp_resources.py --model nvidia/Qwen3.8-27B-NVFP4 \
  --capacity 8192 --prompt-tokens 65536 --layout rank --warmup 1 --repeats 3
```

Use `--prompt-tokens 131072` for the 128K pair. The script configures the engine
context bound as input length plus one output token. Check the model's position
range and native hybrid-KV capacity before choosing longer inputs.

Each bank emits a JSON progress record. The final JSON includes:

- All warm and timed bank durations, their timed median and range.
- Unique communication payload, CUDA allocation and allocator peaks.
- Arena occupancy, per-direction prefetch and credit-wait counters.
- The per-rank transport-close receipts.

Retain the individual samples alongside summary statistics. Credit-wait
counters accumulate host waiting across ranks and phases; use complete-bank
latency to assess their effect on throughput.

## Reporting a comparison

Record the Git revision, package versions, GPU model and ordered placement,
CPU binding, model identifier/revision, precision, context distribution,
graph settings, memory budgets and exact commands. State the warmup count,
repetition count and timed scope. Include memory savings and regressions
alongside latency or throughput gains, and preserve the underlying samples.
