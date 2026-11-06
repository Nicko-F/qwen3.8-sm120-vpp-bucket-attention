# Performance results

Measurements use four NUMA-local RTX 5090 GPUs, mixed NVFP4 model weights,
NVFP4 KV cache, BF16 hidden states/attention and FP32 softmax/recurrent state.
Each comparison uses matched device order and CPU placement. MTP and prefix
caching are disabled.

## Prefill: VPP versus CPP

The prefill sweep covers CPP chunk capacities 2K/4K/8K/16K, VPP capacities
2K/4K/8K, and two DCPP variants, with graph-off and piecewise-graph execution.
There are 20 configurations, 160 shape/configuration points, 480 timed banks
and 160 warmup banks.

Each bank processes four requests and generates one output token per request
(P+1). Timings include scheduling, communication, the LM head, sampling and
output. The table selects the best CPP and VPP configurations at each input
length, using the median of three timed banks.

| Input tokens/request | CPP, ms/bank | VPP, ms/bank | VPP configuration | Latency reduction |
| --- | ---: | ---: | --- | ---: |
| 8,192 | 832.413 | 795.197 | C2048, piecewise | 4.47% |
| 32,768 | 4,304.359 | 3,936.227 | C4096, piecewise | 8.55% |
| 65,536 | 11,647.508 | 10,834.357 | C4096, piecewise | 6.98% |
| 131,071 | 35,241.058 | 33,410.873 | C8192, graph off | 5.19% |

For a single 8K request, CPP C2048/piecewise takes 304.587 ms and the best VPP
configuration takes 310.973 ms. The four-request banks provide more pipeline
work to overlap.

Both paths use TP1/PP4, 8 GiB full-attention KV per rank and two CPU threads per
rank. Sampling uses temperature 0.7 and top-p 0.9. The 128K configuration reserves
one token for output, giving 131,071 input tokens. Native autotune choices are
matched within graph comparisons.

Data: [prefill.csv](../benchmarks/prefill.csv).

## Pipeline cost model

The [equal-stage model](architecture.md#context-growth-and-pipeline-bubbles)
separates adjacent-chunk cost growth from cumulative fill/drain overhead.
Replaying measured compute nodes with transfer and host costs set to zero gives:

| Input tokens/request | Modeled extra pipeline time, s | Share of modeled makespan |
| --- | ---: | ---: |
| 8,192 | 0.4268–0.4272 | 42.19–42.56% |
| 65,536 | 0.8729–0.8824 | 8.40–8.48% |
| 131,071 | 2.1989–2.2148 | 6.73–6.79% |

Here, extra pipeline time equals modeled makespan minus the busiest rank's
accumulated compute. This isolates dependency overhead in the compute model;
PCIe transfer and host submission are separate terms. The absolute overhead
grows with context while its share of total execution falls.

## Decode: full-model latency

Each request generates 256 tokens. Four warm banks precede three timed
repetitions. Measurements use common output windows within positions 17–240,
where every request advances one token. Values are medians of repetition
medians and include model execution, communication, the LM head and sampling.

| Initial context | Native ms/step | Bucket ms/step | Latency reduction |
| --- | ---: | ---: | ---: |
| B1, 512 | 10.0577 | 9.7397 | 3.16% |
| B32, all 8,064 | 17.9077 | 17.9361 | −0.16% |
| B9, 130,816 + 8 × 4,096 | 20.0353 | 14.3697 | 28.28% |
| B32, 130,816 + 31 × 4,096 | 28.6256 | 18.1637 | 36.55% |

The B9 workload uses a padded 16-row graph. The largest gains occur when one
long request dominates otherwise short-request work; uniform B32 is near parity.

Data: [decode.csv](../benchmarks/decode.csv).

## Decode: HTTP serving

The independent bucket comparison uses official `vllm serve` and
`vllm bench serve` over six finite request banks. Priming is excluded from the
timed results; two formal rounds reverse execution order. Native and bucket
each complete 540 requests with zero failures.

| Workload | Bucket output throughput versus native |
| --- | ---: |
| Mixed arrivals | +10.41–11.36% |
| Uniform | +0.03–0.07% |

Short C1 results vary in sign across the two rounds. Request-level TTFT and ITL
vary by cohort, so the benchmark records them alongside aggregate throughput.

## Kernel fusion

| Transformation | Timing scope | Latency reduction |
| --- | --- | ---: |
| Residual/RMSNorm → NVFP4 pack | Direct producer | About 33–39% |
| QK norm/RoPE/gate/KV append | Direct producer | About 52–55% |
| Attention output/gate/static FP8 | Complete XQA attention path | About 0.7–1.3% |
| All three | Full GPU decode call, B1/8/16/32, 1K/2K contexts | 2.04–7.18% |

The full-call comparison runs both arms on the same aligned bucket path,
including the LM head. Producer timings measure the local fusion opportunity;
the complete-call timings measure its effect across model execution.

## Three-profile HTTP comparison

The following comparison runs native, bucket and bucket-plus-fusion profiles
with identical request banks. Each profile completes 180 priming and 360 formal
requests. Throughput is the combined formal output-token count divided by the
sum of formal bank durations across two execution orders.

| Bank | Native tok/s | Bucket tok/s | Bucket + fusion tok/s | Fusion versus bucket |
| --- | ---: | ---: | ---: | ---: |
| Short C1 | 92.64 | 99.00 | 100.74 | +1.76% |
| Uniform32 | 332.85 | 332.53 | 327.68 | −1.46% |
| Skew16 | 248.03 | 255.70 | 252.99 | −1.06% |
| Skew32 | 226.56 | 243.97 | 241.84 | −0.87% |
| Mixed, 2 req/s | 448.19 | 494.11 | 495.77 | +0.33% |
| Mixed, 8 req/s | 452.48 | 496.57 | 497.74 | +0.24% |

All profiles use native prefill, graph/compile endpoints `[32, 2048]`, a
2048-token batch budget, up to 32 requests, 131072 context capacity and 8 GiB
KV per rank. Sampling uses temperature 0.7 and top-p 0.9. Each profile runs in
one fresh process, with two sequential formal orders and zero failed requests.

The per-cohort metrics explain the distinction between decode time and request
latency: in Skew32, adding fusion changes short-cohort mean TTFT from 14.851 to
26.196 s and mean E2E from 33.041 to 33.673 s. TPOT is derived as
`(E2E − TTFT) / (output_tokens − 1)` and is reported together with those times.

Data: [http_fusion.csv](../benchmarks/http_fusion.csv).

## Data and reproduction

[Benchmarking](../benchmarks/README.md) provides commands for native/bucket/fused
serving and CPP/VPP prefill. [VPP communication](vpp-resources.md) records the
shared-buffer memory budget and matched resource measurements.

The source revisions associated with the decode-window dataset are `808a055e`
and `d0eed042`; the three-profile HTTP dataset uses `f4f7ee41`, `c16dcbed` and
`5e3a6be1`. The tables above retain those datasets' configurations. Per-bank VPP
resource samples and run identifiers are in
[vpp_resources.csv](../benchmarks/vpp_resources.csv).
