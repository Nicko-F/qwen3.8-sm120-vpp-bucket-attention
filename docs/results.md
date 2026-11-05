# Results

The performance presentation reuses the completed benchmark datasets below.
The public implementation uses rank-shared communication buffers while keeping
the pipeline order. Allocation, credit and layout-validation records are in
the [resource report](vpp-resources.md); package smoke records are in
[installation](installation.md). The tables retain their original measurement
settings and provenance.

The shared experimental platform was four same-NUMA RTX 5090 GPUs, native mixed
NVFP4 weights and NVFP4 KV, BF16 hidden/attention math and FP32 softmax/state.
Controls and candidates used the same ordered devices and CPU placement. MTP
and prefix caching were disabled. Shapes, graph configuration and measurement
scope are specified below; gains at different levels are not additive.

## Prefill: VPP versus CPP

The static prefill matrix covered 20 fresh configurations and 160 shape/config
points: CPP capacities 2K/4K/8K/16K, VPP 2K/4K/8K, and two DCPP variants, each
with graph off or piecewise graphs. It included 480 formal and 160 warmup banks.
The table uses the median of three complete offline P+1 banks for four requests,
including scheduling, communication, LM head, sampling and output.

| Tokens/request | Best CPP, ms/bank | Best VPP, ms/bank | VPP choice | Reduction |
| --- | ---: | ---: | --- | ---: |
| 8,192 | 832.413 | 795.197 | C2048, piecewise | 4.47% |
| 32,768 | 4,304.359 | 3,936.227 | C4096, piecewise | 8.55% |
| 65,536 | 11,647.508 | 10,834.357 | C4096, piecewise | 6.98% |
| 131,071 | 35,241.058 | 33,410.873 | C8192, graph off | 5.19% |

This is an envelope over the declared configuration matrix, not a validated
per-request layout router. For B1/8192, CPP C2048/piecewise was faster:
304.587 ms versus 310.973 ms for the best VPP. The 47.74% graph improvement in
VPP C2048/B4 is a within-VPP comparison, not a gain over the best CPP.

For each profile, native autotune choices were matched before graph comparison.
No new user installation needs those historical files: its native runtime tunes
and caches normally. P used TP1/PP4, 8 GiB FA KV per rank and two CPU threads per
rank. Sampling was temperature 0.7/top-p 0.9; output length was one token.
"128K" reserves one position for that output and therefore means 131,071 inputs.

## Pipeline cost: model versus observed gaps

The [equal-stage model](architecture.md#prefill-folded-vpp) distinguishes the
approximately fixed adjacent-chunk cost increase from the accumulated fill/drain
term. Replaying measured compute nodes with communication/host/copy cost set to
zero gave the following B4 examples:

| Tokens/request | Modeled extra pipeline time, s | Fraction of modeled makespan |
| --- | ---: | ---: |
| 8,192 | 0.4268–0.4272 | 42.19–42.56% |
| 65,536 | 0.8729–0.8824 | 8.40–8.48% |
| 131,071 | 2.1989–2.2148 | 6.73–6.79% |

This extra time is modeled makespan minus the busiest rank's accumulated
compute. It is not directly measured PCIe time. Actual per-rank gaps include
dependency waits and host submission effects; subtracting this model from an
unprofiled wall time does not isolate communication. A declining relative
fraction can coexist with increasing absolute overhead.

## Decode: full-model common windows

The decode comparison generated 256 output tokens per request. Three formal
repetitions followed four warm banks. Timings use common output intervals where
every request advanced one token and remained within output positions 17–240;
the reported value is the median of repetition medians. It includes model work,
communication, LM head and sampling, rather than one attention kernel.

| Initial context | Native ms/step | Bucket ms/step | Reduction |
| --- | ---: | ---: | ---: |
| B1, 512 | 10.0577 | 9.7397 | 3.16% |
| B32, all 8,064 | 17.9077 | 17.9361 | −0.16% |
| B9, 130,816 + 8 × 4,096 | 20.0353 | 14.3697 | 28.28% |
| B32, 130,816 + 31 × 4,096 | 28.6256 | 18.1637 | 36.55% |

The B9 case used a padded 16-row graph. Uniform fallback retains a small measured
negative; native fallback does not prove zero integration overhead. This path
was opt-in and native remained the default service.

The independent HTTP comparison used official `vllm serve` and
`vllm bench serve`, six finite banks, excluded priming, and forward/reverse formal
orders. Native and bucket each completed 540 requests with zero failures.
Mixed-arrival throughput improved 10.41–11.36% across both rounds. Uniform
throughput changed +0.03–0.07%. Short C1 changed sign between rounds; some long
request TTFT/ITL metrics regressed. Finite-bank throughput is not steady-state
capacity, a production SLO, or a per-request guarantee.

## Fusion: operator and GPU-call measurements

Three optional fusions preserve native consumers and precision contracts:

| Transformation | Measured scope | Observation |
| --- | --- | --- |
| Residual/RMSNorm → NVFP4 pack | Direct producer | About 33–39% lower producer latency |
| QK norm/RoPE/gate/KV append | Direct producer | About 52–55% lower producer latency |
| Attention output/gate/static FP8 | Complete XQA tail | About 0.7–1.3% lower complete tail-window latency |
| All three | Controlled full GPU decode calls, B1/8/16/32, contexts 1K/2K | 2.04–7.18% lower latency versus the same bucket path |

The controlled calls include the LM head without modifying it. They use a matched
forced aligned bucket path to isolate fusion; they are not the automatically
routed HTTP service below. Individual percentages have different denominators
and cannot be summed.

An earlier uniform-B32 module combination regressed 12.17%. The later matched
same-bucket comparison did not reproduce that regression. Historical negatives
are retained; no measured causal attribution to a single instruction follows
from this change of comparison.

## Fusion: matched HTTP comparison, 2026-11-04

All three arms completed 540 requests each: 180 priming plus 360 formal requests.
The comparison contains 1,080 formal requests with no failures. The table divides
the combined formal output tokens by the sum of the two round bank durations.

| Bank | Native tok/s | Bucket tok/s | Bucket + fusion tok/s | Fusion vs bucket |
| --- | ---: | ---: | ---: | ---: |
| Short C1 | 92.64 | 99.00 | 100.74 | +1.76% |
| Uniform32 | 332.85 | 332.53 | 327.68 | −1.46% |
| Skew16 | 248.03 | 255.70 | 252.99 | −1.06% |
| Skew32 | 226.56 | 243.97 | 241.84 | −0.87% |
| Mixed, 2 req/s | 448.19 | 494.11 | 495.77 | +0.33% |
| Mixed, 8 req/s | 452.48 | 496.57 | 497.74 | +0.24% |

This comparison used the same native prefill, graph/compile endpoints
`[32, 2048]`, 2048-token budget, maximum 32 requests, 131072 context capacity,
8 GiB KV/rank, temperature 0.7/top-p 0.9 and identical input banks. It is a new
matched configuration for the fusion experiment; it does not replace the earlier
bucket baseline.

There is no general additional HTTP fusion win. Uniform throughput and mean
TTFT/TPOT slightly regress. Large apparent skew-short TPOT improvements coexist
with later TTFT: in Skew32 the short-cohort mean TTFT grows 14.851 → 26.196 s,
while E2E grows 33.041 → 33.673 s. A lower `(E2E-TTFT)/(output_tokens-1)` alone
does not show that the kernel accelerated. Formal route observers were off;
admission/overlap is an interpretation, not a traced root cause.

Each arm used one fresh process and two sequential formal orders. Small
percentages are descriptive, not statistically established causal effects. No
bitwise or identical-token requirement was imposed; semantic quality was not
evaluated by these performance experiments.

## Source and reproducibility boundaries

The research source revisions were `808a055e` / `d0eed042` for the native/bucket
decode windows, and `f4f7ee41` / `c16dcbed` / `5e3a6be1` for the three HTTP fusion
arms. The original research worktree was `f9f0a90e` at extraction. These are
provenance records, not startup requirements. The upstream fixed-mode FlashInfer
contribution has separate multi-shape measurements and is not substituted for
the automatic-dispatch results above.

Curated numeric tables are in [benchmarks](../benchmarks/README.md). Native CLI
examples there let users measure their own installation. They do not silently
reconstruct private original bank files or promise identical timings.
