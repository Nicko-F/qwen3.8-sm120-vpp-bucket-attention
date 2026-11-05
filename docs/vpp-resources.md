# Rank-shared VPP communication resources

## Decision and fixed contracts

Keep the eight virtual stages, TP1/PP4 folded layer layout, W=7 admission,
7/5/3/1 warmup, Forward/Return FIFO order, native model mathematics and graphs.
This change reorganizes communication resources, not pipeline scheduling.

The starting implementation at `f2bea4e` allocates a seven-slot device pool per
edge endpoint (`native/stages.py`). Middle ranks own four pools. One source
stream is already shared, but incoming edges create separate receive streams.
`receive()` only submits H2D when the stage requests its input; posting a Gloo
header in `admit()` is not payload prefetch.

Source facts come from `native/{stages.py,control.py,device_pool.cpp,dma_edge.cpp}`
and `layout.py`. Existing segmented DMA uses one bounded C++ progress thread per
rank, eight 16 MiB host segments per edge, generation-specific CUDA IPC events,
and READY/ACK publication. Reuse this mechanism and its completion discipline.

## Dataflow and resource budget

- Use the existing serialized compute stream, one shared D2H send stream, and
  one shared H2D receive stream per rank. Do not allocate unused per-pool copy
  streams on this path. Native framework auxiliary streams are outside this
  transport budget.
- Allocate one rank-level source arena with two physical slots and one receive
  arena with three physical slots. Retain separate per-edge sequence, batch,
  generation and control metadata. Logical wire slots need not equal physical
  device slots; never use a physical slot number as another edge's wire identity.
  Allocate both arenas once at transport initialization. The hot path borrows
  fixed-address views; it does not grow, resize or cudaMalloc/free communication
  buffers per chunk. Dynamic ownership is distinct from dynamic allocation.
- Logical endpoint pools may remain lightweight views over those arenas. They
  must not each allocate payload storage. A physical lease is returned only
  after its own D2H completion (source) or final consumer event (destination).
  All endpoints sharing an arena must be progressed during credit waits.
- Preserve direct hidden/residual output publication and its current-generation
  plane events. Do not introduce a staging clone to achieve memory savings.
- Prefetch input across both directions using admitted FIFO headers and upcoming
  local-stage priority. At most two unconsumed speculative receives may occupy
  the three-slot arena; preserve room for the demanded input after the previous
  consumer finishes. Never block on an unavailable future header or enqueue a
  receive-stream wait for a generation whose record has not been published.
  Do not rely on Gloo `Work.is_completed()` for receive readiness: receive Work
  implementations may finalize only in `wait()`. Instead, under the existing
  DMA progress mutex, acquire the next segment's READY publication and inspect
  its stable 96-byte wire descriptor. Validate it against that edge's admitted
  FIFO before reserving a physical receive slot. Peeking must not advance ACK,
  segment or sequence. Keep the original Gloo Work/header buffer owned; demand
  consumption waits for and cross-checks that header before exposing the input.
  Limit speculation to one unconsumed receive per incoming edge (two total on
  middle ranks). No additional header thread or CUDA synchronization is needed.
- Keep the existing host ring: 8 x 16 MiB per edge, not a full chunk per logical
  in-flight entry. This change does not merge the six independent host/control
  channels or change their ACK/event-reuse rules.

For BF16 hidden and residual of width 5120, one slot is
`capacity * 5120 * 2 * 2` bytes. At C8192 that is 160 MiB. Middle-rank device
payload changes from `4 * 7 * 160 = 4480 MiB` to `5 * 160 = 800 MiB`, saving
3680 MiB (3.59375 GiB). At C2048 it changes from 1120 to 200 MiB. These are
communication payload allocations, not total model VRAM or host-ring storage.
Edge ranks previously had two endpoint pools; report their savings separately.

The complete group's latency can first be assessed after shared allocation,
shared streams and cross-direction prefetch all work together. A pool-only
intermediate implementation is not a completed performance candidate.

## Edit and reuse boundary

Change VPP transport/storage/control, its worker hooks, configuration, focused
tests and documentation. Keep native CPP, Decode/bucket/fusion, model layers,
sampling, parallel layout and existing performance tables unchanged. Retain
the edge-owned resource layout as an explicit matched diagnostic control;
default the completed segmented-DMA path to rank-shared resources. No paths,
physical device IDs, run names or source hashes belong in library contracts.

## Focused evidence, without a new full benchmark

1. CPU ownership/control tests and one bounded GPU transport test: both
   directions, more than seven messages, slot reuse, varying valid rows and
   delayed readiness. Check payload identity, generation rejection, no live
   leases at close, peak arena occupancy and credit-stall counters.
2. Installed model anchor using the same four selected idle GPUs in one NUMA
   domain. Compare edge-owned and rank-shared layouts at C2048 with 4 x 8K
   prompts, and C8192 with 4 x 32K prompts. These each exercise 16 chunks;
   retain W=7 and the same model/runtime/tactic settings. Fix native KV allocation
   to 6 GiB/rank and max sequences to eight on both layouts so freed communication
   memory does not merely enlarge native KV allocation. Reuse one loaded engine
   per layout/capacity; one warm bank and three timed banks are sufficient.
3. Record actual unique device-payload allocation, total peak VRAM, arena
   peak-live, per-direction prefetch/issue counters, source/destination credit
   waits and complete-bank latency. Native CUDA-event and transport receipts
   establish ownership and prefetch; profiler/full curves are not required.
4. Compare medians and ranges. Do not describe tiny noisy differences as a gain,
   or attribute a regression to the original algorithm without diagnosis. Token
   equality is not a gate for this resource refactor. Complete every request and
   inspect transport correctness and teardown.

All remote work stays below the private workspace, with bounded threads/build
parallelism, finite timeout/heartbeat, >=128 GiB MemAvailable and task-owned
cleanup. Only the selected four cards must be idle; preserve one NUMA domain.
Record revisions as provenance, not runtime permission gates. Commit code and
results in this repository; preserve the old baseline through Git.

## Measured transport check

The C8192 rank-shared transport check completed on four NUMA-local RTX 5090s.
It sent 20 messages over each of six edges (120 boundary receives), with valid
row counts 0, 1, 17, 4096 and 8192, delayed plane publication and logical-slot
wraparound. Each rank rejected reuse of a retired ticket. The library source
was `8161a2d`; the completed test script was `8e68e9b`.

| Rank | Physical payload | Source peak | Receive peak | Forward / return prefetched and consumed | Blocking source credit waits |
| --- | --- | --- | --- | --- | --- |
| 0 | 800 MiB | 2 | 2 | 0 / 2 | 2 |
| 1 | 800 MiB | 1 | 3 | 18 / 12 | 0 |
| 2 | 800 MiB | 2 | 3 | 14 / 18 | 1 |
| 3 | 800 MiB | 1 | 2 | 9 / 0 | 0 |

All ranks used one compute, one D2H and one H2D transport stream. Speculative
receive occupancy peaked at two on middle ranks. No destination credit wait
occurred. The third receive slot was actually used; reducing its capacity to
two is not justified by this check. Stress credit waits are not model latency.
After bounded collective close, every arena, endpoint, producer, input lease and
prefetch queue had zero live ownership. The process exited 0 and its owned
session was empty after cleanup.

Two initial fixture failures were retained: checking zero ownership immediately
after a single device synchronization was too early for a background DMA queue;
the retired-ticket probe also needed to recognize pybind's `ValueError` mapping
for `std::invalid_argument`. The completed check verifies ownership after the
existing transport drain and accepts only the intended stale-ticket rejection.
Neither fix changed the library, model or stage order.

Evidence run: `public_vpp-resources_20261104T172246_927812Z`. Raw machine logs
stay in the private run archive; this document records the portable summary.

## Matched model checks

Four configuration points completed, each with one warm bank and three timed
banks. These are focused resource checks, not a rerun of the historical full
benchmark. Both layouts used the same ordered four NUMA-local RTX 5090s, the
same model/runtime and default native tactics, a fixed 6 GiB KV allocation per
rank, eight maximum sequences and disabled prefix caching. Actual KV allocation
was identical: 502 blocks with block size 2784. Each bank contained four P+1
requests and 16 input chunks in total.

| Capacity | Prompt tokens/request | Layout | Median bank (s) | Timed range (s) |
| --- | --- | --- | --- | --- |
| 2048 | 8192 | Edge-owned | 1.450957 | 1.447842–1.458284 |
| 2048 | 8192 | Rank-shared | 0.883493 | 0.881769–1.442930 |
| 8192 | 32768 | Edge-owned | 7.028552 | 7.018437–7.032519 |
| 8192 | 32768 | Rank-shared | 4.564127 | 4.557483–7.006229 |

The observed median reductions are 39.1% and 35.1%. Both shared-layout samples
include a much slower bank near the control's latency; those samples remain in
the [CSV](../benchmarks/vpp_resources.csv). The cause of this variability was not
isolated. These measurements show no end-to-end loss in the two checked cases;
they do not establish a stable speedup across loads or context lengths. Any gain
belongs to the combined resource/stream/prefetch change, not buffer reduction
alone. Model loading, compilation and the warm bank are outside bank timing.

| Capacity | Middle-rank edge payload | Middle-rank shared payload | Saved/rank | Middle-rank peak allocated: edge / shared |
| --- | --- | --- | --- | --- |
| 2048 | 1120 MiB | 200 MiB | 0.898438 GiB | 15.869829 / 14.971391 GiB |
| 8192 | 4480 MiB | 800 MiB | 3.593750 GiB | 22.167905 / 18.574155 GiB |

The allocated-peak columns use `torch.cuda.max_memory_allocated` over the worker
lifecycle, including model loading/profile/warmup. They are not just buffer
capacity or allocator-reserved memory. The per-rank allocation differences
exactly match the payload-budget differences with KV held fixed. At C8192,
outer ranks save 1.40625 GiB each; all four ranks together save 10 GiB of fixed
communication payload. Middle-rank payload falls by 82.1%. Normal deployments
may use the released headroom to enlarge native KV, so total VRAM need not fall
by the same amount when its allocation remains automatic.

All six source edges reported 68 direct publications and zero staged
publications in every configuration. The endpoint sequences and per-rank
receive counts also matched. No output-staging copy was added to obtain the
memory reduction. Every run completed its banks, closed the transport with zero
live ownership, exited 0 and released its owned process session.

Native vLLM teardown still emitted forced-EngineCore-termination warnings; the
C2048 shared run additionally reported a forced worker termination and a shared
memory cleanup warning. These logs are retained. Transport ownership drained
successfully, but fully graceful native-engine shutdown is not claimed.

### Occupancy and backpressure

The shared model path reached source occupancy two and receive occupancy three.
Both C2048 layouts and the C8192 edge-owned control had zero blocking slot-credit
waits. C8192 rank-shared had the following cumulative waits:

| Rank | Source waits | Source wait (ms) | Receive waits | Receive wait (ms) |
| --- | --- | --- | --- | --- |
| 0 | 17 | 1351.320 | 0 | 0 |
| 1 | 27 | 1216.830 | 3 | 62.374 |
| 2 | 29 | 2221.144 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |

These counters include native warmup, the warm bank and all three timed banks.
They measure host credit waiting; source credit also remains owned while its
producer is computing. Waits on different ranks can overlap compute and one
another. Their sum is not additional bank latency or isolated D2H time.

The C8192 edge-owned model actually reached source occupancy five on rank 0
and receive occupancy three on several edges. Earlier observations of one or
two slots were therefore not a universal bound. The two-source-slot design
limits host enqueueing with backpressure; the current bank results support
keeping it, without claiming that two slots can never limit another workload.

Reproduce these points with `benchmarks/vpp_resources.py`; run IDs and all
individual timing samples are in the CSV. The bounded checks are complete;
no full curves or additional tuning sweep were run.

## Long-context follow-up plan

Extend only the focused resource comparison to four 65536-token and four
131072-token P+1 requests, holding capacity at 8192. Compare edge-owned and
rank-shared layouts at each length: four configuration points, one warm bank
and three timed banks per point. Keep the same ordered NUMA-local four GPUs,
6 GiB KV/rank, maximum eight sequences, default native tactics, model precision,
graphs, W=7 and stage order. The checkpoint declares 262144 maximum text
positions, so input plus one output fits without a context-limit override.

The previous cache supplied 502 blocks with a 2784-token full-attention block
size: four 128K inputs need 192 full-attention blocks, plus recurrent-group and
rounding overhead. The native prior run reported 1,096,669 tokens of cache
capacity; verify the new lengths' native capacity and actual allocation
receipts rather than ignoring the hybrid cache groups. Physical communication buffers remain 800 MiB/rank at C8192
regardless of context length; only the number of traversing chunks grows (32
and 64 per bank). Longer-history full-attention compute grows while the bytes
per chunk stay fixed. The relative benefit and backpressure therefore require
measurement; neither monotonically increasing gains nor zero credit waits are
assumed.

Reuse `benchmarks/vpp_resources.py`, adding an optional prompt-token argument
whose default preserves the prior workloads. Do not change the installed
library, allocator, transport, pipeline or math. Use the existing private
supervisor and serial four-card launches with the 128 GiB available-memory
floor, bounded threads, finite timeout and owned-session cleanup. Record all
bank samples, exact input lengths, allocator peaks, credit/prefetch counts,
matched output publication counts and drained close receipts. Preserve the
earlier observations and append these longer points separately.

## Long-context measurements

The 64K pair completed with the same library implementation as the earlier
checks. Only the benchmark's optional input-length parameter changed
(`d359b15`). These points retain C8192, four requests, one warm bank, three
timed banks, ordered NUMA-local GPUs and 6 GiB KV/rank.

| Input/request | Edge median (s) | Shared median (s) | Shared latency change | Edge range (s) | Shared range (s) |
| --- | --- | --- | --- | --- | --- |
| 65536 | 11.476845 | 11.490873 | +0.12% | 11.433025–11.496515 | 11.454258–11.508391 |

The 64K distributions overlap. This small difference is consistent with a tie;
the short-bank median gains do not extend to this longer bank. Communication
payload remains 4480 versus 800 MiB on middle ranks. Their allocated peaks are
22.167967 versus 18.574217 GiB, retaining the exact 3.59375 GiB saving.

Native cache capacity was 1,218,502 tokens, with 18.59x capacity concurrency at
65537 configured tokens and group sizes `(65537,65537,65537,2784)`. Both layouts
had 502 cache blocks and no observed preemption messages. All source edges
reported 132 direct publications, zero staged publications and sequence 132;
all banks completed and transport ownership drained to zero.

The shared 64K check accumulated 281 blocking credit waits (18,913.872 ms summed
across ranks and setup/warm/timed phases), all on the source side; the control
had none. Host enqueue backpressure therefore increased while bank latency
stayed effectively unchanged. This is direct evidence that cumulative host
credit-wait duration must not be counted as extra end-to-end delay. Shared
source occupancy remained two and receive occupancy peaked at three.

The 128K pair is the remaining follow-up; its samples will be appended to the
same CSV without replacing the earlier measurements.
