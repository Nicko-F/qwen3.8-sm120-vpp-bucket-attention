# Benchmarking

[Results](../docs/results.md) defines the scope and limitations of the curated
original CSV tables here. They are completed historical measurements, not outputs
of the new package's installation smoke. The focused VPP resource check below
records the subsequent buffer refactor separately. Raw machine logs, weights and environment
archives are not required to run this project.

For a simple new serving comparison, launch one profile at a time, then use the
official vLLM client in the same environment:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --max-concurrency 32 \
  --seed 123 --save-result
```

Compare `native`, `decode-bucket`, and `decode-fused` with identical model,
devices, graph settings, KV budget, inputs and sampling. Prime separately before
formal measurement. Report throughput together with TTFT, TPOT and request-level
tails; do not infer a kernel speedup from TPOT alone. To study uneven KV lengths,
use an explicit mixed-length dataset and retain its distribution. A uniform
random-input bank is not a replacement for the skewed benchmark.

For static prefill, run the package's `prefill` entry with CPP and VPP, matching
chunk capacity, graph mode and complete request bank. Keep same-capacity
comparisons separate from the best measured configuration envelope. P+1 bank
latency is not long-decode serving throughput.

The benchmark interface follows [vLLM's official
CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/). Tencent's
[HPC-Ops benchmarks](https://github.com/Tencent/hpc-ops/tree/main/benchmark)
are the reference for length-distribution comparisons; the current attention
implementation uses its own NVFP4 layout and supported dimensions.

## Focused VPP resource check

Use the installed package and four visible GPUs in one NUMA domain. Run the
bounded transport check before loading a model:

```bash
python benchmarks/vpp_resources.py --mode module --capacity 8192 --layout rank
python benchmarks/vpp_resources.py --model /path/to/model \
  --capacity 2048 --layout rank --warmup 1 --repeats 3
```

Compare `--layout edge` and `--layout rank` at the same capacity. Capacity 2048
uses four 8K prompts; capacity 8192 uses four 32K prompts. Each process loads one
engine, primes one bank and times three complete P+1 banks. Both layouts use
6 GiB native KV per rank and eight maximum sequences. JSON output includes
bank times, unique communication-payload bytes, arena occupancy, prefetch,
credit waits, CUDA memory and the drained close receipt. Capacity and layout
choices in this script define these two focused workloads, not universal
library shape requirements. See [resource design and results](../docs/vpp-resources.md).

For a longer bank, set the per-request input length explicitly while keeping
chunk capacity and the other control settings matched:

```bash
python benchmarks/vpp_resources.py --model /path/to/model \
  --capacity 8192 --prompt-tokens 65536 --layout rank --warmup 1 --repeats 3
```

Repeat with `--layout edge` for the control; `--prompt-tokens 131072` selects the
128K follow-up. The engine context bound is input length plus one output token.
Check the checkpoint's native position range and reported hybrid-KV capacity;
the script does not override a model's position limit.
