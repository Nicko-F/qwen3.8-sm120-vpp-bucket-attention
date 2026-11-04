# Benchmarking

[Results](../docs/results.md) defines the scope and limitations of the curated
CSV tables here. They are completed historical measurements, not outputs of the
new package's installation smoke. Raw machine logs, weights and environment
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
