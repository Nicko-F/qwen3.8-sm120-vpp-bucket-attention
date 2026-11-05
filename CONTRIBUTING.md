# Contributing

Contributions are welcome for kernel performance, runtime compatibility,
correctness, tests and documentation. For a bug report, include a minimal
command, expected and observed behavior, `qwen-infra doctor --cuda` output,
and the relevant error or build log. For a performance change, describe the
workload and comparison settings.

## Development setup

Use a Python virtual environment and install the development tools:

```bash
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m pytest -q -m 'not gpu'
python -m build
python scripts/check_wheel.py dist/*.whl
```

The [installation guide](docs/installation.md) covers GPU runtime and toolkit
setup. [Testing and validation](docs/validation.md) describes model integration,
C++ ownership and VPP transport checks.

## Implementation

Keep each change focused on its component. Native vLLM owns request scheduling,
model execution and KV storage; VPP extends prefill placement and transport;
bucket attention extends decode work assignment and graph replay.

Write reusable interfaces with configurable model, device, cache and dependency
locations. State supported tensor layouts, precision and resource bounds in
code and update the [support table](docs/installation.md#supported-configurations)
when they change. Preserve native fallback for unsupported configurations.

Use CUTLASS/CuTe DSL or explicit CUDA C++/PTX for new high-performance kernels.
Budget live values, registers, shared memory and occupancy before adding fusion.
Review generated code and resource reports when a change introduces unexpected
spills, serialization or data movement. Preserve third-party copyright and
license notices and document adaptations.

## Performance changes

Start with the affected module's correctness and timing checks, then validate
a model integration case. Run broader workload comparisons when a complete
optimization is ready. See [Benchmarking](benchmarks/README.md) for commands and
the measurement settings to record.

Include supported shapes, numerical tolerances, precision, graph mode, warmup,
timing scope, matched before/after samples and memory usage in the pull request.
Use comparisons at the level of the claim: operator timing for a kernel change,
complete-bank or HTTP measurements for model/serving performance. Report gains
and regressions together.

## Pull requests

Explain the concrete problem and resulting behavior, list the checks performed,
and link relevant measurements. Record the source revision for benchmark data
and keep result samples traceable to their workload and configuration.

Keep model weights, datasets, credentials, environment archives and build
caches outside Git. Use Git history for iteration and keep documentation links
and command examples current.
