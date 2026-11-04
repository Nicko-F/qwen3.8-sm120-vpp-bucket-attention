# Contributing

Install development dependencies in a private virtual environment, then run the
tests and linter described in `pyproject.toml`. Keep changes focused: native
vLLM remains the reference, VPP owns prefill placement/transport, and bucket
attention owns decode work assignment and its graph lifecycle.

For a performance change, include the supported shape, precision, graph mode,
complete timing scope, warmup protocol and matched before/after measurements.
Report regressions as well as gains. A kernel measurement is not a serving
throughput claim. Numerical/quantization tolerances are appropriate; bitwise or
identical sampled-token requirements are not the project's performance goal.

Paths, models, devices, caches and deployment policies belong in configuration.
Do not require a private checkout, historical run, exact source hash or the
author's environment. Keep actual tensor/layout and resource bounds explicit,
and retain native fallback for unsupported inputs. New high-performance kernels
use CUDA C++/PTX or CUTLASS/CuTe; upstream/native implementations are preserved.

Preserve third-party copyright/license notices and state adaptations. Do not
commit model weights, datasets, credentials, environment archives, build caches
or proprietary reference artifacts. Use Git history rather than version-copy
directories. Issues and pull requests should include a minimal command and
relevant package/hardware information without private machine logs.
