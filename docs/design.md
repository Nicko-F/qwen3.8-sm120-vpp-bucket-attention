# Release design

This repository packages two independently usable inference optimizations:
folded virtual pipeline parallelism (VPP) for prefill, and load-balanced paged
attention for decode. CUDA fusion modules form a separate optional layer.
It is an extension of vLLM and FlashInfer, rather than a replacement serving
engine. Native vLLM remains available as a comparison and fallback.

## Interface and deployment

- Install a normal Python package in a private environment. Use a small command
  line entry point for setup, serving, prefill, diagnostics and examples.
- Accept a Hugging Face model identifier or local model directory, visible CUDA
  devices, parallelism, memory budget and output/cache locations as configuration.
- Keep the OpenAI-compatible API and native vLLM request handling. Do not add a
  second scheduler, experiment supervisor or HTTP framework for packaging.
- Provide a one-command bootstrap into a checkout-local virtual environment;
  show the command before starting a service and support dry-run diagnostics.
- Dependency constraints describe a tested installation. Git revisions record
  provenance; startup checks inspect required APIs and tensor layouts, not source
  hashes, usernames, historical runs or an author's environment names.

## Reuse boundaries

The first release retains the working automatic native/aligned/persistent Decode
dispatch and three-fusion lifecycle. Its XQA task body becomes maintained,
licensed source rather than runtime source-string substitutions. The newer
source-level FlashInfer contribution remains an upstream-development reference:
it exposes fixed ownership modes and does not yet contain this complete routing
and fusion path. Measurements must remain distinct. A new import/package layout
is not automatically GPU-validated because the original implementation worked.

VPP reuses the working native vLLM runner/executor adapters and segmented CUDA
DMA/NCCL boundary implementation. Remove experiment evidence writers, historical
autotune-table requirements and private-root checks from the deployment path.
Keep bounded queue ownership, buffer lifetime and required capability checks.
The current folded adapter has a genuine, documented model/parallelism contract;
packaging does not imply support for arbitrary models or asynchronous decode.

The maintained public VPP path is the rank-shared segmented-DMA implementation.
It allocates two source and three receive slots once per rank, shares one stream
per copy direction across both traversals, and preserves W=7 and stage order.
The edge-owned layout is an explicit diagnostic control. Source and installed
model checks for the current implementation are recorded in
[the resource report](vpp-resources.md), including the 64K/128K latency and
memory comparisons. The original full prefill curves retain their historical
measurement scope.

Fusion is opt-in. Preserve residual precision, native quantization scales and
consumer layout. Normal compiler cache identities are legitimate; exact-source
hash access gates are not. The small HTTP fusion differences do not establish
universal serving gains.

## Validation and resources

First verify package imports, configuration, installed wheel contents and command
construction in CPU-only tests. Then use an isolated installed-package smoke for
the selected supported runtime and a bounded model integration anchor. Reuse the
completed performance experiments; do not rerun their full curves for packaging.
Report separately what was measured before extraction and validated afterward.

On the shared development server, all work stays in the private workspace,
selected devices must be idle, four-GPU roles stay within one NUMA node, threads
and build jobs are bounded, and available host memory must stay above 128 GiB.
These operational rules are not universal hardware requirements of the library.

## Public repository contents

Include source, necessary redistributable dependency patches, unit/integration
tests, small examples, CI, documentation and curated benchmark tables. Preserve
third-party licenses and modification notices. Exclude model weights, datasets,
private NVIDIA reference binaries/PTX, credentials, personal connection settings,
environment caches, historical transfer scripts and raw private machine logs.

The existing research repository and immutable experimental evidence remain
unchanged. This checkout is the maintained public source tree, with Git history
instead of version-copy directories.

## Acceptance

1. A fresh checkout has a documented install and launch path without the author's
   filesystem, private repository or historical outputs.
2. Both VPP and bucket attention contain real implementations and runnable entry
   points; unsupported combinations are explained before model allocation.
3. Native mode remains usable independently of optional extensions.
4. Package tests and the bounded installation smoke have recorded outcomes;
   pending GPU validation is stated precisely.
5. The public README explains architecture, usage, support and measured benefits
   without mixing module, model and HTTP measurements.
