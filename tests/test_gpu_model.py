"""Opt-in installed-package integration anchor; no throughput claim.

Run only on four selected idle GPUs under the deployment's resource policy:
QWEN_INFRA_TEST_MODEL=/path/to/model pytest -m gpu -s tests/test_gpu_model.py
"""

import os

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("profile", ["native", "decode-bucket", "decode-fused"])
def test_actual_vllm_model_returns_finite_complete_requests(profile):
    model = os.environ.get("QWEN_INFRA_TEST_MODEL")
    if not model:
        pytest.skip("Set QWEN_INFRA_TEST_MODEL to opt in to a bounded model anchor")
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() != 4:
        pytest.skip("Expose exactly four selected idle GPUs for this model anchor")
    from vllm import LLM, SamplingParams
    from qwen_infra.config import decode_options

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    options = decode_options(profile)
    options.update(max_num_seqs=8, max_num_batched_tokens=2048)
    llm = LLM(
        model=model,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        **options,
    )
    try:
        outputs = llm.generate(
            ["Explain gravity."] * 8, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        )
        assert len(outputs) == 8 and all(output.finished for output in outputs)
        assert all(output.outputs and 1 <= len(output.outputs[0].token_ids) <= 8 for output in outputs)
        if profile != "native":
            status = llm.collective_rpc("qwen_infra_status", timeout=60)
            assert len(status) == 4
            assert all(row["fallback_reason"] is None and row["bucket_installed"] for row in status)
            assert all(dict(rows=8, route="aligned") in row["captured"] for row in status)
            assert all(row["last_selected_route"] == "aligned" for row in status)
    finally:
        llm.llm_engine.engine_core.shutdown()
