from argparse import Namespace
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen_infra.cli import serve_command
from qwen_infra.config import PrefillConfig, decode_contract, decode_options, host_reserve_bytes
from qwen_infra.kernels.service_fusions import plain_copy
from qwen_infra.owners import OWNER_ABI, stable_owner_key
from qwen_infra.limits import MAX_SEQUENCE_LENGTH


def test_native_command_preserves_user_tokens_and_has_no_extension_options():
    args = Namespace(profile="native", tensor_parallel_size=None, model="a model $(literal)")
    tail = ["--host", "127.0.0.1", "--api-key", "spaces and $literal"]
    command = serve_command(args, tail)
    assert command == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model, *tail]


def test_compile_identity_is_stamped_before_model_construction():
    options = decode_options("decode-fused")
    assert options["additional_config"]["qwen_infra_attention_owner_abi"] == OWNER_ABI
    assert options["compilation_config"]["compile_ranges_endpoints"] == [32, 2048]
    assert decode_options("native") == {}


def test_helper_config_copy_preserves_counter_missing_key_semantics():
    original = {"seen": Counter({"a": 3}), "nested": [{"value": 1}]}
    copied = plain_copy(original)
    assert isinstance(copied["seen"], Counter)
    copied["seen"]["unseen"] += 1
    copied["nested"][0]["value"] = 2
    assert original["seen"]["unseen"] == 0
    assert original["nested"][0]["value"] == 1


def test_named_owner_identity_survives_fresh_python_process():
    name = "language_model.model.layers.3.self_attn.attn"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    value = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from qwen_infra.owners import stable_owner_key; print(stable_owner_key(" + repr(name) + "))",
        ],
        env=env,
        text=True,
        timeout=15,
    )
    assert int(value) == stable_owner_key(name)
    assert -(1 << 63) <= int(value) < 0
    assert stable_owner_key(name) != stable_owner_key(name.replace(".3.", ".7."))


def test_geometry_contract_allows_smaller_configured_context():
    text = SimpleNamespace(
        num_attention_heads=24, num_key_value_heads=4, head_dim=256, hidden_size=5120, rms_norm_eps=1e-6
    )
    kwargs = dict(tp=4, pp=1, dtype="bfloat16", cache_dtype="nvfp4", max_model_len=32768, fused=True)
    assert decode_contract(text, **kwargs) is None
    assert decode_contract(text, **dict(kwargs, max_model_len=262144)) is None
    assert decode_contract(text, **dict(kwargs, max_model_len=MAX_SEQUENCE_LENGTH + 1)) is not None
    assert decode_contract(text, **dict(kwargs, tp=2)) is not None


def test_vpp_and_cpp_share_native_precision_and_resource_configuration():
    cpp = PrefillConfig("model", profile="cpp").engine_options()
    vpp = PrefillConfig("model", profile="vpp").engine_options()
    for name in cpp:
        assert cpp[name] == vpp[name]
    assert vpp["additional_config"]["qwen_infra_vpp_static_scales"] is True
    compatible = PrefillConfig("model", transport="nccl").engine_options()
    assert compatible["additional_config"]["qwen_infra_vpp_transport"] == "native"
    with pytest.raises(ValueError):
        PrefillConfig("model", chunk_size=16385).engine_options()


def test_cpu_cli_does_not_import_gpu_packages():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    program = "from qwen_infra.cli import main; import sys; main(['doctor']); assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
    output = subprocess.check_output([sys.executable, "-c", program], text=True, env=env, timeout=15)
    assert "packages" in json.loads(output)


def test_optional_host_reserve_is_separate_from_transport_abi():
    assert host_reserve_bytes("0") == 0
    assert host_reserve_bytes("128") == 128 * 2**30
    for value in ("-1", "1.5", "inf", "", str(1 << 64)):
        with pytest.raises(ValueError):
            host_reserve_bytes(value)
