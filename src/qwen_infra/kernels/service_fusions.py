"""Service-only native compiler fallbacks and existing extra-XQA capture permit."""

from copy import copy
from collections import Counter
from unittest.mock import patch


def plain_copy(value):
    if isinstance(value, Counter):
        return copy(value)
    if isinstance(value, dict):
        return {k: plain_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(plain_copy(v) for v in value)
    if isinstance(value, set):
        return {plain_copy(v) for v in value}
    return value


def helper_configuration(cfg, owned_prepass, graph_none):
    """Copy configuration metadata, never registered model/cache tensors."""
    result = copy(cfg)
    cc = result.compilation_config = copy(cfg.compilation_config)
    for name, value in vars(cc).items():
        if isinstance(value, (dict, list, tuple, set)):
            setattr(cc, name, plain_copy(value))
    cc.static_forward_context = {}
    cc.pass_config = copy(cc.pass_config)
    for name, value in vars(cc.pass_config).items():
        if isinstance(value, (dict, list, tuple, set)):
            setattr(cc.pass_config, name, plain_copy(value))
    passes = cc.inductor_compile_config
    if owned_prepass is not None and passes.get("post_grad_custom_pre_pass") is owned_prepass:
        del passes["post_grad_custom_pre_pass"]
    cc.cudagraph_mode = graph_none
    return result


def native_helpers(torch, cfg):
    from vllm import ir
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    @support_torch_compile(use_aot_compile=False, dynamic_arg_dims={"x": 0, "residual": 0})
    class NativeNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, residual, raw_gamma, global_inverse):
            normalized, updated = ir.ops.fused_add_rms_norm(x, residual, raw_gamma.float() + 1.0, 1e-6)
            q, scales = scaled_fp4_quant(
                normalized,
                global_inverse,
                is_sf_swizzled_layout=True,
                backend="flashinfer-cutlass",
                padded_n=5120,
            )
            # The native wrapper exposes FP8 scales; the producer replaces its
            # underlying .out ABI, which carries the same bytes as INT32.
            return q, scales.view(torch.int32), updated

    @support_torch_compile(use_aot_compile=False, dynamic_arg_dims={"output": 0, "gate": 0})
    class NativePost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR, num_token_padding=None)

        def forward(self, output, gate, input_scale):
            return self.quant(output * torch.sigmoid(gate), input_scale)[0]

    modules = []
    backends = []
    original = cfg.compilation_config.init_backend
    with set_current_vllm_config(cfg):
        for name, kind in (("norm", NativeNorm), ("post", NativePost)):

            def initialize(*args, _name=name, **kwargs):
                backend = original(*args, **dict(kwargs, prefix="fusion_service_native_" + _name))
                backends.append(backend)
                return backend

            with patch.object(cfg.compilation_config, "init_backend", initialize):
                modules.append(kind(vllm_config=cfg))
    if len(backends) != 2:
        raise RuntimeError("Expected two original native helper compiler backends")
    return modules, backends


class ServiceFusions:
    """No weights/attention owners in helper modules; all tensor inputs are arguments."""

    def __init__(self, worker):
        import torch
        from vllm.config import CUDAGraphMode
        from qwen_infra.kernels.nvfp4_norm_compile import NormPackPass
        from qwen_infra.kernels.nvfp4_norm_pack import register_producer

        self.worker = worker
        self.observers = {"enabled": True}
        self.warming = False
        self.receipt = dict(
            norm_modes={},
            post_modes={},
            compile_audits=[],
            warmup=[],
            capture_permit="all existing lifecycle adapters enabled inside extra-XQA capture",
            helper_weight_parameters=0,
            helper_static_forward_context_empty=True,
            helper_graph_mode="NONE",
            norm_row_interval=[1, 32],
        )
        cc = worker.compilation_config
        if "post_grad_custom_pre_pass" in cc.inductor_compile_config:
            raise RuntimeError("Service NormPack prepass already has an owner")
        self.cfg = helper_configuration(worker.vllm_config, None, CUDAGraphMode.NONE)
        self.modules, self.backends = native_helpers(torch, self.cfg)
        _, self.receipt["producer_resources"] = register_producer(
            torch,
            worker.kernel_cache,
            permit=self.permitted,
            native_fallback=self.norm,
            route_observer=self.observe_norm,
            observer_control=self.observers,
        )

        def audit(graph, limits, count):
            emitted = sum(
                n.op == "call_function" and "qwen_sm120.gemma_nvfp4_pack" in str(n.target)
                for n in graph.nodes
            )
            if emitted != count or count and not 1 <= limits[0] <= limits[1] <= 32:
                raise RuntimeError("Service NormPack escaped its Decode compiler interval")
            self.receipt["compile_audits"].append(
                dict(limits=list(limits), replacements=count, emitted=emitted)
            )

        self.prepass = NormPackPass(audit, row_interval=(1, 32))
        cc.inductor_compile_config["post_grad_custom_pre_pass"] = self.prepass
        self.receipt["pass_uuid"] = self.prepass.uuid()

    def permitted(self):
        lifecycle = getattr(self.worker, "full_lifecycle", None)
        adapters = getattr(lifecycle, "adapters", ())
        return bool(adapters) and all(adapter.enabled for adapter in adapters)

    def phase(self):
        import torch

        return (
            "warmup" if self.warming else "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        )

    def observe_norm(self, route, rows):
        key = f"{route}:{self.phase()}:{rows}"
        counts = self.receipt["norm_modes"]
        counts[key] = counts.get(key, 0) + 1

    def invoke(self, index, *args):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        # Only norm/pointwise work runs here. Restore the outer Attention metadata
        # and slots even if native compilation or execution raises.
        with set_forward_context(
            None, self.cfg, num_tokens=args[0].shape[0], cudagraph_runtime_mode=CUDAGraphMode.NONE
        ):
            return self.modules[index](*args)

    def norm(self, x, residual, raw_gamma, global_inverse):
        return self.invoke(0, x, residual, raw_gamma, global_inverse)

    def post(self, output, gate, input_scale):
        if self.observers["enabled"]:
            key = f"{self.phase()}:{output.shape[0]}"
            counts = self.receipt["post_modes"]
            counts[key] = counts.get(key, 0) + 1
        return self.invoke(1, output, gate, input_scale)

    def bind_and_warm(self):
        import torch

        runner = self.worker.model_runner
        fusion = runner and self.worker.full_lifecycle.fusions
        if fusion is None:
            raise RuntimeError("The service fusion worker requires existing Attention owners")
        for owner in fusion.owners.values():
            owner.native_post = self.post
        pair = next(
            (
                (block.post_attention_layernorm.weight, block.mlp.gate_up_proj.input_global_scale_inv)
                for block in runner.model.modules()
                if hasattr(block, "post_attention_layernorm")
                and hasattr(getattr(block, "mlp", None), "gate_up_proj")
            ),
            None,
        )
        if pair is None:
            raise ValueError("No native Gemma/NVFP4 norm inputs found")
        gamma, inverse = pair
        layer = next(iter(fusion.owners.values())).layer
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        self.warming = True
        try:
            with torch.inference_mode():
                for rows in (32, 1, 2, 8, 33, 2048):
                    x = torch.zeros((rows, 5120), device=runner.device, dtype=torch.bfloat16)
                    residual = torch.zeros_like(x)
                    norm = self.norm(x, residual, gamma, inverse)
                    del norm, x, residual
                    qkv = torch.zeros(
                        (rows, layer.q_size * 2 + layer.kv_size * 2),
                        device=runner.device,
                        dtype=torch.bfloat16,
                    )
                    dummy = runner.model_state.prepare_dummy_inputs(1, rows)
                    positions = torch.zeros_like(
                        dummy.get("positions", runner.input_buffers.positions[:rows])
                    )
                    q, k, v, gate = layer._project_qkv_gate(qkv, positions)
                    output = torch.zeros_like(q)
                    value = self.post(output, gate, layer.o_proj.input_scale)
                    self.receipt["warmup"].append(
                        dict(
                            rows=rows,
                            output_shape=list(output.shape),
                            output_stride=list(output.stride()),
                            gate_stride=list(gate.stride()),
                            output_dtype=str(output.dtype),
                            gate_dtype=str(gate.dtype),
                        )
                    )
                    del value, output, q, k, v, gate, qkv, positions, dummy
                torch.cuda.synchronize()
        finally:
            self.warming = False
        self.receipt.update(
            warmed_before_profile_and_capture=True,
            helper_allocated_delta_bytes=torch.cuda.memory_allocated() - allocated,
            helper_reserved_delta_bytes=torch.cuda.memory_reserved() - reserved,
            helper_backends=[type(b).__name__ for b in self.backends],
        )
