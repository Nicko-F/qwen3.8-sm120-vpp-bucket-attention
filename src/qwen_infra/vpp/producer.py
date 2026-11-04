"""V5 TP1 boundary output binding and opt-in compiled plane publication.

Quantization is adapted from vLLM's Apache-2.0 FlashInfer CUTLASS apply_weights.
Only the destination of the last GEMM changes. Cold views/capability preparation
stays outside compile. Publication ops are explicit data dependencies, not
Python module hooks; capture/PW publication is intentionally unsupported.
"""

import hashlib
from pathlib import Path

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from vllm._custom_ops import scaled_fp4_quant
from vllm.compilation.backends import VllmBackend
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.passes.inductor_pass import CustomGraphPass


@torch.library.custom_op("qwen_infra_vpp::nvfp4_mm_out", mutates_args={"out"}, device_types="cuda")
def nvfp4_mm_out(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> None:
    from flashinfer import mm_fp4

    # Identical transpose/cast, block16, scales, autotuner and workspace to the
    # installed vllm.utils.flashinfer wrapper. FI passes out.data_ptr to CUTLASS D.
    mm_fp4(
        a,
        weight.t(),
        a_scale.view(torch.uint8),
        weight_scale.view(torch.uint8).t(),
        alpha,
        out_dtype=out.dtype,
        out=out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cutlass",
        use_nvfp4=True,
    )


@nvfp4_mm_out.register_fake
def _fake(a, weight, a_scale, weight_scale, alpha, out):
    return None


@torch.library.custom_op(
    "qwen_infra_vpp::residual_ready", mutates_args={"normalized", "out", "token"}, device_types="cuda"
)
def residual_ready(
    normalized: torch.Tensor, residual: torch.Tensor, out: torch.Tensor, token: torch.Tensor
) -> None:
    # Eager fallback. The native last pass exposes this copy to Inductor so it
    # can fuse into Norm, while retaining the explicit MUTATED destination ABI.
    out.copy_(residual)
    residual_record(normalized, out, token)


@residual_ready.register_fake
def _residual_fake(normalized, residual, out, token):
    return None


@torch.library.custom_op(
    "qwen_infra_vpp::residual_record", mutates_args={"normalized", "out", "token"}, device_types="cuda"
)
def residual_record(normalized: torch.Tensor, out: torch.Tensor, token: torch.Tensor) -> None:
    from .vpp_native import load

    # A conservative mutation of normalized makes publication precede Gate/Up
    # in the compiler's dataflow. Its numeric values are not changed. token's
    # CPU mask is actually updated; the capability strongly owns pool/ticket.
    load().record_plane(out, token, 1)


@residual_record.register_fake
def _record_fake(normalized, out, token):
    return None


@torch.library.custom_op("qwen_infra_vpp::hidden_ready", mutates_args={"token"}, device_types="cuda")
def hidden_ready(hidden: torch.Tensor, token: torch.Tensor) -> None:
    from .vpp_native import load

    load().record_plane(hidden, token, 0)


@hidden_ready.register_fake
def _hidden_fake(hidden, token):
    return None


class SlotOutputPass(CustomGraphPass):
    """Use the native last-pass helper only for our explicit write-only out ABI."""

    def __init__(self, manager, audit):
        self.manager, self.audit = manager, audit

    def __call__(self, graph):
        self.manager(graph)
        fix = self.manager.fix_functionalization
        fix.nodes_to_remove = []
        contracts = (
            (torch.ops.qwen_infra_vpp.residual_ready.default, ["normalized", "out", "token"]),
            (torch.ops.qwen_infra_vpp.nvfp4_mm_out.default, ["out"]),
            (torch.ops.qwen_infra_vpp.hidden_ready.default, ["token"]),
        )
        count = 0
        for op, expected in contracts:
            names = tuple(arg.name for arg in op._schema.arguments)
            mutated = [
                arg.name
                for arg in op._schema.arguments
                if arg.alias_info is not None and arg.alias_info.is_write
            ]
            if mutated != expected:
                raise RuntimeError("Boundary publication mutation ABI changed")
            for node in list(graph.nodes):
                if node.op == "call_function" and node.target is auto_functionalized and node.args[0] == op:
                    # Explicit source-plane/capability inputs, never borrowed
                    # model activations whose pre-mutation value must survive.
                    fix.defunctionalize(
                        graph, node, {i + 1: name for i, name in enumerate(mutated)}, args=names
                    )
                    count += 1
        for node in fix.nodes_to_remove:
            graph.erase_node(node)
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.qwen_infra_vpp.residual_ready.default:
                normalized, residual, out, token = node.args
                # Insert AFTER functionalization/native copy propagation. A
                # readonly publication input can become an equal temporary;
                # an explicit mutating out must stay the caller's destination.
                with graph.inserting_before(node):
                    copy = graph.call_function(torch.ops.aten.copy_.default, args=(out, residual))
                    copy.meta.update(out.meta)
                node.target = torch.ops.qwen_infra_vpp.residual_record.default
                node.args = (normalized, out, token)
        # No DCE after reinplacement: a mutation-only node may appear unused.
        self.audit(graph, count)

    def uuid(self):
        return hashlib.sha256(Path(__file__).read_bytes() + self.manager.uuid().encode()).hexdigest()


class SlotOutputBackend(VllmBackend):
    def __init__(self, config, *, prefix="", is_encoder=False, use_aot_compile=True, audit):
        super().__init__(config, prefix=prefix, is_encoder=is_encoder, use_aot_compile=use_aot_compile)
        self.output_audit = audit

    def configure_post_pass(self):
        super().configure_post_pass()
        self.inductor_config[self.pass_key] = SlotOutputPass(self.pass_manager, self.output_audit)


@support_torch_compile(
    use_aot_compile=False,
    dynamic_arg_dims={"hidden_states": 0, "residual": 0, "hidden_out": 0, "residual_out": 0},
)
class BoundaryProducer(torch.nn.Module):
    """Actual last-block Norm/MLP chain, explicit output tensors in native FX.

    This is the P TP1 shard, not a new single-GPU deployment. Scope intentionally
    excludes optional layer_scale, bias, experts and TP collectives; callers
    must retain the native path for those. No weights are repacked or cloned.
    """

    def __init__(self, norm, mlp, *, early_ready=False):
        super().__init__()
        down = mlp.down_proj
        if (
            type(down.quant_method.kernel).__name__ != "FlashInferCutlassNvFp4LinearKernel"
            or down.tp_size != 1
            or down.bias is not None
            or mlp.expert_gate is not None
            or tuple(down.weight.shape) != (5120, 8704)
            or down.output_size_per_partition != 5120
        ):
            raise ValueError("Expected actual TP1 dense, unpadded CUTLASS NVFP4 Down")
        self.norm, self.mlp = norm, mlp
        self.early_ready = early_ready

    def forward(self, hidden_states, residual, hidden_out, residual_out, publication_token=None):
        normalized, updated_residual = self.norm(hidden_states, residual)
        # Native norm has no public out. Preserve FP32 sum/variance and count
        # this copy unless actual generated code proves it fused/eliminated.
        if self.early_ready:
            residual_ready(normalized, updated_residual, residual_out, publication_token)
        else:
            residual_out.copy_(updated_residual)
        gate_up, _ = self.mlp.gate_up_proj(normalized)
        activated = self.mlp.act_fn(gate_up)
        down = self.mlp.down_proj
        packed, scales = scaled_fp4_quant(
            activated,
            down.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
            padded_n=activated.shape[-1],
        )
        nvfp4_mm_out(packed, down.weight, scales, down.weight_scale, down.alpha, hidden_out)
        if self.early_ready:
            hidden_ready(hidden_out, publication_token)
        return hidden_out, residual_out
