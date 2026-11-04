"""Opt-in FP4 producer rewrite; legacy singleton selection remains the default."""

import hashlib
import operator
from pathlib import Path

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch.fx.experimental.symbolic_shapes import statically_known_true
from vllm.compilation.passes.inductor_pass import CustomGraphPass, get_pass_context


def tensor(node, shape, dtype):
    value = getattr(node, "meta", {}).get("val")
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "cuda"
        and value.dtype == dtype
        and value.is_contiguous()
        and len(value.shape) == len(shape)
        and all(statically_known_true(a == b) for a, b in zip(value.shape, shape))
    )


def gemma_raw(weight):
    if (
        getattr(weight, "op", None) != "call_function"
        or weight.target
        not in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, torch.ops.prims.add.default)
        or len(weight.args) != 2
        or type(weight.args[1]) not in (float, int)
        or weight.args[1] != 1.0
        or any(k != "alpha" or v != 1 for k, v in weight.kwargs.items())
    ):
        return None
    converted = weight.args[0]
    if getattr(converted, "op", None) != "call_function":
        return None
    if converted.target == torch.ops.prims.convert_element_type.default:
        if converted.args[1:] != (torch.float32,) or converted.kwargs:
            return None
    elif converted.target == torch.ops.aten._to_copy.default:
        if len(converted.args) != 1 or converted.kwargs != {"dtype": torch.float32}:
            return None
    else:
        return None
    raw = converted.args[0]
    return raw if tensor(raw, (5120,), torch.bfloat16) else None


def items(node, allowed):
    users = list(node.users)
    if any(
        u.op != "call_function"
        or u.target is not operator.getitem
        or len(u.args) != 2
        or type(u.args[1]) is not int
        or u.args[1] not in allowed
        for u in users
    ):
        return None
    return {i: [u for u in users if u.args[1] == i] for i in allowed}


class NormPackPass(CustomGraphPass):
    def __init__(self, audit, *, prefill=False, row_interval=None):
        if row_interval is not None:
            if (
                prefill
                or not isinstance(row_interval, (tuple, list))
                or len(row_interval) != 2
                or any(type(n) is not int for n in row_interval)
                or not 1 <= row_interval[0] <= row_interval[1] <= 8192
            ):
                raise ValueError("Expected one explicit producer row interval within1..8192")
            row_interval = tuple(row_interval)
        self.audit = audit
        self.prefill = prefill
        self.row_interval = row_interval

    def eligible(self, limits):
        if self.row_interval is not None:
            return self.row_interval[0] <= limits[0] <= limits[1] <= self.row_interval[1]
        return (1 <= limits[0] <= limits[1] <= 8192) if self.prefill else limits == (2048, 2048)

    def __call__(self, graph):
        span = get_pass_context().compile_range
        limits = (span.start, span.end)
        count = 0
        if self.eligible(limits):
            quant_op = torch.ops._C.scaled_fp4_quant.out
            names = [a.name for a in quant_op._schema.arguments]
            if names != ["input", "input_scale", "is_sf_swizzled_layout", "output", "output_scale"]:
                raise RuntimeError("Native FP4 quant ABI changed")
            for quant in list(graph.nodes):
                if (
                    quant.op != "call_function"
                    or quant.target is not auto_functionalized
                    or quant.args != (quant_op,)
                    or set(quant.kwargs) != set(names)
                    or quant.kwargs["is_sf_swizzled_layout"] is not True
                ):
                    continue
                normalized = quant.kwargs["input"]
                q = quant.kwargs["output"]
                s = quant.kwargs["output_scale"]
                if (
                    normalized.op != "call_function"
                    or normalized.target is not operator.getitem
                    or normalized.args[1:] != (0,)
                    or list(normalized.users) != [quant]
                ):
                    continue
                norm = normalized.args[0]
                if (
                    norm.op != "call_function"
                    or norm.target != torch.ops.vllm_ir.fused_add_rms_norm.default
                    or norm.kwargs
                    or len(norm.args) not in (4, 5)
                    or norm.args[3] != 1e-6
                    or (len(norm.args) == 5 and norm.args[4] is not None)
                ):
                    continue
                branches = items(norm, (0, 1))
                returns = items(quant, (0, 1, 2))
                if (
                    branches is None
                    or branches[0] != [normalized]
                    or not branches[1]
                    or not any(n.users for n in branches[1])
                    or returns is None
                    or any(n.users for n in returns[0])
                    or len(returns[1]) != 1
                    or len(returns[2]) != 1
                ):
                    continue
                x, residual, weight = norm.args[:3]
                raw = gemma_raw(weight)
                scale = quant.kwargs["input_scale"]
                value = x.meta.get("val")
                if not isinstance(value, torch.Tensor) or value.ndim != 2:
                    continue
                rows = value.shape[0]
                scale_rows = ((rows + 127) // 128) * 128
                if (
                    raw is None
                    or x is residual
                    or not tensor(x, (rows, 5120), torch.bfloat16)
                    or not tensor(residual, (rows, 5120), torch.bfloat16)
                    or not tensor(scale, (), torch.float32)
                    or not tensor(q, (rows, 2560), torch.uint8)
                    or not tensor(s, (scale_rows, 80), torch.int32)
                    or any(
                        n.op != "call_function"
                        or n.target != torch.ops.aten.empty.memory_format
                        or list(n.users) != [quant]
                        for n in (q, s)
                    )
                ):
                    continue
                if any(n.meta["val"].device != x.meta["val"].device for n in (residual, raw, scale, q, s)):
                    continue
                with graph.inserting_before(norm):
                    replacement = graph.call_function(
                        torch.ops.qwen_sm120.gemma_nvfp4_pack.default, args=(x, residual, raw, scale)
                    )
                    replacement.meta["val"] = (q.meta["val"], s.meta["val"], branches[1][0].meta["val"])
                    outputs = [graph.call_function(operator.getitem, args=(replacement, i)) for i in range(3)]
                    for new, old in zip(outputs, (returns[1][0], returns[2][0], branches[1][0])):
                        new.meta["val"] = old.meta["val"]
                    returns[1][0].replace_all_uses_with(outputs[0])
                    returns[2][0].replace_all_uses_with(outputs[1])
                    for old in branches[1]:
                        old.replace_all_uses_with(outputs[2])
                for old in (*returns[0], *returns[1], *returns[2]):
                    graph.erase_node(old)
                graph.erase_node(quant)
                graph.erase_node(normalized)
                for old in branches[1]:
                    graph.erase_node(old)
                graph.erase_node(norm)
                graph.erase_node(q)
                graph.erase_node(s)
                count += 1
            graph.lint()
        self.audit(graph, limits, count)

    def uuid(self):
        source = b"".join(
            Path(__file__).with_name(n).read_bytes()
            for n in ("nvfp4_norm_compile.py", "nvfp4_norm_pack.py", "nvfp4_norm_pack.cu")
        )
        return hashlib.sha256(source + repr((self.prefill, self.row_interval)).encode()).hexdigest()
