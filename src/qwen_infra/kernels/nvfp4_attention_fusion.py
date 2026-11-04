"""Opt-in complete-FA module adapters, retaining native projections and TP SUM."""

import torch
from qwen_infra.owners import stable_owner_key

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

_OWNERS = {}


@torch.library.custom_op("qwen_sm120::fa_front_append", mutates_args=(), device_types="cuda")
def front_op(qkv: torch.Tensor, positions: torch.Tensor, owner_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Like native unified_kv_cache_update, cache storage comes from the current
    # attention context. Q is the ordering dependency for the following reader.
    return _OWNERS[owner_id].front(qkv, positions)


@front_op.register_fake
def fake_front(qkv, positions, owner_id):
    owner = _OWNERS[owner_id]
    q = qkv.new_empty((qkv.shape[0], owner.layer.q_size))
    return q, torch.empty_like(q)


@torch.library.custom_op("qwen_sm120::fa_tail_fp8", mutates_args=(), device_types="cuda")
def tail_op(
    q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None, gate: torch.Tensor, owner_id: int
) -> torch.Tensor:
    return _OWNERS[owner_id].tail(q, k, v, gate)


@tail_op.register_fake
def fake_tail(q, k, v, gate, owner_id):
    return q.new_empty(q.shape, dtype=torch.float8_e4m3fn)


class AttentionFusionOwner:
    def __init__(self, layer, front_library, *, capture_only=False, stable_name=None):
        if (
            layer.head_dim != 256
            or layer.num_heads != 6
            or layer.num_kv_heads != 1
            or not layer.attn_output_gate
            or not layer.o_proj.input_is_parallel
            or get_input_quant_key(layer.o_proj) != kFp8StaticTensorSym
            or layer.o_proj.input_scale.dtype != torch.float32
            or layer.o_proj.input_scale.numel() != 1
        ):
            raise ValueError("Expected original TP4 Q6/KV1/D256/static FP8 OUT consumer")
        self.layer, self.library = layer, front_library
        self.capture_only = capture_only
        self.native_post = None
        self.observe_calls = True
        self.adapter = None
        self.front_calls = 0
        self.tail_calls = 0
        self.fallback_calls = 0
        self.stable_name = stable_name
        self.owner_id = id(self) if stable_name is None else stable_owner_key(stable_name)
        existing = _OWNERS.get(self.owner_id)
        if existing is not None:
            raise RuntimeError(
                f"FA owner key already live: {existing.stable_name!r} conflicts with {stable_name!r}"
            )
        _OWNERS[self.owner_id] = self

    def front(self, qkv, positions):
        from vllm.model_executor.layers.attention.attention import get_attention_context
        from vllm.v1.attention.backends.flashinfer import _nvfp4_kv_views
        from .nvfp4_attention_front import launch

        _, attention, cache, slots = get_attention_context(self.layer.attn.layer_name)
        rope = self.layer.rotary_emb
        if (
            (self.capture_only and (self.adapter is None or not self.adapter.enabled))
            or slots is None
            or cache.numel() == 0
            or not self.layer.use_fused_qk_norm_rope_gate
            or rope.rotary_dim not in (64, 128, 256)
            or rope.cos_sin_cache.dtype not in (torch.float32, torch.bfloat16)
        ):
            q, k, v, gate = self.layer._project_qkv_gate(qkv, positions)
            if slots is not None:
                attention.impl.do_kv_cache_update(
                    attention, k.view(-1, 1, 256), v.view(-1, 1, 256), cache, slots
                )
            if self.observe_calls:
                self.fallback_calls += 1
            return q, gate
        data, scales, layout = _nvfp4_kv_views(cache, 1)
        if layout == "HND":
            data = tuple(t.transpose(1, 2) for t in data)
            scales = tuple(t.transpose(1, 2) for t in scales)
        result = launch(
            torch,
            self.library,
            qkv,
            positions,
            self.layer.q_norm.weight,
            self.layer.k_norm.weight,
            rope.cos_sin_cache,
            slots,
            data,
            scales,
            attention._k_scale,
            attention._v_scale,
            qheads=6,
            kvheads=1,
            rotary=rope.rotary_dim,
            epsilon=self.layer.q_norm.variance_epsilon,
            mrope_section=getattr(rope, "mrope_section", None) if positions.ndim == 2 else None,
        )
        if self.observe_calls:
            self.front_calls += 1
        return result

    def tail(self, q, k, v, gate):
        adapter = self.adapter
        if adapter is None or not adapter.enabled:
            # Cold profiling/fallback only. The native model/default stays separate.
            output = self.layer.attn(q, k, v)
            if self.observe_calls:
                self.fallback_calls += 1
            if self.native_post is not None:
                return self.native_post(output, gate, self.layer.o_proj.input_scale)
            output = output * torch.sigmoid(gate)
            return self.layer.o_proj.quant_method.kernel.quant_fp8(output, self.layer.o_proj.input_scale)[0]
        output = q.new_empty(q.shape, dtype=torch.float8_e4m3fn)
        if adapter.tail is not None:
            raise RuntimeError("Nested tail ownership")
        adapter.tail = (gate.view(-1, 6, 256), self.layer.o_proj.input_scale, output.view(-1, 6, 256))
        try:
            # Preserve original Attention dispatch and (unless front fused) writer.
            # The native BF16 return is private scratch, not a producer/consumer copy.
            self.layer.attn(q, k, v)
        finally:
            adapter.tail = None
        if self.observe_calls:
            self.tail_calls += 1
        return output

    def module(self, *, front=False, tail=False):
        return FusedAttention(self, front=front, tail=tail)

    def close(self):
        self.adapter = None
        if _OWNERS.get(self.owner_id) is self:
            del _OWNERS[self.owner_id]


class ModelAttentionFusions:
    """Explicit worker opt-in, installed once before profile/compilation.

    Opaque front/tail bodies capture native fallbacks in the native Graph and
    fusion kernels only in the adapter-enabled extra XQA Graph. Profile cache
    generations may release adapters without retiring these model owners.
    """

    def __init__(self, model, attentions, front_library):
        expected = {id(a): a for a in attentions}
        self.owners = {}
        self.installed = []
        try:
            for layer in tuple(model.modules()):
                attention = getattr(layer, "attn", None)
                if id(attention) not in expected:
                    continue
                name = attention.layer_name
                if name in self.owners or "forward" in vars(layer):
                    raise RuntimeError("FA parent forward already has an instance owner")
                owner = AttentionFusionOwner(layer, front_library, capture_only=True, stable_name=name)
                module = owner.module(front=True, tail=True)
                replacement = module.forward
                layer.forward = replacement
                self.owners[name] = owner
                self.installed.append((layer, replacement))
            if len(self.owners) != len(expected):
                raise ValueError("Every native FA must have one supported parent")
        except Exception:
            self.restore()
            raise

    def release_adapters(self):
        for owner in self.owners.values():
            owner.adapter = None

    def restore(self):
        while self.installed:
            layer, replacement = self.installed[-1]
            if vars(layer).get("forward") is not replacement:
                raise RuntimeError("Model fusion forward ownership changed")
            delattr(layer, "forward")
            self.installed.pop()
        for owner in self.owners.values():
            owner.close()
        self.owners.clear()


class FusedAttention(torch.nn.Module):
    def __init__(self, owner, *, front, tail):
        super().__init__()
        self.layer = owner.layer
        self.owner_id = owner.owner_id
        self.front, self.tail = front, tail

    def forward(self, positions, hidden_states):
        qkv, _ = self.layer.qkv_proj(hidden_states)
        if self.front:
            q, gate = front_op(qkv, positions, self.owner_id)
            k = v = None
        else:
            q, k, v, gate = self.layer._project_qkv_gate(qkv, positions)
        if self.tail:
            data = tail_op(q, k, v, gate, self.owner_id)
            activation = QuantizedActivation(
                data, self.layer.o_proj.input_scale, torch.bfloat16, data.shape, kFp8StaticTensorSym
            )
        else:
            activation = self.layer.attn(q, k, v) * torch.sigmoid(gate)
        output, _ = self.layer.o_proj(activation)
        return output
