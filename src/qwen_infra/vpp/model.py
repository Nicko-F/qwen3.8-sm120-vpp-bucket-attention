# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vLLM a9fe30e9 Qwen3_5Model/Qwen3NextModel and
# Huawei VPP c409752 patch_vpp_make_layers.py (Apache-2.0).
"""Opt-in CUDA fold-back placement, with two independent compile identities.

No arithmetic/kernel replacement. Global checkpoint/cache layer names remain
unchanged. Only the selected worker imports/installs this adapter.
"""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import qwen3_5 as native
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.model_executor.offloader import get_offloader
from vllm.sequence import IntermediateTensors

from qwen_infra.vpp.layout import layer_ranges

_NativeBackbone = native.Qwen3_5Model


def received_input_views(intermediate, rows):
    """Metadata-only validation; the caller owns the live destination lease."""
    if type(rows) is not int or rows < 0:
        raise ValueError("Expected a nonnegative native padded-row count")
    if set(intermediate.tensors) != {"hidden_states", "residual"}:
        raise ValueError("Direct input requires the dual-BF16 boundary")
    hidden, residual = (intermediate[name] for name in ("hidden_states", "residual"))
    if (
        any(
            value.ndim != 2
            or value.shape[1] != 5120
            or value.shape[0] < rows
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            for value in (hidden, residual)
        )
        or hidden.device != residual.device
    ):
        raise ValueError("Direct input requires contiguous BF16 [rows,5120] planes")
    return IntermediateTensors({name: value[:rows] for name, value in intermediate.tensors.items()})


class ConstructionGroup:
    """Model construction only: the final norm/head owner is logical rank0."""

    def __getattr__(self, key):
        return getattr(get_pp_group(), key)

    @property
    def is_last_rank(self):
        return get_pp_group().rank_in_group == 0


@support_torch_compile(
    use_aot_compile=False,
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_out": 0,
        "residual_out": 0,
    },
)
class VirtualStage(nn.Module):
    def __init__(self, *, backbone, span, first, last, vllm_config, prefix="", boundary_producer=None):
        super().__init__()
        self.first, self.last = first, last
        # The backbone owns registration/checkpoint names and cache storage.
        # These aliases must not create a second set of checkpoint names.
        object.__setattr__(self, "blocks", tuple(backbone.layers[i] for i in range(*span)))
        object.__setattr__(self, "embedding", backbone.embed_tokens)
        object.__setattr__(self, "final_norm", backbone.norm)
        object.__setattr__(self, "boundary_producer", boundary_producer)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        hidden_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
        publication_token: torch.Tensor | None = None,
    ):
        if self.first:
            hidden = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
            residual = None
        else:
            hidden = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        blocks = self.blocks if self.boundary_producer is None else self.blocks[:-1]
        for layer in blocks:
            hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
        if self.boundary_producer is not None:
            # Exact native dense/non-scaled last-FA prefix (cold-validated).
            layer = self.blocks[-1]
            hidden, residual = layer.input_layernorm(hidden, residual)
            hidden = layer.self_attn(hidden_states=hidden, positions=positions)
            # Inline the already-validated Producer body into this stage's FX;
            # no nested compiled call, Python hook or pointer baked into FX.
            hidden, residual = self.boundary_producer.forward(
                hidden, residual, hidden_out, residual_out, publication_token
            )
        if self.last:
            hidden, _ = self.final_norm(hidden, residual)
            return hidden
        return IntermediateTensors({"hidden_states": hidden, "residual": residual})


class FoldedBackbone(nn.Module):
    hf_to_vllm_mapper = _NativeBackbone.hf_to_vllm_mapper
    load_weights = _NativeBackbone.load_weights

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        group = get_pp_group()
        if (
            config.num_hidden_layers != 64
            or config.hidden_size != 5120
            or group.world_size != 4
            or vllm_config.parallel_config.tensor_parallel_size != 1
            or "moe" in config.model_type
        ):
            raise ValueError("VPP port supports only the dense64-layer TP1/PP4 model")
        self.config, self.quant_config = config, vllm_config.quant_config
        self.rank = group.rank_in_group
        self.ranges = layer_ranges(self.rank)
        self.start_layer, self.end_layer = self.ranges[0][0], self.ranges[1][1]
        self.vocab_size = config.vocab_size
        self.embed_tokens = native.VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        local = {i for span in self.ranges for i in range(*span)}
        wrapped = iter(
            get_offloader().wrap_modules(
                native.Qwen3_5DecoderLayer(
                    vllm_config, layer_type=config.layer_types[i], prefix=f"{prefix}.layers.{i}"
                )
                for i in sorted(local)
            )
        )
        self.layers = nn.ModuleList(next(wrapped) if i in local else PPMissingLayer() for i in range(64))
        self.norm = (
            native.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.rank == 0
            else PPMissingLayer()
        )
        self.num_redundant_experts = 0
        self.is_fused_shared_expert_enabled = False
        self.aux_hidden_state_layers = ()
        self.make_empty_intermediate_tensors = native.make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.stages = nn.ModuleList(
            VirtualStage(
                backbone=self,
                span=span,
                first=self.rank == 0 and phase == 0,
                last=self.rank == 0 and phase == 1,
                vllm_config=vllm_config,
                prefix=f"{prefix}.vpp_stage_{phase}",
            )
            for phase, span in enumerate(self.ranges)
        )
        # None is native memory-profile/warmup: exercise all16 owned layers.
        self.traversal = None
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.stage_positions = [None, None]
        self.return_inputs = None
        self.last_runtime_mode = None
        self.direct_inputs = bool(vllm_config.additional_config.get("qwen_infra_vpp_direct_inputs", False))
        self.direct_inputs_active = False  # Real calls only; warmup keeps native addresses.
        self.input_binding_stats = dict(direct_calls=0, staged_calls=0, logical_copy_bytes_avoided=0)
        self.output_binding = None
        self.profile_outputs = None  # Cold dummy path, outside compiled regions.
        object.__setattr__(self, "bound_stages", (None, None))
        self.boundary_compiler_audits = []

    def initialize_bound_stages(self, vllm_config):
        """After weight postprocessing; aliases share native checkpoint owners."""
        from unittest.mock import patch
        from qwen_infra.vpp.producer import BoundaryProducer, SlotOutputBackend

        if any(stage is not None for stage in self.bound_stages):
            raise RuntimeError("Construct output-bound stages once per loaded model")
        bound = []
        for phase, span in enumerate(self.ranges):
            if (phase == 0 and self.rank == 3) or (phase == 1 and self.rank == 0):
                bound.append(None)  # Local fold and final norm/head keep native paths.
                continue
            layer = self.layers[span[1] - 1]
            if (
                layer.layer_type != "full_attention"
                or layer.layer_scale
                or layer.use_attn_reduce_scatter_for_moe
            ):
                raise ValueError("Expected the actual non-scaled dense TP1 FA boundary")

            def audit(graph, count):
                self.boundary_compiler_audits.append(count)

            def factory(
                config, prefix="", is_encoder=False, use_aot_compile=True, tag=f"vpp_bound_stage_{phase}"
            ):
                return SlotOutputBackend(
                    config, prefix=tag, is_encoder=is_encoder, use_aot_compile=use_aot_compile, audit=audit
                )

            with patch.object(vllm_config.compilation_config, "init_backend", factory):
                producer = BoundaryProducer(
                    layer.post_attention_layernorm, layer.mlp, early_ready=True, vllm_config=vllm_config
                )
                bound.append(
                    VirtualStage(
                        backbone=self,
                        span=span,
                        first=self.rank == 0 and phase == 0,
                        last=False,
                        vllm_config=vllm_config,
                        boundary_producer=producer,
                    )
                )
        # Do not register aliases a second time in state_dict/load_weights.
        object.__setattr__(self, "bound_stages", tuple(bound))

    @property
    def use_sequence_parallel(self):
        return False

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        self.last_runtime_mode = get_forward_context().cudagraph_runtime_mode
        if self.traversal is None:
            output = self.run_stage(0, input_ids, positions, intermediate_tensors, inputs_embeds)
            return self.run_stage(1, None, positions, output, None)
        return self.run_stage(self.traversal, input_ids, positions, intermediate_tensors, inputs_embeds)

    def run_stage(self, phase, input_ids, positions, intermediate, inputs_embeds):
        # Outside compiled regions: native PW wrappers require stable addresses.
        n = positions.shape[-1]
        if n > self.max_tokens:
            raise ValueError("VPP input exceeds the fixed native token budget")
        if self.stage_positions[phase] is None:
            self.stage_positions[phase] = positions.new_empty((*positions.shape[:-1], self.max_tokens))
        # The first full-budget trace specializes a contiguous [3, N] input.
        # A [:, :n] view would retain max_tokens row stride at smaller N,
        # violating native guard-free compiled code. Keep one base pointer but
        # reshape its flat prefix to the current contiguous logical shape.
        stable_positions = self.stage_positions[phase].view(-1)[: positions.numel()].view(positions.shape)
        stable_positions.copy_(positions)
        if phase == 1:
            real_direct_policy = getattr(self, "direct_inputs", False) and self.direct_inputs_active
            direct = real_direct_policy and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
            if direct:
                intermediate = received_input_views(intermediate, n)
            else:
                if self.return_inputs is None:
                    self.return_inputs = {
                        name: value.new_empty((self.max_tokens, *value.shape[1:]))
                        for name, value in intermediate.tensors.items()
                    }
                intermediate = IntermediateTensors(
                    {name: value[:n].copy_(intermediate[name]) for name, value in self.return_inputs.items()}
                )
            if real_direct_policy:
                self.input_binding_stats["direct_calls" if direct else "staged_calls"] += 1
                if direct:
                    self.input_binding_stats["logical_copy_bytes_avoided"] += n * 5120 * 4
        profile = getattr(self, "profile_outputs", None)
        output = getattr(self, "output_binding", None)
        cold = profile is not None and self.bound_stages[phase] is not None
        if cold:
            if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE:
                raise RuntimeError("Cold output priming must be outside Graph capture")
            output = profile.prepare(n)
        if output is not None and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE:
            stage = self.bound_stages[phase]
            if stage is None:
                raise RuntimeError("No wire output belongs to this virtual stage")
            result = stage(input_ids, stable_positions, intermediate, inputs_embeds, *output)
            if cold:
                profile.finish()
            return result
        return self.stages[phase](input_ids, stable_positions, intermediate, inputs_embeds)


def install():
    """Worker-process scoped; native service processes do not import this module."""
    native.get_pp_group = lambda: ConstructionGroup()
    native.Qwen3_5Model = FoldedBackbone


def find_backbone(model):
    found = [module for module in model.modules() if isinstance(module, FoldedBackbone)]
    if len(found) != 1:
        raise RuntimeError(f"Expected one folded text backbone, got {len(found)}")
    return found[0]
