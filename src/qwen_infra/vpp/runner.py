# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Native vLLM a9fe30e9 runner hooks; context retention follows Huawei VPP
# c409752 model_runner_v1.py. The native CUDA attention/GDN/MLP are unchanged.
"""P-only V2 runner: prepare once, return with saved inputs, advance once."""

from copy import copy
from collections import deque
from dataclasses import fields, is_dataclass

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

from qwen_infra.vpp.model import find_backbone, install, received_input_views


def snapshot(value):
    """Own input/metadata buffers, never copy model/KV/state storage."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value):
        result = copy(value)
        for field in fields(value):
            setattr(result, field.name, snapshot(getattr(value, field.name)))
        return result
    if isinstance(value, tuple):
        return tuple(snapshot(x) for x in value)
    if isinstance(value, list):
        return [snapshot(x) for x in value]
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return value


class FoldedRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        from vllm.distributed.parallel_state import get_pp_group

        install()
        transfer = vllm_config.kv_transfer_config
        if transfer is not None:
            raise ValueError("This standalone P+1 adapter does not provide a KV connector")
        if (
            vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or vllm_config.scheduler_config.async_scheduling
            or vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError("P-only VPP: MTP/LoRA/async/prefix must remain off")
        super().__init__(vllm_config, device)
        self.vpp_rank = get_pp_group().rank_in_group
        self.vpp_rank_local = bool(vllm_config.additional_config.get("qwen_infra_vpp_rank_local", False))
        self.is_last_pp_rank = self.vpp_rank == 0
        self.pp_handler.is_last_rank = self.is_last_pp_rank
        self.pp_handler.last_rank = get_pp_group().ranks[0]
        self.pp_handler.queue = deque([] if self.is_last_pp_rank else [None] * 4)
        self.vpp_contexts = {}
        self.vpp_active = False
        self.vpp_sampling = False
        self.vpp_warmup = False
        self.vpp_prepared = None
        self.vpp_counters = dict(first=0, returned=0, advanced=0, sampled=0, peak_live=0)
        self.vpp_modes = {"first": {}, "return": {}}
        self.vpp_scale_writers = None
        self.vpp_direct_inputs = bool(
            vllm_config.additional_config.get("qwen_infra_vpp_direct_inputs", False)
        )
        if self.vpp_direct_inputs and (
            not self.vpp_rank_local
            or vllm_config.additional_config.get("qwen_infra_vpp_transport")
            not in ("fixed_nccl", "segment_dma")
        ):
            raise ValueError("Direct inputs require rank-local scheduling and leased fixed slots")
        self.vpp_received_input = None
        self.vpp_input_stats = dict(direct_calls=0, staged_calls=0, logical_copy_bytes_avoided=0)
        self.vpp_direct_outputs = bool(
            vllm_config.additional_config.get("qwen_infra_vpp_direct_outputs", False)
        )
        if self.vpp_direct_outputs and (
            not self.vpp_rank_local
            or vllm_config.additional_config.get("qwen_infra_vpp_transport") != "segment_dma"
        ):
            raise ValueError("Output publication requires rank-local segmented DMA")
        self.vpp_output_transport = self.vpp_output_request = None
        self.vpp_output_prime = None
        if vllm_config.additional_config.get("qwen_infra_vpp_static_scales", False):
            from qwen_infra.vpp.static_scale import StaticScaleWriters

            self.vpp_scale_writers = StaticScaleWriters()
            self.vpp_scale_writers.set_enabled(True)

    def invalidate_vpp_scales(self):
        owner = self.vpp_scale_writers
        if owner is not None:
            if self.vpp_contexts or self.vpp_active or self.vpp_sampling:
                raise RuntimeError("Static scales may change only with a drained runner")
            owner.invalidate()

    def refresh_vpp_scales(self):
        if self.vpp_scale_writers is not None:
            self.vpp_scale_writers.refresh(self.model)

    def initialize_kv_cache(self, *args, **kwargs):
        self.invalidate_vpp_scales()
        super().initialize_kv_cache(*args, **kwargs)
        manager = self.cudagraph_manager
        if manager is not None:
            if manager.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                raise ValueError("VPP requires NONE or PIECEWISE, not a combined-stage FULL graph")
            if manager.use_breakable_cg or self.ubatch_runner is not None:
                raise ValueError("VPP uses native per-stage compile, not breakable graphs or DBO")
            manager.is_last_pp_rank = self.vpp_rank == 0
        self.refresh_vpp_scales()

    def count_mode(self, phase, mode):
        counts = self.vpp_modes[phase]
        counts[mode.name] = counts.get(mode.name, 0) + 1

    def load_model(self, *args, **kwargs):
        self.invalidate_vpp_scales()
        super().load_model(*args, **kwargs)
        self.vpp_backbone = find_backbone(self.model)
        if self.vpp_direct_outputs:
            self.vpp_backbone.initialize_bound_stages(self.vllm_config)
        # Prefix caching is off in all comparison arms. Aligned state migration
        # needs per-virtual-stage state-frontier ownership, not a silent fallback.
        if getattr(self.model_state, "_align_mode", False):
            raise ValueError("VPP P-only requires native mamba_cache_mode=none")
        self.refresh_vpp_scales()

    def reload_weights(self, *args, **kwargs):
        self.invalidate_vpp_scales()
        result = super().reload_weights(*args, **kwargs)
        self.refresh_vpp_scales()
        return result

    @torch.inference_mode()
    def profile_run(self, randomize_inputs=False):
        super().profile_run(randomize_inputs=randomize_inputs)
        if self.vpp_direct_outputs:
            from qwen_infra.vpp.native.stages import ProfileOutputs

            prime = ProfileOutputs(self.vpp_output_transport.native, self.device, self.max_num_tokens)
            self.vpp_backbone.profile_outputs = prime
            try:
                # Native dummy preparation/context with attention skipped, just
                # as memory profiling. All ranks compile locally before the
                # native warmup RPC can enter the cross-rank message chain.
                self._dummy_run(
                    self.max_num_tokens, skip_attn=True, is_profile=True, randomize_inputs=randomize_inputs
                )
                self.vpp_output_prime = prime.close()
            finally:
                self.vpp_backbone.profile_outputs = None

    def add_requests(self, scheduler_output):
        for request in [] if self.vpp_warmup else scheduler_output.scheduled_new_reqs:
            params = request.sampling_params
            if (
                params is None
                or params.max_tokens != 1
                or params.prompt_logprobs is not None
                or params.structured_outputs is not None
            ):
                raise ValueError("VPP experiment accepts plain text P+1 requests only")
        # first() temporarily uses non-last output semantics, but the sampler
        # and its request parameters still belong to the global final rank0.
        previous = self.is_last_pp_rank
        self.is_last_pp_rank = self.vpp_rank == 0
        try:
            return super().add_requests(scheduler_output)
        finally:
            self.is_last_pp_rank = previous

    def prepare_attn(self, input_batch):
        result = super().prepare_attn(input_batch)
        if self.vpp_active:
            self.vpp_prepared = snapshot(result)
        return result

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
        batch = super().prepare_inputs(scheduler_output, batch_req_state, batch_desc, num_active_loras)
        if (
            self.vpp_direct_inputs
            and self.vpp_active
            and not self.vpp_warmup
            and self.vpp_received_input is not None
        ):
            direct = batch_desc.cg_mode == CUDAGraphMode.NONE
            if direct:
                # Keep native execute_model: its copy_(received[:n]) now has
                # identical source/destination views. Restore fixed PW inputs
                # in first() even if a later preparation/model step fails.
                self.intermediate_tensors = received_input_views(
                    self.vpp_received_input, batch.num_tokens_after_padding
                )
                self.vpp_input_stats["logical_copy_bytes_avoided"] += (
                    batch.num_tokens_after_padding * 5120 * 4
                )
            self.vpp_input_stats["direct_calls" if direct else "staged_calls"] += 1
        if getattr(self, "vpp_direct_outputs", False) and self.vpp_output_request is not None:
            self.prepare_boundary_output(*self.vpp_output_request, batch.num_tokens_after_padding)
        return batch

    def prepare_boundary_output(self, batch_id, stage, rows):
        self.vpp_backbone.output_binding = self.vpp_output_transport.prepare_send(batch_id, stage, rows)

    @torch.inference_mode()
    def first(self, batch_id, scheduler_output, intermediate):
        limit = 7 if self.vpp_rank_local else 6
        if batch_id in self.vpp_contexts or len(self.vpp_contexts) >= limit:
            raise RuntimeError("VPP in-flight ownership/buffer bound violated")
        self.vpp_active = True
        self.vpp_backbone.traversal = 0
        self.is_first_pp_rank = self.vpp_rank == 0
        self.is_last_pp_rank = False
        stable_inputs = self.intermediate_tensors
        self.vpp_received_input = intermediate
        self.vpp_output_request = (batch_id, self.vpp_rank) if self.vpp_rank < 3 else None
        self.vpp_backbone.direct_inputs_active = not self.vpp_warmup
        try:
            output = super().execute_model(scheduler_output, intermediate)
            if scheduler_output.total_num_scheduled_tokens == 0:
                return output
            state = self.execute_model_state
            saved_batch = snapshot(state.input_batch)
            if not self.vpp_warmup and compute_need_sampled_mask(saved_batch) is not None:
                raise RuntimeError("Decode was admitted into a P-only VPP pipeline")
            # FlashInfer wrapper plans are mutable. Retain input tables, rebuild
            # native metadata on return, do not shallow-copy a live wrapper.
            state = state._replace(input_batch=saved_batch, attn_metadata=None, slot_mappings_by_layer=None)
            # Native DefaultModelState.prepare_inputs returns M-RoPE positions.
            # Capture them now: recomputing on return would use the advanced
            # GPU token frontier (and a later chunk's positions).
            rope = self.model_state.rope_state
            extra = (
                {}
                if rope is None
                else {"positions": snapshot(rope.get_positions(saved_batch.num_tokens_after_padding))}
            )
            extra["vpp_mode"] = self.vpp_backbone.last_runtime_mode
            self.count_mode("first", extra["vpp_mode"])
            self.vpp_contexts[batch_id] = (state, self.vpp_prepared, extra)
            self.execute_model_state = None
            super().postprocess_num_computed_tokens(saved_batch)
            self.model_state.postprocess_state(saved_batch.idx_mapping, 0)
            self.vpp_counters["first"] += 1
            self.vpp_counters["advanced"] += 1
            self.vpp_counters["peak_live"] = max(self.vpp_counters["peak_live"], len(self.vpp_contexts))
            return output
        finally:
            self.intermediate_tensors = stable_inputs
            self.vpp_received_input = None
            self.vpp_backbone.direct_inputs_active = False
            self.vpp_backbone.output_binding = None
            self.vpp_output_request = None
            self.vpp_active = False
            self.vpp_prepared = None
            self.vpp_backbone.traversal = None
            self.is_first_pp_rank = self.vpp_rank == 0
            self.is_last_pp_rank = self.vpp_rank == 0

    @torch.inference_mode()
    def returning(self, batch_id, intermediate):
        state, (block_tables, slot_mappings), extra = self.vpp_contexts[batch_id]
        batch = state.input_batch
        extra = dict(extra)
        mode = extra.pop("vpp_mode")
        if mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            raise RuntimeError("Unexpected combined-stage CUDA Graph replay")
        metadata = self.model_state.prepare_attn(
            batch, mode, block_tables, slot_mappings, self.attn_groups, self.kv_cache_config
        )
        slots = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
        padding = self.input_buffers.is_padding[: batch.num_tokens_after_padding]
        padding.copy_(batch.is_padding)
        self.vpp_backbone.traversal = 1
        self.vpp_backbone.direct_inputs_active = not self.vpp_warmup
        try:
            if getattr(self, "vpp_direct_outputs", False) and self.vpp_rank:
                self.prepare_boundary_output(batch_id, 7 - self.vpp_rank, batch.num_tokens_after_padding)
            with set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=batch.num_tokens_after_padding,
                cudagraph_runtime_mode=mode,
                batch_descriptor=BatchDescriptor(num_tokens=batch.num_tokens_after_padding),
                slot_mapping=slots,
                is_padding=padding,
            ):
                model_inputs = dict(
                    input_ids=None,
                    positions=batch.positions,
                    inputs_embeds=None,
                    intermediate_tensors=intermediate,
                )
                model_inputs.update(extra)
                result = self.model(**model_inputs)
        finally:
            self.vpp_backbone.traversal = None
            self.vpp_backbone.direct_inputs_active = False
            self.vpp_backbone.output_binding = None
        self.vpp_contexts[batch_id] = (
            state._replace(
                attn_metadata=metadata,
                slot_mappings_by_layer=slots,
                hidden_states=result if self.vpp_rank == 0 else None,
            ),
            None,
            None,
        )
        self.vpp_counters["returned"] += 1
        self.count_mode("return", mode)
        return result

    def finish(self, batch_id):
        self.execute_model_state, _, _ = self.vpp_contexts.pop(batch_id)
        self.is_first_pp_rank = self.vpp_rank == 0
        self.is_last_pp_rank = self.vpp_rank == 0

    def postprocess_num_computed_tokens(self, input_batch):
        if not self.vpp_sampling:
            super().postprocess_num_computed_tokens(input_batch)

    def postprocess_sampled(
        self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None
    ):
        # Native post_update treats query_start_loc=None as a zero increment.
        return super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            None if self.vpp_sampling else query_start_loc,
        )

    def sample_tokens(self, grammar_output):
        self.vpp_sampling = True
        try:
            result = super().sample_tokens(grammar_output)
            self.vpp_counters["sampled"] += 1
            return result
        finally:
            self.vpp_sampling = False
