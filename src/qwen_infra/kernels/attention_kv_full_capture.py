"""Cold extra capture inside the native manager's measured lifecycle."""

from contextlib import ExitStack
import time


def current_inputs(runner, *, rows=16):
    inputs = model_inputs(runner, rows=rows)
    return tuple(inputs[name] for name in sorted(inputs) if inputs[name] is not None) + (
        runner.input_buffers.is_padding[:rows],
    )


def model_inputs(runner, *, rows=16):
    # Same override order as the native capture factory (M-RoPE positions).
    return {
        "input_ids": runner.input_buffers.input_ids[:rows],
        "positions": runner.input_buffers.positions[:rows],
        **runner.model_state.prepare_dummy_inputs(rows, rows),
    }


def capture_entry(entry):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.distributed.parallel_state import graph_capture
    from vllm.distributed.device_communicators import pynccl_allocator
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.offloader.base import get_offloader
    from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture

    runner, manager = entry.runner, entry.manager
    rows = entry.rows
    if (
        entry.graph is not None
        or manager.use_breakable_cg
        or runner.pcp_manager is not None
        or runner.ubatch_runner is not None
        or manager.use_aux_hidden_state_outputs
        or not manager.is_first_pp_rank
        or not manager.is_last_pp_rank
        or runner.max_model_len > entry.capacity
        or manager.hidden_states is None
    ):
        raise ValueError("Expected one original TP4 MTP-off FULL output/capture lifecycle")
    descriptor = next(
        d
        for d in manager._capture_descs[CUDAGraphMode.FULL]
        if d.num_tokens == d.num_reqs == rows
        and d.uniform_token_count == 1
        and d.num_active_loras == 0
        and d.num_ubatches == 1
    )
    native_graphs = dict(manager.graphs)
    native_samples = tuple(manager._capture_mem_samples or ())
    output = manager.hidden_states
    saved_pool = pynccl_allocator._graph_pool_id
    offloader = get_offloader()
    torch.cuda.synchronize()
    free_before = torch.cuda.mem_get_info()[0]
    allocated_before, reserved_before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    start = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    fusion_counters = getattr(entry, "cold_fusion_counters", None)
    fusion_before = fusion_counters() if fusion_counters is not None else {}
    try:
        with graph_capture(device=runner.device) as gc:
            for warmup in (True, False):
                inputs = model_inputs(runner, rows=rows)
                metadata, slots = prepare_inputs_to_capture(
                    rows,
                    rows,
                    runner.model_state,
                    runner.input_buffers,
                    runner.block_tables,
                    runner.attn_groups,
                    runner.kv_cache_config,
                    full_cudagraph=True,
                    max_query_len=1,
                )
                plans = entry.prepare_plans(metadata)

                def forward():
                    entry.prepare_forward()
                    with set_forward_context(
                        metadata,
                        runner.vllm_config,
                        num_tokens=rows,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        slot_mapping=slots,
                        is_padding=runner.input_buffers.is_padding[:rows],
                    ):
                        result = runner.model(**inputs)
                    output[:rows].copy_(result)

                with ExitStack() as permits:
                    for adapter, plan in zip(entry.adapters, plans, strict=True):
                        permits.enter_context(adapter.capture(plan))
                    if warmup:
                        forward()
                    else:
                        offloader.sync_prev_onload()
                        pynccl_allocator.set_graph_pool_id(manager.pool)
                        with torch.cuda.graph(graph, pool=manager.pool, stream=gc.stream):
                            forward()
                            offloader.join_after_forward()
    finally:
        pynccl_allocator.set_graph_pool_id(saved_pool)
    torch.cuda.synchronize()
    if (
        manager.graphs != native_graphs
        or manager.hidden_states is not output
        or tuple(manager._capture_mem_samples or ()) != native_samples
    ):
        raise RuntimeError("Extra capture changed native graph dictionary/output owner")
    receipt = dict(
        extra_entries=1,
        shared_native_pool=True,
        native_graph_dictionary_preserved=True,
        original_output_owner=True,
        descriptor=repr(descriptor),
        **entry.plan_receipt(plans),
        dummy_null_slots=True,
        elapsed_seconds=time.perf_counter() - start,
        driver_extra_free_delta_bytes=free_before - torch.cuda.mem_get_info()[0],
        torch_extra_allocated_delta_bytes=torch.cuda.memory_allocated() - allocated_before,
        torch_extra_reserved_delta_bytes=torch.cuda.memory_reserved() - reserved_before,
        memory_inside_native_capture_measurement=True,
        native_sample_list_unmodified=True,
        native_full_memory_samples=list(native_samples),
        native_full_descriptors=len(manager._capture_descs[CUDAGraphMode.FULL]),
        native_graph_keys=[repr(d) for d in manager.graphs],
        pool_state_restored=pynccl_allocator._graph_pool_id == saved_pool,
    )
    if fusion_counters is not None:
        fusion_after = fusion_counters()
        receipt["attention_fusion_warmup_capture_deltas"] = {
            name: {key: counts[key] - fusion_before[name][key] for key in counts}
            for name, counts in fusion_after.items()
        }
        receipt["attention_fusion_witness_boundary"] = (
            "combined cold warmup plus graph capture; no graph replay Python counters"
        )
    entry.bind(graph, descriptor, metadata, plans, current_inputs(runner, rows=rows), receipt)
