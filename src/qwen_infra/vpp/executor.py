# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Queue continuation is adapted from Huawei VPP c409752 patch_vpp_core.py;
# native a9fe30e9 iteration/error/abort hooks are retained.
"""Opt-in P-only native executor with author-style continuation overlap."""

from collections import deque
import itertools

import vllm.envs as envs
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from qwen_infra.vpp.layout import VppContinuation, VppStep


def step_with_vpp_queue(self):
    if not isinstance(self.model_executor, FoldedExecutor):
        return _native_step(self)
    if not getattr(self, "_vpp_queue_initialized", False):
        if self.batch_queue:
            raise RuntimeError("Cannot replace an occupied native batch queue")
        self.batch_queue_size = 7 if getattr(self.model_executor, "vpp_rank_local", False) else 5
        # Two traversals: queued return futures coexist with the next W first
        # futures. This host bound is NOT the number of live GPU contexts:
        # FIFO RPCs issue each named finish before its replacement admission.
        # W=5 on the historical route; the rank-local topology uses W=7.
        self.batch_queue = deque(maxlen=2 * self.batch_queue_size)
        self._vpp_queue_initialized = True
        self._vpp_allow_overlap = False
    queue = self.batch_queue
    executed = False
    can_schedule = len(queue) < self.batch_queue_size or self._vpp_allow_overlap
    self._vpp_allow_overlap = False
    if self.scheduler.has_requests() and can_schedule:
        if len(queue) >= queue.maxlen:
            raise RuntimeError("VPP host queue bound exceeded; refusing silent eviction")
        schedule = self.scheduler.schedule(self._should_throttle_prefills())
        if schedule.pending_structured_output_tokens:
            raise ValueError("P-only VPP does not support structured-output deferral")
        with self.log_error_detail(schedule):
            future = self.model_executor.execute_model(schedule, non_block=True)
        queue.appendleft((future, schedule, future))
        executed = schedule.total_num_scheduled_tokens > 0
        if (
            executed
            and len(queue) < self.batch_queue_size
            and (getattr(self.model_executor, "vpp_rank_local", False) or not queue[-1][0].done())
        ):
            return None, executed
    if not queue:
        return None, False
    future, schedule, execute_future = queue.pop()
    with self.capture_iteration_details(schedule) as details, self.log_error_detail(schedule):
        output = future.result()
        if isinstance(output, VppContinuation):
            execute_future = self.model_executor.return_model(schedule, output.batch_id, non_block=True)
            # Rank-local return includes native sampling in the same worker
            # call. Avoid an extra None response in the bounded native MQ.
            future = (
                execute_future
                if getattr(self.model_executor, "vpp_rank_local", False)
                else self.model_executor.sample_tokens(None, non_block=True)
            )
            queue.appendleft((future, schedule, execute_future))
            self._vpp_allow_overlap = True
            return None, executed
        if output is None:
            execute_future.result()
            raise RuntimeError("VPP worker did not produce a completed model output")
    self._process_aborts_queue()
    outputs = self.scheduler.update_from_output(schedule, output)
    self._attach_iteration_details(outputs, details)
    self._attach_weight_version(outputs)
    return outputs, executed


_native_step = None


class FoldedExecutor(MultiprocExecutor):
    supports_pp = True

    def __init__(self, *args, **kwargs):
        from vllm.v1.engine.core import EngineCore

        global _native_step
        if _native_step is None:
            _native_step = EngineCore.step_with_batch_queue
            EngineCore.step_with_batch_queue = step_with_vpp_queue
        self.vpp_batch_ids = itertools.count()
        super().__init__(*args, **kwargs)
        self.vpp_rank_local = bool(self.vllm_config.additional_config.get("qwen_infra_vpp_rank_local", False))

    def _get_output_rank(self):
        return 0

    def _step(self, step, non_block):
        return self.collective_rpc(
            "execute_model",
            args=(step,),
            unique_reply_rank=0,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def execute_model(self, scheduler_output, non_block=False):
        return self._step(VppStep(scheduler_output, next(self.vpp_batch_ids)), non_block)

    def return_model(self, scheduler_output, batch_id, non_block=False):
        step = VppStep(scheduler_output, batch_id, 1)
        if self.vpp_rank_local:
            return self.collective_rpc(
                "vpp_return_and_sample",
                args=(step,),
                unique_reply_rank=0,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            )
        return self._step(step, non_block)
