# SPDX-License-Identifier: Apache-2.0
"""Native worker lifecycle with bounded, two-direction VPP transfers."""

from collections import deque
from contextlib import nullcontext

import torch
from vllm.distributed.parallel_state import get_pp_group, init_model_parallel_group
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_worker import Worker

from qwen_infra.vpp.layout import (
    RankLocalOrder,
    VppContinuation,
    VppStep,
    profile_annotation,
)
from qwen_infra.vpp.runner import FoldedRunner


class FoldedWorker(Worker):
    def stage_profile(self, schedule, batch_id, traversal, kind):
        if self.profiler is None or not self.profiler.should_annotate:
            return nullcontext()
        return self.profiler.annotate_context_manager(
            profile_annotation(schedule, self.model_runner.vpp_rank, traversal, batch_id, kind)
        )

    def compile_or_warm_up_model(self):
        # Native V2 JIT warmup calls worker callbacks directly, without the
        # executor continuation queue. Run both traversals before it samples.
        runner = self.model_runner
        runner.vpp_warmup = True
        self.vpp_warmup_batch = -1
        runner.pp_handler.queue = deque([] if runner.vpp_rank == 0 else [None])
        try:
            result = super().compile_or_warm_up_model()
            if runner.vpp_contexts:
                raise RuntimeError("Native warmup left a VPP continuation live")
            self.vpp_warmup_receipt = dict(runner.vpp_counters)
            for key in runner.vpp_counters:
                runner.vpp_counters[key] = 0
            runner.vpp_modes = {"first": {}, "return": {}}
            return result
        finally:
            runner.vpp_warmup = False
            runner.pp_handler.queue = deque([] if runner.vpp_rank == 0 else [None] * 4)

    def _make_model_runner(self):
        if not self.use_v2_model_runner:
            raise ValueError("VPP adapter targets the installed native V2 runner")
        transport = self.vllm_config.additional_config.get("qwen_infra_vpp_transport", "native")
        if transport not in ("native", "fixed_nccl", "segment_dma"):
            raise ValueError("Unknown VPP boundary transport")
        self.vpp_transport = None
        pp = get_pp_group()
        # Separate ordered communicators avoid a return send blocking the next
        # forward receive. This uses native NCCL transport, not hardware P2P.
        self.vpp_return_group = init_model_parallel_group(
            [pp.ranks],
            self.local_rank,
            "nccl",
            group_name="qwen_infra_vpp_return",
            use_device_communicator=False,
        )
        self.vpp_sends = []
        runner = FoldedRunner(self.vllm_config, self.device)
        self.vpp_order = RankLocalOrder(runner.vpp_rank)
        self.vpp_schedules = {}
        if transport in ("fixed_nccl", "segment_dma"):
            if not runner.vpp_rank_local:
                raise ValueError("Fixed transport uses the existing W7 rank-local schedule")
            from qwen_infra.vpp.native.stages import StageBoundaries

            self.vpp_transport = StageBoundaries(
                pp,
                self.device,
                self.vllm_config.scheduler_config.max_num_batched_tokens,
                backend=transport,
                direct_outputs=runner.vpp_direct_outputs,
                resource_layout=self.vllm_config.additional_config.get("qwen_infra_vpp_resource_layout"),
            )
            runner.vpp_output_transport = self.vpp_transport
        return runner

    def _send(self, group, intermediate, dst, batch, stage):
        if self.vpp_transport is not None:
            self.vpp_transport.send(batch, stage, intermediate.tensors)
            return
        # Own the boundary until NCCL finishes; graph outputs can alias buffers
        # reused by later forwards (same issue documented by SGLang PP).
        payload = {name: tensor.clone() for name, tensor in intermediate.tensors.items()}
        handles = group.isend_tensor_dict(payload, dst=dst)
        self.vpp_sends.append((handles, payload))
        self.vpp_sends[:] = [
            (work, data) for work, data in self.vpp_sends if not all(handle.is_completed() for handle in work)
        ]
        limit = 14 if self.model_runner.vpp_rank_local else 12
        if len(self.vpp_sends) > limit:
            raise RuntimeError("VPP pending send ownership exceeds bounded pipeline")

    def _recv(self, group, src, batch, stage):
        if self.vpp_transport is not None:
            return IntermediateTensors(self.vpp_transport.receive(batch, stage))
        payload, handles, callbacks = group.irecv_tensor_dict(src=src)
        for handle in handles:
            handle.wait()
        for callback in callbacks:
            callback()
        return IntermediateTensors(payload)

    def _retire_boundary(self, batch, stage):
        transport = getattr(self, "vpp_transport", None)
        if transport is not None:
            transport.retire(batch, stage)

    def _rank_local_advance(self, draining=False):
        op = self.vpp_order.next(draining=draining)
        if op is not None:
            self._run_local_stage(*op)
        return op

    def _rank_local_step(self, step):
        batch = step.batch_id
        if step.traversal == 0:
            self.vpp_order.admit(batch)
            self.vpp_schedules[batch] = step.scheduler_output
            self._rank_local_advance()
            if self.vpp_order.rank == 3:
                self._rank_local_advance()  # stages3/4 share one device
            return VppContinuation(batch)
        self._rank_local_advance(draining=True)
        while batch not in self.vpp_order.returned:
            if self._rank_local_advance(draining=True) is None:
                raise RuntimeError("Missing VPP stage while draining a named batch")
        self.model_runner.finish(batch)
        self.vpp_order.release(batch)
        del self.vpp_schedules[batch]
        return None  # Native sampling must consume this named batch next.

    def vpp_return_and_sample(self, step):
        """One native response for rank-local finish + sample; no output copy."""
        runner = self.model_runner
        if (
            not isinstance(step, VppStep)
            or step.traversal != 1
            or not runner.vpp_rank_local
            or runner.vpp_warmup
            or not step.scheduler_output.total_num_scheduled_tokens
        ):
            raise ValueError("Combined return requires a nonempty rank-local P step")
        self.execute_model(step)
        # Keep Worker.sample_tokens decorators and WorkerProc AsyncOutput
        # handling. In particular, do not call get_output() in this adapter.
        return self.sample_tokens(None)

    def _run_local_stage(self, batch, stage):
        runner = self.model_runner
        rank, pp = runner.vpp_rank, get_pp_group()
        traversal = int(stage != rank)
        schedule = self.vpp_schedules[batch]
        with self.annotate_profile(schedule):
            with self.stage_profile(schedule, batch, traversal, "recv"):
                if traversal == 0:
                    intermediate = self._recv(pp, rank - 1, batch, rank - 1) if rank else None
                elif rank == 3:
                    intermediate = self.vpp_fold_output
                    self.vpp_fold_output = None
                else:
                    intermediate = self._recv(self.vpp_return_group, rank + 1, batch, stage - 1)
            with self.stage_profile(schedule, batch, traversal, "compute"):
                output = (
                    runner.first(batch, schedule, intermediate)
                    if traversal == 0
                    else runner.returning(batch, intermediate)
                )
            if traversal == 0 and rank == 3:
                self.vpp_fold_output = output  # consumed immediately; no clone/wire
            elif traversal == 0 or rank:
                group, dst = (pp, rank + 1) if traversal == 0 else (self.vpp_return_group, rank - 1)
                with self.stage_profile(schedule, batch, traversal, "send"):
                    self._send(group, output, dst, batch, stage)
            self._retire_boundary(batch, stage)

    def execute_model(self, step):
        if not isinstance(step, VppStep):
            if not self.model_runner.vpp_warmup:
                raise TypeError("FoldedWorker requires an explicit VPP step")
            if step.total_num_scheduled_tokens == 0:
                return self.model_runner.execute_model(step)
            batch_id = self.vpp_warmup_batch
            self.vpp_warmup_batch -= 1
            self.execute_model(VppStep(step, batch_id, 0))
            return self.execute_model(VppStep(step, batch_id, 1))
        schedule, batch_id = step.scheduler_output, step.batch_id
        runner = self.model_runner
        transport = getattr(self, "vpp_transport", None)
        if transport is not None and step.traversal == 0 and schedule.total_num_scheduled_tokens:
            transport.admit(batch_id)
        if (
            getattr(runner, "vpp_rank_local", False)
            and not runner.vpp_warmup
            and schedule.total_num_scheduled_tokens
        ):
            return self._rank_local_step(step)
        rank, pp = runner.vpp_rank, get_pp_group()
        with self.annotate_profile(schedule):
            if schedule.total_num_scheduled_tokens == 0:
                return runner.execute_model(schedule)
            if step.traversal == 0:
                with torch.profiler.record_function(f"VPP first batch{batch_id} rank{rank}"):
                    with self.stage_profile(schedule, batch_id, 0, "recv"):
                        intermediate = self._recv(pp, rank - 1, batch_id, rank - 1) if rank else None
                    with self.stage_profile(schedule, batch_id, 0, "compute"):
                        output = runner.first(batch_id, schedule, intermediate)
                    if rank < 3:
                        with self.stage_profile(schedule, batch_id, 0, "send"):
                            self._send(pp, output, rank + 1, batch_id, rank)
                        self._retire_boundary(batch_id, rank)
                    else:
                        # Local fold: no wire transfer between stages3 and4.
                        with self.stage_profile(schedule, batch_id, 1, "compute"):
                            output = runner.returning(batch_id, output)
                        with self.stage_profile(schedule, batch_id, 1, "send"):
                            self._send(self.vpp_return_group, output, rank - 1, batch_id, 4)
                        self._retire_boundary(batch_id, 4)
                return VppContinuation(batch_id)
            with torch.profiler.record_function(f"VPP return batch{batch_id} rank{rank}"):
                if rank < 3:
                    with self.stage_profile(schedule, batch_id, 1, "recv"):
                        intermediate = self._recv(self.vpp_return_group, rank + 1, batch_id, 6 - rank)
                    with self.stage_profile(schedule, batch_id, 1, "compute"):
                        output = runner.returning(batch_id, intermediate)
                    if rank:
                        with self.stage_profile(schedule, batch_id, 1, "send"):
                            self._send(self.vpp_return_group, output, rank - 1, batch_id, 7 - rank)
                    self._retire_boundary(batch_id, 7 - rank)
                runner.finish(batch_id)
            return None

    def qwen_infra_vpp_close_transport(self):
        """Cold all-rank RPC while native workers/groups are still alive."""
        transport = self.vpp_transport
        if transport is None:
            return dict(backend="native", closed=False)
        if self.model_runner.vpp_contexts or self.vpp_order.live:
            raise RuntimeError("Finish all VPP batches before closing transport")
        if not transport.closed:
            transport.close()
        return dict(
            backend=transport.backend,
            closed=transport.closed,
            progress_stats=transport.progress_stats,
            live_leases=len(transport.leases),
            live_producers=len(transport.producer_leases),
            live_slots={str(stage): pool.live for stage, (_, pool, _) in transport.endpoints.items()},
            resources=transport.receipt(),
        )

    def qwen_infra_vpp_resources(self):
        """Observe resources while loaded; native model/cache bytes stay separate."""
        transport = self.vpp_transport
        return dict(
            rank=self.model_runner.vpp_rank,
            transport=transport.receipt() if transport is not None else None,
            cuda=dict(allocated=torch.cuda.memory_allocated(self.device),
                      reserved=torch.cuda.memory_reserved(self.device),
                      peak_allocated=torch.cuda.max_memory_allocated(self.device),
                      peak_reserved=torch.cuda.max_memory_reserved(self.device)),
            kv_cache=dict(configured_bytes=self.vllm_config.cache_config.kv_cache_memory_bytes,
                          num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks,
                          block_size=self.vllm_config.cache_config.block_size),
        )

    def shutdown(self):
        # Do not wait on an unmatched send after a worker failure; native worker
        # shutdown/supervisor owns process-group teardown and the finite timeout.
        try:
            transport = getattr(self, "vpp_transport", None)
            if transport is not None and not transport.closed:
                transport.close()
        finally:
            group = getattr(self, "vpp_return_group", None)
            if group is not None:
                group.destroy()
                self.vpp_return_group = None
            super().shutdown()
