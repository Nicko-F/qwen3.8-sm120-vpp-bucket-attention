# SPDX-License-Identifier: Apache-2.0
"""Thin native-worker adapter; C++ owns payload slots, streams and completion."""

import time

import torch
import torch.distributed as dist
from qwen_infra.config import host_reserve_bytes

from . import load
from .control import BoundaryControl


class ProfileOutputs:
    """Cold rank-local publication/retirement, before any peer starts waiting.

    Reuse the existing real slot/capability and loopback APIs. These two private
    one-slot pools are not attached to an edge; no header or DMA job is emitted.
    A single pair is reused sequentially across the rank's output-bound stages.
    """

    def __init__(self, native, device, capacity):
        self.native, self.device, self.capacity = native, device, capacity
        self.sequence, self.active = 0, None
        self.source = native.DeviceSlotPool(native.SlotRole.Source, capacity, 1, 1, 0, device.index)
        self.destination = native.DeviceSlotPool(native.SlotRole.Destination, capacity, 1, 1, 0, device.index)
        self.payload_bytes = self.source.payload_bytes + self.destination.payload_bytes

    def prepare(self, rows):
        if self.active is not None:
            raise RuntimeError("Finish the previous cold producer before reusing its slot")
        wire = self.native.boundary(self.capacity, 1, 1, -1, self.sequence, 0, rows)
        source, destination = self.source.reserve(wire), self.destination.reserve(wire)
        if source is None or destination is None:
            raise RuntimeError("Cold publication slots were not retired")
        views = self.source.producer_views(source)
        token = self.source.publication_token(source)
        self.active = source, destination, views, token
        return *views, token

    def finish(self):
        if self.active is None:
            raise RuntimeError("No active cold producer")
        source, destination, _, _ = self.active
        self.source.publish_written(source)
        self.source.loopback(source, self.destination, destination)
        self.destination.discard_received(destination)
        torch.cuda.synchronize(self.device)  # Cold setup only, never a timed stage.
        self.source.progress()
        self.destination.progress()
        if self.source.live or self.destination.live:
            raise RuntimeError("Cold publication did not drain")
        self.active = None
        self.sequence += 1

    def close(self):
        if self.active is not None:
            raise RuntimeError("Cannot close an unfinished cold producer")
        self.source.close()
        self.destination.close()
        return dict(
            calls=self.sequence,
            payload_bytes=self.payload_bytes,
            cross_rank_headers=0,
            cross_rank_jobs=0,
            closed=True,
        )


class StageBoundaries:
    def __init__(self, pp, device, capacity, *, backend="fixed_nccl", direct_outputs=False, resource_layout=None):
        if len(pp.ranks) != 4 or not 0 < capacity <= 16384:
            raise ValueError("Expected four ranks and a boundary token capacity within 1..16384")
        if backend not in ("fixed_nccl", "segment_dma"):
            raise ValueError("Unknown fixed-boundary backend")
        if type(direct_outputs) is not bool or (direct_outputs and backend != "segment_dma"):
            raise ValueError("Independent output publication requires segmented DMA")
        resource_layout = resource_layout or ("rank" if backend == "segment_dma" else "edge")
        if resource_layout not in ("rank", "edge") or resource_layout == "rank" and backend != "segment_dma":
            raise ValueError("Rank-shared resources require segmented DMA")
        self.native = load()
        self.rank, self.group = pp.rank_in_group, pp.cpu_group
        self.ranks, self.backend = pp.ranks, backend
        self.endpoints, self.leases = {}, {}
        self.failed = self.closed = False
        self.capacity = capacity
        self.minimum_available_bytes = host_reserve_bytes()
        self.direct_outputs = direct_outputs
        self.resource_layout = resource_layout
        self.arenas, self.prefetched = {}, {}
        self.close_receipt = None
        self.credit_waits = {role: dict(calls=0, milliseconds=0) for role in ("source", "destination")}
        self.prefetch_stats = {direction: dict(issued=0, consumed=0) for direction in ("forward", "return")}
        self.peak_prefetched = 0
        self.producer_leases, self.output_stats = {}, {}
        self.engine = self.native.DmaProgress(device.index) if backend == "segment_dma" else None
        self.streams = [torch.cuda.Stream(device=device)] if self.engine is not None else []
        if resource_layout == "rank":
            self.streams.append(torch.cuda.Stream(device=device))
            for role, slots in (("source", 2), ("destination", 3)):
                native_role = self.native.SlotRole.Source if role == "source" else self.native.SlotRole.Destination
                self.arenas[role] = self.native.DeviceBufferArena(native_role, capacity, slots, device.index)
        self.host_receipts, self.progress_stats = {}, None
        self.record_wait_ns = self.record_wait_calls = 0
        rank_of = lambda s: s if s < 4 else 7 - s
        try:
            # Every rank bootstraps the same edge order. No per-step objects or
            # metadata broadcasts; the 128-byte UID exchange is cold only.
            for stage in (0, 1, 2, 4, 5, 6):
                src, dst = rank_of(stage), rank_of(stage + 1)
                source = self.rank == src
                pool = None
                if self.rank in (src, dst):
                    role = self.native.SlotRole.Source if source else self.native.SlotRole.Destination
                    pool = self.native.DeviceSlotPool(
                        role, capacity, 7, 1, stage, device.index,
                        self.arenas.get("source" if source else "destination"),
                    )
                if self.engine is not None:
                    edge = self._make_dma(stage, src, dst, pool, device)
                else:
                    uid = bytearray(self.native.nccl_unique_id() if source else bytes(128))
                    dist.broadcast(
                        torch.frombuffer(uid, dtype=torch.uint8), src=pp.ranks[src], group=self.group
                    )
                    edge = self.native.NcclEdge(pool, bytes(uid)) if pool is not None else None
                if pool is None:
                    continue
                control = BoundaryControl(
                    self.native,
                    self.group,
                    pp.ranks,
                    source=source,
                    stage=stage,
                    capacity=capacity,
                    slots=7,
                    epoch=1,
                    timeout_seconds=30,
                )
                self.endpoints[stage] = (control, pool, edge)
                if source:
                    self.output_stats[str(stage)] = dict(
                        direct_calls=0, staged_calls=0, logical_copy_bytes_avoided=0
                    )
            if self.engine is not None:
                dist.barrier(group=self.group)
                self.engine.start()
        except BaseException:
            self.abort()
            raise

    def _make_dma(self, stage, src, dst, pool, device):
        """Cold fixed-size Gloo identity/event exchange; no payload serialization."""
        source = self.rank == src
        region = (
            self.native.HostRegion.create_dma(
                self.capacity, 7, self.native.current_numa_node(), 16 * 2**20, self.minimum_available_bytes
            )
            if source
            else None
        )
        # Initialize the shared control page before publishing its identity.
        # A receiver may leave broadcast first; it must not attach to zeros.
        edge = (
            self.native.DmaEdge(
                pool, region, self.streams[0].cuda_stream, device.index, self.engine, 16 * 2**20
            )
            if source
            else None
        )
        handle = torch.tensor(list(region.handle) if source else [0] * 7, dtype=torch.int64)
        dist.broadcast(handle, src=self.ranks[src], group=self.group)
        if pool is not None and not source:
            region = self.native.HostRegion.attach(handle.tolist())
            if self.resource_layout == "edge":
                self.streams.append(torch.cuda.Stream(device=device))
            edge = self.native.DmaEdge(
                pool, region, self.streams[1 if self.resource_layout == "rank" else -1].cuda_stream,
                device.index, self.engine, 16 * 2**20
            )
        events = {}
        for owner in (src, dst):
            event = bytearray(edge.export_event() if self.rank == owner else bytes(8 * 64))
            if len(event) != 8 * 64:
                raise RuntimeError("Expected eight CUDA IPC event handles")
            dist.broadcast(
                torch.frombuffer(event, dtype=torch.uint8), src=self.ranks[owner], group=self.group
            )
            events[owner] = bytes(event)
        if edge is not None:
            edge.connect(events[dst if source else src])
            self.host_receipts[str(stage)] = dict(
                source=source, handle=handle.tolist(), bytes=edge.host_bytes, numa_receipt=edge.numa_receipt
            )
        return edge

    def _check(self):
        if self.failed or self.closed:
            raise RuntimeError("Stopped native VPP transport epoch")

    def admit(self, batch):
        self._check()
        for control, _, _ in self.endpoints.values():
            if not control.source:
                control.post(batch)  # CPU-only; no speculative NCCL receive.
        self.prefetch()

    @staticmethod
    def _direction(stage):
        return "forward" if stage < 3 else "return"

    def prefetch(self, priority_stage=None, *, exclude_stage=None):
        """Issue at most one unconsumed FIFO input per incoming edge.

        Two speculative inputs can occupy the three-slot arena. A third slot
        serves the demanded input/previous consumer; no speculative CUDA wait
        is submitted before the source publishes this generation's READY.
        """
        self._check()
        if self.resource_layout != "rank":
            return
        self.engine.progress()
        stages = sorted(self.endpoints, key=lambda stage: stage != priority_stage)
        for stage in stages:
            control, pool, edge = self.endpoints[stage]
            if control.source or stage == exclude_stage or stage in self.prefetched or len(self.prefetched) >= 2:
                continue
            wire = edge.peek_header()
            if wire is None:
                continue
            batch = control.peek_admitted(wire)
            if batch is None:
                continue
            ticket = pool.reserve(wire)
            if ticket is None:
                continue
            edge.submit(ticket, self.direct_outputs)
            self.prefetched[stage] = batch, ticket, wire
            self.prefetch_stats[self._direction(stage)]["issued"] += 1
            self.peak_prefetched = max(self.peak_prefetched, len(self.prefetched))

    def _reserve(self, stage, wire):
        control, pool, edge = self.endpoints[stage]
        if self.engine is None:
            return edge.reserve(wire, 30000)
        # A rank-level credit can be owned by the OTHER direction. Harvest all
        # endpoints, and keep admitting ready inputs while waiting for D2H.
        start = None
        deadline = time.monotonic() + 30
        while True:
            self.engine.progress()
            self.prefetch(stage, exclude_stage=stage if not control.source else None)
            ticket = pool.reserve(wire)
            if ticket is not None:
                if start is not None:
                    stat = self.credit_waits["source" if control.source else "destination"]
                    stat["calls"] += 1
                    stat["milliseconds"] += (time.perf_counter_ns() - start) / 1e6
                return ticket
            if start is None:
                start = time.perf_counter_ns()
            if time.monotonic() >= deadline:
                raise TimeoutError("VPP rank GPU credit deadline expired")
            time.sleep(0.001)

    def prepare_send(self, batch, stage, rows):
        """Outside compile/capture: borrow outputs and pre-admit this generation."""
        self._check()
        key = batch, stage
        if not self.direct_outputs or key in self.producer_leases:
            raise RuntimeError("Expected one independent producer lease per boundary")
        control, pool, edge = self.endpoints[stage]
        wire = control.describe(batch, rows)
        ticket = self._reserve(stage, wire)
        views = pool.producer_views(ticket)
        token = pool.publication_token(ticket)
        self.producer_leases[key] = ticket, views, token
        # This header is admission, NOT GPU completion. C++ will not advance
        # either plane until THIS token's Record was submitted and completed.
        control.send(batch, wire)
        edge.submit(ticket, True)
        return *views, token

    def send(self, batch, stage, tensors):
        self._check()
        if set(tensors) != {"hidden_states", "residual"}:
            raise ValueError("Expected native dual-BF16 boundary, no extra payload")
        control, pool, edge = self.endpoints[stage]
        if self.direct_outputs:
            key = batch, stage
            if key not in self.producer_leases:
                # Native warmup/PW fallback: publication remains outside capture.
                self.prepare_send(batch, stage, tensors["hidden_states"].shape[0])
            ticket, views, token = self.producer_leases[key]
            values = tuple(tensors[name] for name in ("hidden_states", "residual"))
            if any(
                a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or not a.is_contiguous()
                for a, b in zip(values, views)
            ):
                raise ValueError("Output must match the native contiguous dual-BF16 boundary")
            aliases = [a.data_ptr() == b.data_ptr() and a.shape == b.shape for a, b in zip(values, views)]
            if any(aliases) and not all(aliases):
                raise RuntimeError("Partial output-slot binding is not a complete producer")
            stat = self.output_stats[str(stage)]
            if all(aliases):
                stat["direct_calls"] += 1
                stat["logical_copy_bytes_avoided"] += values[0].shape[0] * 5120 * 4
            else:
                own_storage = views[0].untyped_storage().data_ptr()
                if any(a.untyped_storage().data_ptr() == own_storage for a in values):
                    raise RuntimeError("Fallback input cannot alias another source-pool slot")
                views[1].copy_(values[1])
                self.native.record_plane(views[1], token, 1)
                views[0].copy_(values[0])
                self.native.record_plane(views[0], token, 0)
                stat["staged_calls"] += 1
            pool.publish_written(ticket)  # Strictly requires BOTH current-generation callbacks.
            del self.producer_leases[key]
            self.prefetch()
            return
        wire = control.describe(batch, tensors["hidden_states"].shape[0])
        ticket = self._reserve(stage, wire)
        pool.publish(ticket, tensors["hidden_states"], tensors["residual"])
        # First blocking GroupEnd may exchange bootstrap data with the peer.
        # Let that peer admit its matching recv before entering NCCL. This
        # header is not completion: CUDA ready/transfer events still guard data.
        control.send(batch, wire)
        edge.submit(ticket)
        self.prefetch()

    def receive(self, batch, stage):
        self._check()
        control, pool, edge = self.endpoints[stage]
        if (batch, stage) in self.leases:
            raise RuntimeError("Duplicate live destination lease")
        self.prefetch(stage, exclude_stage=stage)
        wire, _ = control.receive(batch)
        if stage in self.prefetched:
            admitted, ticket, prefetched_wire = self.prefetched.pop(stage)
            if admitted != batch or wire != prefetched_wire:
                raise RuntimeError("Gloo demand header differs from the admitted DMA READY header")
            self.prefetch_stats[self._direction(stage)]["consumed"] += 1
        else:
            ticket = self._reserve(stage, wire)
            if self.engine is not None:
                edge.submit(ticket, self.direct_outputs)
            else:
                edge.submit(ticket)
        if self.backend == "segment_dma":
            started = time.perf_counter_ns()
            edge.wait_recorded(ticket, 30000)
            self.record_wait_ns += time.perf_counter_ns() - started
            self.record_wait_calls += 1
        hidden, residual = pool.consume(ticket)  # Current compute stream waits on this receive.
        self.leases[batch, stage] = ticket
        self.prefetch()
        return {"hidden_states": hidden, "residual": residual}

    def retire(self, batch, stage):
        """After compute AND any forwarding copy, including the local3->4 fold."""
        self._check()
        if stage in (0, 3):
            return  # No input / retain stage3 input through its local successor.
        incoming = 2 if stage == 4 else stage - 1
        _, pool, _ = self.endpoints[incoming]
        ticket = self.leases[batch, incoming]
        pool.record_consumer_done(ticket)
        del self.leases[batch, incoming]
        self.prefetch()

    def receipt(self):
        if self.closed:
            return self.close_receipt
        self._check()
        return dict(
            backend=self.backend,
            capacity=self.capacity,
            slots=7,
            resource_layout=self.resource_layout,
            unique_payload_bytes=(sum(a.payload_bytes for a in self.arenas.values()) if self.arenas else
                                  sum(p.payload_bytes for _, p, _ in self.endpoints.values())),
            streams=dict(compute=1, d2h=int(self.engine is not None),
                         h2d=len(self.streams) - int(self.engine is not None)),
            arenas={name: dict(slots=2 if name == "source" else 3, live=a.live,
                               peak_live=a.peak_live, stalls=a.stalls, payload_bytes=a.payload_bytes)
                    for name, a in self.arenas.items()},
            credit_waits=self.credit_waits,
            prefetch=dict(directions=self.prefetch_stats, live=len(self.prefetched), peak_live=self.peak_prefetched),
            host_regions=self.host_receipts,
            progress_threads=int(self.engine is not None),
            record_ready_wait_ms=self.record_wait_ns / 1e6,
            record_ready_calls=self.record_wait_calls,
            producer_outputs=dict(
                enabled=self.direct_outputs, live=len(self.producer_leases), stages=self.output_stats
            ),
            header_prepost="dma_ready_acquire" if self.resource_layout == "rank" else "cpu_only",
            live_leases=len(self.leases),
            endpoints={
                str(stage): dict(
                    source=c.source,
                    sequence=c.next_sequence,
                    live=p.live,
                    peak_live=p.peak_live,
                    payload_bytes=p.payload_bytes,
                    payload_owner=("source" if c.source else "destination") if self.arenas else str(stage),
                )
                for stage, (c, p, _) in self.endpoints.items()
            },
        )

    def abort(self):
        self.failed = True
        if self.engine is not None:
            self.engine.stop()
        for _, _, edge in self.endpoints.values():
            edge.abort()  # Existing C++ fatal cleanup; no request-level skip/resume.

    def close(self):
        self._check()
        try:
            if self.leases or self.producer_leases or self.prefetched:
                raise RuntimeError("Cannot close with an unretired producer/destination lease")
            for control, _, _ in self.endpoints.values():
                control.close()
            deadline = time.monotonic() + 30
            while True:  # Cold drain; progress every endpoint before closing.
                for _, _, edge in self.endpoints.values():
                    edge.progress()
                if not any(p.live for _, p, _ in self.endpoints.values()):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("VPP transport shutdown drain expired")
                time.sleep(0.001)
            dist.barrier(group=self.group)
            if self.engine is not None:
                self.engine.stop()
                self.progress_stats = self.engine.stats()
                for _, _, edge in self.endpoints.values():
                    edge.disconnect()
                dist.barrier(group=self.group)
            self.close_receipt = self.receipt()
            self.close_receipt.update(closed=True, progress_stats=self.progress_stats)
            for _, pool, edge in self.endpoints.values():
                edge.close()
                pool.close()
            for arena in self.arenas.values():
                arena.close()
            self.closed = True
        except BaseException:
            self.abort()
            raise
