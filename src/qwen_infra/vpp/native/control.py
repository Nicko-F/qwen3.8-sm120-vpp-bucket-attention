# SPDX-License-Identifier: Apache-2.0
"""Fixed CPU headers on the existing Gloo group; no payload or GPU progress.

The C++ codec owns the ABI. Header prepost is not NCCL prepost: the stage adapter
must establish a valid payload/consumer dependency separately. One host owner,
bounded buffers, no background thread or per-message tensor allocation.
"""

from collections import deque
from datetime import timedelta
from threading import get_ident

import torch
import torch.distributed as dist


class BoundaryControl:
    def __init__(self, native, group, ranks, *, source, stage, capacity, slots, epoch, timeout_seconds=120):
        native.boundary(capacity, slots, epoch, 0, 0, stage, 0)
        if type(source) is not bool or not 0 < timeout_seconds <= 120:
            raise ValueError("Expected endpoint role and bounded control timeout")
        ranks = tuple(ranks)
        if (
            len(ranks) != 4
            or len(set(ranks)) != 4
            or tuple(dist.get_process_group_ranks(group)) != ranks
            or dist.get_backend(group) != "gloo"
        ):
            raise ValueError("Use the existing four-rank native Gloo control group")
        rank_of = lambda s: s if s < 4 else 7 - s
        src, dst = ranks[rank_of(stage)], ranks[rank_of(stage + 1)]
        if dist.get_rank() != (src if source else dst):
            raise ValueError("Control endpoint belongs to another physical rank")
        self.native, self.group = native, group
        self.source, self.peer, self.stage = source, dst if source else src, stage
        self.capacity, self.slots, self.epoch = capacity, slots, epoch
        self.timeout = timedelta(seconds=timeout_seconds)
        self.tag = 0x565050 + stage  # Gloo only; NCCL does not support tags.
        self._buffers = [bytearray(96) for _ in range(slots)]
        self._tensors = [torch.frombuffer(b, dtype=torch.uint8) for b in self._buffers]
        self._work = [None] * slots
        self._pending = deque()
        self._sequence = 0
        self._owner = get_ident()
        self._failed = self._closed = False

    @property
    def next_sequence(self):
        return self._sequence

    def _check(self, source):
        if self._failed or self._closed or self.source != source or self._owner != get_ident():
            raise RuntimeError("Wrong owner/role or stopped control epoch")

    def _wait(self, slot):
        work = self._work[slot]
        if work is not None:
            try:
                if not work.wait(self.timeout):
                    raise RuntimeError("Gloo control completion failed")
            except BaseException:
                self._failed = True
                raise
            self._work[slot] = None

    def describe(self, batch, rows):
        """Use actual producer tensor rows, which may include native padding."""
        self._check(True)
        return self.native.boundary(
            self.capacity, self.slots, self.epoch, batch, self._sequence, self.stage, rows
        )

    def send(self, batch, wire):
        """Keep this admission header until Gloo completes; it is NOT GPU ready.

        Legacy callers publish first; independent-ready callers reserve/lease
        first and let current-generation CUDA plane events guard actual data.
        """
        self._check(True)
        self.native.check_boundary(
            self.capacity, self.slots, self.epoch, batch, self._sequence, self.stage, wire
        )
        slot = self._sequence % self.slots
        self._wait(slot)  # Never overwrite a live asynchronous send buffer.
        self._buffers[slot][:] = wire
        try:
            work = dist.isend(self._tensors[slot], self.peer, self.group, self.tag)
            if work is None:
                raise RuntimeError("Gloo send did not return an owned Work")
            self._work[slot] = work
            self._sequence += 1
        except BaseException:
            self._failed = True
            raise

    def post(self, batch):
        """Prepost one fixed CPU header, without issuing a CUDA receive."""
        self._check(False)
        self.native.boundary(self.capacity, self.slots, self.epoch, batch, self._sequence, self.stage, 0)
        slot = self._sequence % self.slots
        if self._work[slot] is not None or any(b == batch for b, _ in self._pending):
            raise RuntimeError("Control window full or duplicate live batch")
        try:
            work = dist.irecv(self._tensors[slot], self.peer, self.group, self.tag)
            if work is None:
                raise RuntimeError("Gloo receive did not return an owned Work")
            self._work[slot] = work
            self._pending.append((batch, self._sequence))
            self._sequence += 1
        except BaseException:
            self._failed = True
            raise

    def receive(self, batch):
        self._check(False)
        if not self._pending or self._pending[0][0] != batch:
            raise RuntimeError("Consume the next admitted control record in FIFO order")
        _, sequence = self._pending[0]
        slot = sequence % self.slots
        self._wait(slot)
        wire = bytes(self._buffers[slot])
        try:
            rows = self.native.check_boundary(
                self.capacity, self.slots, self.epoch, batch, sequence, self.stage, wire
            )
        except BaseException:
            self._failed = True
            raise
        self._pending.popleft()
        return wire, rows

    def peek_admitted(self, wire):
        """Validate a DMA READY header against admission, without touching Gloo.

        Gloo P2P Work.is_completed is not a portable receive readiness signal.
        The DMA release/acquire protocol supplies the speculative wire; the
        original Work/buffer remain owned until receive() verifies it on demand.
        """
        self._check(False)
        if not self._pending:
            return None
        batch, sequence = self._pending[0]
        self.native.check_boundary(
            self.capacity, self.slots, self.epoch, batch, sequence, self.stage, wire
        )
        return batch

    def close(self):
        self._check(self.source)
        if self._pending:
            raise RuntimeError("Drain admitted receives before closing the control channel")
        for slot in range(self.slots):
            self._wait(slot)
        self._closed = True
