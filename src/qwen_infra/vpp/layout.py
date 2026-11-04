# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fold-back topology derived from Huawei VPP c409752 (Apache-2.0).

Original: vllm_ascend/distributed/vpp_utils.py. This P-only CUDA port fixes
the owner-approved64-layer/4-rank/2-traversal topology; no device imports.
"""

from dataclasses import dataclass
from collections import deque
import json


def layer_ranges(rank):
    if type(rank) is not int or not 0 <= rank < 4:
        raise ValueError("Expected one of four logical P ranks")
    return ((rank * 8, (rank + 1) * 8), ((7 - rank) * 8, (8 - rank) * 8))


def communication(rank, traversal):
    layer_ranges(rank)
    if traversal not in (0, 1):
        raise ValueError("Expected forward or return traversal")
    if traversal == 0:
        return (rank - 1 if rank else None, rank + 1 if rank < 3 else None)
    return (rank + 1 if rank < 3 else None, rank - 1 if rank else None)


@dataclass
class VppStep:
    scheduler_output: object
    batch_id: int
    traversal: int = 0


@dataclass
class VppContinuation:
    batch_id: int


class RankLocalOrder:
    """CPU-only order shared by replay and the opt-in rank-local GPU adapter.

    One forward and one return FIFO preserve each layer's recurrent frontier.
    The seven-batch limit is derived from the eight-stage folded topology.
    Returned contexts remain owned until the global sample/release RPC.
    """

    capacity = 7

    def __init__(self, rank):
        layer_ranges(rank)
        self.rank = rank
        self.warmup = 7 - 2 * rank
        self.pending, self.returns = deque(), deque()
        self.live, self.returned = set(), set()
        self.last_batch = -1
        self.prefer_first = True
        self.peak_live = 0

    def admit(self, batch):
        if type(batch) is not int or batch <= self.last_batch:
            raise ValueError("Batch IDs must strictly increase")
        if len(self.live) >= self.capacity:
            raise RuntimeError("Seven-batch admission bound exceeded")
        self.last_batch = batch
        self.pending.append(batch)
        self.live.add(batch)
        self.peak_live = max(self.peak_live, len(self.live))

    def next(self, *, draining=False):
        """Choose one local stage, never skip a not-yet-admitted first batch."""
        first = self.warmup > 0 or self.prefer_first
        if first and self.pending:
            batch = self.pending.popleft()
            self.returns.append(batch)
            self.warmup = max(0, self.warmup - 1)
            self.prefer_first = self.warmup > 0
            return batch, self.rank
        if self.returns and (not first or draining):
            batch = self.returns.popleft()
            self.returned.add(batch)
            if self.warmup > 0:
                # An early drain retired a first before warmup filled its
                # window. Restore that credit for later admissions.
                self.warmup += 1
            self.prefer_first = True
            return batch, 7 - self.rank
        return None

    def release(self, batch):
        if batch not in self.returned:
            raise ValueError("Cannot release an unreturned or already released batch")
        self.returned.remove(batch)
        self.live.remove(batch)
        if not self.live:
            self.warmup = 7 - 2 * self.rank
            self.prefer_first = True


def rank_local_stage_orders(batch_ids):
    """Finite-bank CPU replay of broadcast admit/return/sample messages.

    A return message advances each rank once, then drains to its sample target
    if needed. No GPU readiness, communication time or speed is inferred.
    """
    batches = tuple(batch_ids)
    if not batches or len(batches) > 2048:
        raise ValueError("Expected one bounded nonempty bank")
    planners = [RankLocalOrder(r) for r in range(4)]
    orders = [[] for _ in planners]

    def advance(rank, draining=False):
        op = planners[rank].next(draining=draining)
        if op is not None:
            orders[rank].append(op)
        return op

    def admit(batch):
        for rank, planner in enumerate(planners):
            planner.admit(batch)
            advance(rank)
            if rank == 3:  # Local fold, not a seventh wire transfer.
                advance(rank)

    window = min(7, len(batches))
    for batch in batches[:window]:
        admit(batch)
    for index, batch in enumerate(batches):
        for rank, planner in enumerate(planners):
            advance(rank, draining=True)
            while batch not in planner.returned:
                if advance(rank, draining=True) is None:
                    raise RuntimeError("Missing local dependency while draining")
        for planner in planners:
            planner.release(batch)
        if index + window < len(batches):
            admit(batches[index + window])
    return orders


def retained_boundary_bytes(tokens, live_batches=6):
    if tokens <= 0 or not 1 <= live_batches <= 6:
        raise ValueError("Positive tokens and bounded in-flight queue required")
    return tokens * 5120 * 2 * 2 * live_batches


def profile_annotation(schedule, rank, traversal, batch_id, kind):
    """CPU scheduler metadata only; invoked only inside active profiling."""
    spans = layer_ranges(rank)
    if traversal not in (0, 1) or kind not in ("compute", "send", "recv"):
        raise ValueError("Expected one virtual stage and its compute/transfer scope")
    history = {r.req_id: r.num_computed_tokens for r in schedule.scheduled_new_reqs}
    cached = schedule.scheduled_cached_reqs
    history.update(zip(cached.req_ids, cached.num_computed_tokens))
    requests = [dict(id=str(r), q=q, h=history[r]) for r, q in schedule.num_scheduled_tokens.items()]
    if (
        any(type(r["q"]) is not int or r["q"] <= 0 or type(r["h"]) is not int or r["h"] < 0 for r in requests)
        or sum(r["q"] for r in requests) != schedule.total_num_scheduled_tokens
    ):
        raise ValueError("Invalid trace query/history metadata")
    return "qwen_infra_vpp:" + json.dumps(
        dict(
            rank=rank,
            stage=rank if traversal == 0 else 7 - rank,
            batch=batch_id,
            kind=kind,
            layers=spans[traversal],
            requests=requests,
        ),
        separators=(",", ":"),
    )
