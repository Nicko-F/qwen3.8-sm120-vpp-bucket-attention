import math
import random

import pytest

from qwen_infra.kernels.nvfp4_bucket_xqa import task_plan
from qwen_infra.kernels.nvfp4_kv_bucket_worker import xqa_route
from qwen_infra.vpp.layout import RankLocalOrder, layer_ranges, rank_local_stage_orders


@pytest.mark.parametrize("batch", [1, 8, 16, 24, 32])
@pytest.mark.parametrize("aligned", [False, True])
def test_partition_covers_each_real_kv_tile_once(batch, aligned):
    rng = random.Random(987 + batch)
    for _ in range(10):
        lengths = [rng.choice([0, 1, 31, 256, 2049, 32768, 131072]) for _ in range(batch)]
        plan = task_plan(lengths, ctas=170, aligned=aligned, table_width=4097)
        assert len(plan["tasks"]) <= 170 + batch - 1
        for request, length in enumerate(lengths):
            tasks = [task for task in plan["tasks"] if task[0] == request]
            actual = [tile for _, lo, hi, stride in tasks for tile in range(lo, hi, stride)]
            assert actual == list(range(math.ceil(length / 256)))
            begin, count = plan["requests"][request]
            assert plan["tasks"][begin : begin + count] == tasks
        assigned = [slot for lo, hi in plan["buckets"] for slot in range(lo, hi)]
        assert sorted(assigned) == list(range(len(plan["tasks"])))


def test_route_distinguishes_severe_skew_empty_native_splits_and_native_geometry():
    assert xqa_route([131072] + [256] * 15, table_width=4096, sm_count=170) == "persistent"
    assert xqa_route([256] * 16, table_width=4096, sm_count=170) == "aligned"
    assert xqa_route([32768] * 16, table_width=4096, sm_count=170) == "native"
    assert xqa_route([131073] + [256] * 15, table_width=4096, sm_count=170) == "native"
    assert xqa_route([32768] * 16, table_width=1, sm_count=170) == "native"


def test_folded_ranges_cover_each_layer_once():
    ownership = [i for rank in range(4) for lo, hi in layer_ranges(rank) for i in range(lo, hi)]
    assert sorted(ownership) == list(range(64))


@pytest.mark.parametrize("size", [1, 4, 7, 8, 20])
def test_folded_queue_is_bounded_and_preserves_each_stage_fifo(size):
    bank = tuple(range(10, 10 + size))
    for rank, order in enumerate(rank_local_stage_orders(bank)):
        for stage in (rank, 7 - rank):
            assert [batch for batch, current in order if current == stage] == list(bank)


def test_rank_local_queue_rejects_over_admission_and_double_release():
    order = RankLocalOrder(0)
    for batch in range(7):
        order.admit(batch)
    with pytest.raises(RuntimeError):
        order.admit(7)
    while 0 not in order.returned:
        assert order.next(draining=True) is not None
    order.release(0)
    with pytest.raises(ValueError):
        order.release(0)
