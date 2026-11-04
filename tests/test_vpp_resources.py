"""CPU checks of independent physical credits and nonblocking FIFO prefetch."""

from collections import deque
import importlib.util
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


def test_physical_credits_keep_both_edge_generations(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required")
    repo = Path(__file__).resolve().parents[1]
    source = repo / "src/qwen_infra/vpp/native"
    executable = tmp_path / "slot-pool.exe"
    subprocess.run([compiler, "-std=c++17", "-I" + str(source), str(repo / "tests/slot_pool.cpp"),
                    str(source / "slot_pool.cpp"), str(source / "interface.cpp"), "-o", str(executable)],
                   check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=15)


@pytest.fixture
def modules():
    pytest.importorskip("torch")
    from qwen_infra.vpp.native.control import BoundaryControl
    from qwen_infra.vpp.native.stages import StageBoundaries
    return BoundaryControl, StageBoundaries


def test_ready_header_matches_admission_without_waiting_or_retiring_work(modules):
    control_cls, _ = modules
    control = object.__new__(control_cls)
    control._check = lambda source: None
    control._pending = deque([(21, 8)])
    control.capacity, control.slots, control.epoch, control.stage = 2048, 7, 1, 5
    checked = []
    control.native = SimpleNamespace(check_boundary=lambda *args: checked.append(args))
    # No Work exists in this fixture: speculative validation must never wait.
    assert control.peek_admitted(b"current-generation") == 21
    assert checked[0][3:6] == (21, 8, 5)
    assert list(control._pending) == [(21, 8)]


def transport_fixture(stage_cls, *, ready=True):
    transport = object.__new__(stage_cls)
    transport.failed = transport.closed = False
    transport.resource_layout = "rank"
    transport.direct_outputs = True
    transport.engine = SimpleNamespace(progress=lambda: None)
    transport.prefetched, transport.endpoints = {}, {}
    transport.prefetch_stats = {d: dict(issued=0, consumed=0) for d in ("forward", "return")}
    transport.peak_prefetched = 0
    issued, reservations = [], []
    for stage in (0, 5):
        control = SimpleNamespace(source=False, peek_admitted=lambda wire: 11)
        pool = SimpleNamespace(reserve=lambda wire: reservations.append(wire) or object())
        edge = SimpleNamespace(peek_header=lambda: b"published" if ready else None,
                               submit=lambda ticket, independent: issued.append((ticket, independent)))
        transport.endpoints[stage] = control, pool, edge
    return transport, issued, reservations


def test_prefetch_issues_both_directions_once_and_preserves_demand_room(modules):
    _, stages = modules
    transport, issued, _ = transport_fixture(stages)
    transport.prefetch(priority_stage=5)
    transport.prefetch()
    assert list(transport.prefetched) == [5, 0]
    assert len(issued) == 2 and all(independent for _, independent in issued)
    assert transport.peak_prefetched == 2
    assert transport.prefetch_stats == {d: dict(issued=1, consumed=0) for d in ("forward", "return")}


def test_unpublished_future_header_never_reserves_or_enqueues_a_receive(modules):
    _, stages = modules
    transport, issued, reservations = transport_fixture(stages, ready=False)
    transport.prefetch()
    assert not issued and not reservations and not transport.prefetched


def test_demand_credit_wait_progresses_other_edge_and_excludes_consumed_header(modules):
    _, stages = modules
    transport, _, _ = transport_fixture(stages)
    progress_calls, prefetch_calls = [], []
    transport.engine = SimpleNamespace(progress=lambda: progress_calls.append(True))
    attempts = iter([None, "demand-ticket"])
    control, _, edge = transport.endpoints[0]
    transport.endpoints[0] = control, SimpleNamespace(reserve=lambda wire: next(attempts)), edge
    transport.prefetch = lambda priority, **kw: prefetch_calls.append(kw["exclude_stage"])
    transport.credit_waits = {role: dict(calls=0, milliseconds=0) for role in ("source", "destination")}
    assert transport._reserve(0, b"header-consumed") == "demand-ticket"
    assert len(progress_calls) == 2 and prefetch_calls == [0, 0]
    assert transport.credit_waits["destination"]["calls"] == 1


def test_demand_verifies_gloo_and_consumes_existing_prefetch_once(modules):
    _, stages = modules
    transport, _, _ = transport_fixture(stages)
    transport.prefetch = lambda *args, **kwargs: None
    transport.backend, transport.leases = "segment_dma", {}
    transport.record_wait_ns = transport.record_wait_calls = 0
    waited = []
    control = SimpleNamespace(receive=lambda batch: (b"identical-wire", 17))
    pool = SimpleNamespace(consume=lambda ticket: ("hidden", "residual"))
    edge = SimpleNamespace(wait_recorded=lambda ticket, timeout: waited.append(ticket))
    transport.endpoints[0] = control, pool, edge
    transport.prefetched[0] = 11, "prefetch-ticket", b"identical-wire"
    assert transport.receive(11, 0) == dict(hidden_states="hidden", residual="residual")
    assert waited == ["prefetch-ticket"]
    assert transport.leases[11, 0] == "prefetch-ticket" and not transport.prefetched
    assert transport.prefetch_stats["forward"]["consumed"] == 1


def test_demand_rejects_gloo_header_different_from_speculative_generation(modules):
    _, stages = modules
    transport, _, _ = transport_fixture(stages)
    transport.prefetch = lambda *args, **kwargs: None
    transport.leases = {}
    control = SimpleNamespace(receive=lambda batch: (b"different-wire", 17))
    transport.endpoints[0] = control, None, None
    transport.prefetched[0] = 11, "prefetch-ticket", b"published-wire"
    with pytest.raises(RuntimeError, match="Gloo demand header differs"):
        transport.receive(11, 0)


def test_resource_snapshot_remains_available_after_close(modules):
    _, stages = modules
    transport = object.__new__(stages)
    transport.closed = True
    transport.close_receipt = dict(closed=True, unique_payload_bytes=5 * 2048 * 5120 * 4)
    assert transport.receipt() == transport.close_receipt


@pytest.mark.parametrize("exception", [ValueError, RuntimeError])
def test_gpu_stress_accepts_only_expected_native_stale_ticket_rejection(exception):
    source = Path(__file__).resolve().parents[1] / "benchmarks/vpp_resources.py"
    spec = importlib.util.spec_from_file_location("vpp_resource_benchmark", source)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    def reject(ticket):
        raise exception("stale completion or retired ticket")

    assert benchmark.check_retired_ticket(SimpleNamespace(consume=reject), object()) == 1
    with pytest.raises(RuntimeError, match="Retired ticket accepted"):
        benchmark.check_retired_ticket(SimpleNamespace(consume=lambda ticket: None), object())

    def unrelated(ticket):
        raise RuntimeError("CUDA failure")

    with pytest.raises(RuntimeError, match="CUDA failure"):
        benchmark.check_retired_ticket(SimpleNamespace(consume=unrelated), object())
