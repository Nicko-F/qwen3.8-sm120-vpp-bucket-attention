"""One loaded-engine VPP resource comparison point, or bounded transport stress.

Run model points in separate supervised processes. Select devices/NUMA placement
and host-memory admission in the deployment launcher, rather than this library.
"""

import argparse
from datetime import timedelta
import json
import os
import statistics
import tempfile
import time


def environment():
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("NCCL_COMM_BLOCKING", "1")
    from qwen_infra.runtime_paths import prepare_jit_environment
    prepare_jit_environment()


def model_point(args):
    environment()
    from qwen_infra.config import PrefillConfig
    from qwen_infra.vpp.executor import FoldedExecutor
    from vllm import LLM, SamplingParams

    prompt_tokens = 4 * args.capacity
    options = PrefillConfig(
        args.model, chunk_size=args.capacity, max_model_len=prompt_tokens + 1,
        max_num_seqs=8, resource_layout=args.layout,
    ).engine_options()
    options["distributed_executor_backend"] = FoldedExecutor
    # A fixed KV budget prevents saved communication memory from silently
    # becoming a larger native cache, confounding the matched comparison.
    options["kv_cache_memory_bytes"] = 6 * 2**30
    llm = LLM(**options)
    banks, receipts, close = [], None, None
    try:
        seed = llm.get_tokenizer().encode("Hello", add_special_tokens=False)
        if not seed:
            raise ValueError("Tokenizer returned an empty seed")
        prompts = [{"prompt_token_ids": [seed[0]] * prompt_tokens} for _ in range(4)]
        params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
        for bank in range(args.warmup + args.repeats):
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            seconds = time.perf_counter() - started
            if len(outputs) != 4 or not all(o.finished and len(o.outputs[0].token_ids) == 1 for o in outputs):
                raise RuntimeError("Every bank must complete four P+1 requests")
            row = dict(bank=bank, warmup=bank < args.warmup, seconds=seconds, completed=len(outputs))
            banks.append(row)
            print(json.dumps(dict(event="bank", **row)), flush=True)
        receipts = llm.collective_rpc("qwen_infra_vpp_resources", timeout=60)
    finally:
        try:
            close = llm.collective_rpc("qwen_infra_vpp_close_transport", timeout=60)
        finally:
            llm.llm_engine.engine_core.shutdown()
    timed = [row["seconds"] for row in banks if not row["warmup"]]
    result = dict(
        mode="model", model=args.model, capacity=args.capacity, layout=args.layout,
        requests=4, prompt_tokens=prompt_tokens, max_num_seqs=8, kv_cache_memory_bytes=6 * 2**30,
        banks=banks, median_seconds=statistics.median(timed), range_seconds=[min(timed), max(timed)],
        resources=receipts, close=close,
    )
    print(json.dumps(result), flush=True)


def _stress_worker(rank, args):
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from qwen_infra.vpp.layout import RankLocalOrder
    from qwen_infra.vpp.native.stages import StageBoundaries

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=args.rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=45))
    pp = SimpleNamespace(rank_in_group=rank, ranks=list(range(4)), cpu_group=dist.group.WORLD)
    transport = StageBoundaries(pp, device, args.capacity, backend="segment_dma",
                                direct_outputs=True, resource_layout=args.layout)
    order = RankLocalOrder(rank)
    received, stale_rejected = [], 0

    def rows(batch):
        return (0, 1, 17, args.capacity // 2, args.capacity)[batch % 5]

    def run_next(draining=False):
        nonlocal stale_rejected
        op = order.next(draining=draining)
        if op is None:
            return None
        batch, stage = op
        incoming = rank - 1 if stage == rank else stage - 1
        if stage not in (0, 4):
            values = transport.receive(batch, incoming)
            for name, expected in (("hidden_states", batch), ("residual", incoming)):
                value = values[name]
                if value.shape != (rows(batch), 5120) or not bool(torch.all(value == expected)):
                    raise RuntimeError("Payload identity/valid-row mismatch")
            received.append(dict(batch=batch, stage=incoming, rows=rows(batch)))
        if stage in transport.endpoints and transport.endpoints[stage][0].source:
            hidden, residual, token = transport.prepare_send(batch, stage, rows(batch))
            # A source job is admitted before either current-generation plane
            # is published. A short asynchronous delay exercises that contract.
            if batch % 3 == 0:
                torch.cuda._sleep(200_000)
            residual.fill_(stage)
            transport.native.record_plane(residual, token, 1)
            if batch % 4 == 0:
                torch.cuda._sleep(200_000)
            hidden.fill_(batch)
            transport.native.record_plane(hidden, token, 0)
            transport.send(batch, stage, dict(hidden_states=hidden, residual=residual))
        transport.retire(batch, stage)
        return op

    def admit(batch):
        transport.admit(batch)
        order.admit(batch)
        run_next()
        if rank == 3:
            run_next()

    try:
        window = min(7, args.messages)
        for batch in range(window):
            admit(batch)
        for batch in range(args.messages):
            run_next(draining=True)
            while batch not in order.returned:
                if run_next(draining=True) is None:
                    raise RuntimeError("Missing local stress dependency")
            order.release(batch)
            if batch + window < args.messages:
                admit(batch + window)
            if batch % 4 == 0:
                print(json.dumps(dict(event="stress_progress", rank=rank, batch=batch)), flush=True)
        torch.cuda.synchronize(device)
        transport.engine.progress()
        if transport.arenas and any(arena.live for arena in transport.arenas.values()):
            raise RuntimeError("Completed stress left live physical leases")
        # A retired native ticket must reject consumer reuse, even after logical
        # slots have wrapped. Use an isolated loopback pool to avoid poisoning a
        # live edge's publication state with a deliberately invalid operation.
        native = transport.native
        source = native.DeviceSlotPool(native.SlotRole.Source, 1, 1, 1, 0, rank)
        destination = native.DeviceSlotPool(native.SlotRole.Destination, 1, 1, 1, 0, rank)
        wire = native.boundary(1, 1, 1, 0, 0, 0, 1)
        st, dt = source.reserve(wire), destination.reserve(wire)
        pair = source.producer_views(st)
        source.publish_written(st)
        source.loopback(st, destination, dt)
        destination.consume(dt)
        destination.record_consumer_done(dt)
        torch.cuda.synchronize(device)
        source.progress()
        destination.progress()
        try:
            destination.consume(dt)
        except RuntimeError:
            stale_rejected += 1
        else:
            raise RuntimeError("Retired ticket accepted")
        source.close()
        destination.close()
        del pair
        before = transport.receipt()
        if args.layout == "rank":
            expected_bytes = 5 * args.capacity * 5120 * 4
            if before["unique_payload_bytes"] != expected_bytes or before["prefetch"]["peak_live"] > 2:
                raise RuntimeError("Rank-shared payload/speculation budget exceeded")
            for name, slots in (("source", 2), ("destination", 3)):
                if before["arenas"][name]["peak_live"] > slots:
                    raise RuntimeError("Physical arena ownership exceeded its fixed capacity")
        transport.close()
        receipt = dict(rank=rank, received=received, stale_rejected=stale_rejected,
                       before_close=before, close=transport.receipt())
        results = [None] * 4
        dist.all_gather_object(results, receipt)
        if rank == 0:
            print(json.dumps(dict(mode="module", capacity=args.capacity, layout=args.layout,
                                  messages=args.messages, ranks=results)), flush=True)
    finally:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--capacity", type=int, choices=(2048, 8192), required=True)
    parser.add_argument("--layout", choices=("edge", "rank"), required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("model", "module"), default="model")
    parser.add_argument("--messages", type=int, default=20)
    args = parser.parse_args(argv)
    if args.mode == "model":
        if not args.model or args.warmup < 0 or args.repeats < 1:
            parser.error("Model mode requires --model, nonnegative warmup and positive repeats")
        model_point(args)
    else:
        if not 8 <= args.messages <= 64:
            parser.error("Module stress requires 8..64 messages")
        environment()
        import torch.multiprocessing as mp
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix="vpp-resource-") as rendezvous:
            args.rendezvous = (Path(rendezvous) / "store").as_uri()
            mp.spawn(_stress_worker, args=(args,), nprocs=4, join=True)


if __name__ == "__main__":
    main()
