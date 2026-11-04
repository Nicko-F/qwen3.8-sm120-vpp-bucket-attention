"""Automatic XQA routing on the native FULL graph lifecycle."""

from .attention_kv_full_bridge import FullWorkerLifecycle, WorkerFullEntry, metadata_binding
from .attention_kv_fullgraph import input_key
from .nvfp4_bucket_xqa import MAX_TABLE_WIDTH
from qwen_infra.limits import MAX_SEQUENCE_LENGTH


def xqa_route(lengths, *, table_width, sm_count, forced="auto", diagnostic=False):
    """Conservative geometry rule; CPU metadata only, no timing-table fit."""
    if forced not in ("auto", "native", "persistent", "aligned"):
        raise ValueError("Expected auto/native/persistent/aligned route")
    supported = (
        isinstance(lengths, (tuple, list))
        and len(lengths) in (1, 8, 16, 24, 32)
        and type(sm_count) is int
        and len(lengths) <= sm_count <= 340
        and type(table_width) is int
        and 1 <= table_width <= MAX_TABLE_WIDTH
        and all(type(n) is int and 0 <= n <= table_width * 32 for n in lengths)
        and any(lengths)
    )
    if supported:
        tiles = [(n + 255) // 256 for n in lengths]
        splits = min(max(1, sm_count // len(lengths)), (table_width * 32 + 255) // 256)
        bucket = max(2, (sum(tiles) + sm_count - 1) // sm_count)
        critical = max((n + splits - 1) // splits for n in tiles)
        empty = len(lengths) * splits - sum(min(n, splits) for n in tiles)
        auto = (
            "persistent" if critical >= 2 * bucket else "aligned" if critical == 1 and empty > 0 else "native"
        )
        route = auto if forced == "auto" else forced
        if not diagnostic:
            return route
        info = dict(
            route=route,
            auto_route=auto,
            supported=True,
            forced=forced,
            active_requests=sum(n > 0 for n in lengths),
            padded_rows=len(lengths),
            table_width=table_width,
            sm_count=sm_count,
            native_splits=splits,
            native_empty_subsequences=empty,
            native_critical_tiles=critical,
            interval_tiles_per_bucket=bucket,
            reason="geometry",
        )
        return info
    return dict(route="native", supported=False, reason="unsupported") if diagnostic else "native"


class CUDAFullEntry(WorkerFullEntry):
    """One native padded-q1 descriptor, same capture and replay hooks."""

    capacity = MAX_SEQUENCE_LENGTH

    def __init__(self, runner, adapters, workspace, *, observe=None, backend="xqa", route=None):
        super().__init__(runner, adapters, observe=observe)
        self.workspace = workspace
        self.rows = workspace.batch
        self.backend, self.route = backend, route
        self.selected_route = "native"

    def prepare_plans(self, metadata):
        self.metadata_key(metadata)
        first = metadata[min(self.fa_names)]
        self.table_width = first.decode.block_tables.shape[1]
        return [
            a.prepare_plan(metadata[a.attention.layer_name], workspace=self.workspace, route=self.route)
            for a in self.adapters
        ]

    def metadata_key(self, metadata):
        from .nvfp4_bucket_cuda import cuda_metadata_key

        fa = [metadata[name] for name in sorted(self.fa_names)]
        if not fa or any(item is not fa[0] for item in fa):
            raise ValueError("Full-attention layers must share one native metadata owner")
        gdn = {name: item for name, item in metadata.items() if name not in self.fa_names}
        return cuda_metadata_key(fa[0], self.rows), metadata_binding(gdn, set(), rows=self.rows)

    def prepare_forward(self):
        self.workspace.assign()

    def selected(self, metadata, lengths):
        return not getattr(self, "force_native", False) and self.selected_route == self.route

    def observed_selection(self, selected):
        return dict(
            bucket=selected,
            route=self.route if selected else "native",
            route_geometry=getattr(self, "route_geometry", None),
        )

    def plan_receipt(self, plans):
        if (
            len(plans) != len(self.adapters)
            or len(set(plans)) != 1
            or any(a.workspace is not self.workspace or a.assign_tasks for a in self.adapters)
        ):
            raise RuntimeError("Expected one device task plan shared across native full attention")
        return dict(
            strategy="xqa_" + self.route if self.backend == "xqa" else "cuda_nvfp4_compact_buckets",
            backend=self.backend,
            route=self.route,
            capacities=[self.capacity] * len(self.adapters),
            padded_rows=self.rows,
            workspace_count=1,
            workspace_logical_bytes=self.workspace.logical_bytes,
            workspace_pointers=list(self.workspace.pointers),
            shared_workspace_all_fa=True,
            assign_every_fa=False,
            cross_layer_task_reuse=True,
            task_refresh_every_forward=True,
            ctas=self.workspace.ctas,
            tile=self.workspace.tile,
            table_width=getattr(self, "table_width", None),
            callback_counts=[a.calls for a in self.adapters],
        )

    def replay(self, descriptor):
        current, self.current = self.current, None
        selected = binding_checked = False
        if self.graph is not None and descriptor == self.descriptor and current is not None:
            metadata, lengths = current
            selected = self.selected(metadata, lengths)
        # Native owns its fallback Graph and buffers. Only a selected private
        # replay needs the extra FA/GDN/input binding walk below.
        if selected:
            from .attention_kv_full_capture import current_inputs

            if (
                frozenset(metadata) != self.names
                or self.metadata_key(metadata) != self.binding
                or self.manager.hidden_states is not self.output
                or input_key((self.output,)) != self.output_key
                or input_key(current_inputs(self.runner, rows=self.rows)) != self.inputs_key
                or self.manager.graphs.get(descriptor) is not self.native_graph
                or self.native_graph is None
            ):
                raise ValueError("Original NVFP4 FULL staging/output binding changed")
            binding_checked = True
        if self.observe is not None:
            self.observe(
                "replay",
                dict(
                    **self.observed_selection(selected),
                    mode=descriptor.cg_mode.name,
                    tokens=descriptor.num_tokens,
                    requests=descriptor.num_reqs,
                    lengths=list(current[1]) if current else None,
                    current_binding_checked=binding_checked,
                ),
            )
        if not selected:
            return self.native_replay(descriptor)
        # Cache/weight owners are immutable and checked at cold RPC boundaries;
        # initialize_kv_cache releases this entry before replacing any storage.
        from vllm.model_executor.offloader.base import get_offloader

        get_offloader().sync_prev_onload()
        self.graph.replay()
        return self.output[: descriptor.num_tokens]


class CUDAMultiEntry(CUDAFullEntry):
    """Select among native R16/24/32 descriptors without altering dispatch."""

    def __init__(self, runner, adapters, workspaces, *, observe=None, backend="xqa", sm_count=170):
        root = (16, "persistent") if backend == "xqa" else 16
        super().__init__(
            runner,
            adapters,
            workspaces[root],
            observe=observe,
            backend=backend,
            route="persistent" if backend == "xqa" else None,
        )
        self.sm_count, self.force_route = sm_count, "auto"
        self.entries = {
            key: self
            if key == root
            else CUDAFullEntry(
                runner,
                adapters,
                ws,
                observe=observe,
                backend=backend,
                route=key[1] if backend == "xqa" else None,
            )
            for key, ws in workspaces.items()
        }
        self.supported_rows = set(workspaces) if backend != "xqa" else set()

    def capture(self):
        from .attention_kv_full_capture import capture_entry

        receipts = []
        if self.backend == "xqa":
            from vllm.config import CUDAGraphMode

            native_rows = {
                d.num_reqs
                for d in self.manager._capture_descs[CUDAGraphMode.FULL]
                if d.num_tokens == d.num_reqs
                and d.uniform_token_count == 1
                and d.num_active_loras == 0
                and d.num_ubatches == 1
            }
        for entry in self.entries.values():
            if self.backend == "xqa" and entry.rows not in native_rows:
                continue
            capture_entry(entry)
            receipts.append(dict(entry.capture_receipt))
        if self.backend == "xqa":
            self.supported_rows = {e.rows for e in self.entries.values() if e.graph is not None}
        self.capture_receipt = dict(
            self.capture_receipt or {}, extra_entries=len(receipts), captures_by_rows=receipts
        )

    def decode_lengths(self, batch):
        from .attention_kv_full_bridge import current_decode_lengths

        rows = batch.num_reqs_after_padding
        if rows not in self.supported_rows:
            return None
        return current_decode_lengths(batch, rows=rows, allow_padding=True)

    def replay(self, descriptor):
        if self.backend == "xqa":
            geometry = None
            route = "native"
            current = self.current
            representative = self.entries.get((descriptor.num_reqs, "persistent"))
            if (
                current is not None
                and representative is not None
                and representative.graph is not None
                and descriptor == representative.descriptor
            ):
                forced = "native" if getattr(self, "force_native", False) else self.force_route
                if forced != "native" or self.observe is not None:
                    chosen = xqa_route(
                        current[1],
                        table_width=representative.table_width,
                        sm_count=self.sm_count,
                        forced=forced,
                        diagnostic=self.observe is not None,
                    )
                    if self.observe is not None:
                        geometry = chosen
                        route = chosen["route"]
                    else:
                        route = chosen
            entry = self.entries.get((descriptor.num_reqs, route), self)
            entry.selected_route = route
            if self.observe is not None:
                entry.route_geometry = geometry
        else:
            entry = self.entries.get(descriptor.num_reqs, self)
        self.last_selected_route = route if self.backend == "xqa" else "native"
        self.last_padded_rows = descriptor.num_reqs
        if entry is self:
            return super().replay(descriptor)
        entry.current, self.current = self.current, None
        entry.native_replay = self.native_replay
        entry.force_native = getattr(self, "force_native", False)
        return entry.replay(descriptor)

    def restore(self):
        for entry in self.entries.values():
            if entry is not self:
                entry.restore()
        super().restore()
        # The R16 member is self: break that cycle before native profile teardown.
        self.entries.clear()


class CUDAFullLifecycle(FullWorkerLifecycle):
    def __init__(self, runner, attentions, *, observe=None, backend_observer=None):
        super().__init__(runner, attentions, observe=observe, backend_observer=backend_observer)
        if runner.ubatch_runner is not None or not 1 <= runner.max_model_len <= MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Bucket decode requires sequential q1 execution within signed-32-bit table capacity"
            )
        import torch
        from qwen_infra.cache import build_directory
        from .nvfp4_bucket_xqa import build, Workspace

        directory = build_directory("decode")
        self.backend = "xqa"
        self.fusions = None
        self.sm_count = torch.cuda.get_device_properties(runner.device).multi_processor_count
        if not 32 <= self.sm_count <= 340:
            raise ValueError("Device SM count exceeds the bounded task planner capacity")
        flat, flat_resources = build(directory, persistent=False)
        persistent, persistent_resources = build(directory, persistent=True)
        self.library = {"aligned": flat, "persistent": persistent}
        self.resources = dict(backend="xqa", aligned=flat_resources, persistent=persistent_resources)
        tails = {}
        front = None
        fusion = runner.vllm_config.additional_config.get("qwen_infra_attention_fusions", False)
        if fusion:
            from .nvfp4_attention_front import build as build_front
            from .nvfp4_attention_fusion import ModelAttentionFusions

            for route in self.library:
                tails[route], self.resources[route + "_tail"] = build(
                    directory, persistent=route == "persistent", tail=True
                )
            front, self.resources["front"] = build_front(directory)
        self.workspaces = {
            (rows, route): Workspace(
                torch, lib, rows, ctas=self.sm_count, device=runner.device, tail_library=tails.get(route)
            )
            for rows in (1, 8, 16, 24, 32)
            for route, lib in self.library.items()
        }
        self.workspace = self.workspaces[(16, "persistent")]
        # Publish model hooks only after every fallible allocation/build has
        # completed. A failed constructor must leave native model methods intact.
        if front is not None:
            self.fusions = ModelAttentionFusions(runner.model, self.attentions, front)
        self.allocation_receipt = dict(
            logical_bytes=sum(w.logical_bytes for w in self.workspaces.values()),
            backend="xqa",
            sm_count=self.sm_count,
            resources=self.resources,
        )

    def make_adapter(self, attention, observer):
        import torch
        from .nvfp4_bucket_cuda import NativeCUDAAdapter

        adapter = NativeCUDAAdapter(
            torch,
            attention,
            self.workspace,
            observer=observer,
            workspaces=self.workspaces,
            backend=self.backend,
        )
        if self.fusions is not None:
            self.fusions.owners[attention.layer_name].adapter = adapter
        return adapter

    def make_entry(self):
        entry = CUDAMultiEntry(
            self.runner,
            self.adapters,
            self.workspaces,
            observe=self.observe,
            backend=self.backend,
            sm_count=self.sm_count,
        )
        if self.fusions is not None:
            owners = self.fusions.owners

            def counters():
                return {
                    name: dict(front=o.front_calls, tail=o.tail_calls, fallback=o.fallback_calls)
                    for name, o in owners.items()
                }

            for member in entry.entries.values():
                member.cold_fusion_counters = counters
        return entry

    def release(self):
        super().release()
        if getattr(self, "fusions", None) is not None:
            self.fusions.release_adapters()
        for workspace in getattr(self, "workspaces", {}).values():
            workspace.release_bindings()
            if self.backend == "xqa":
                workspace.plan.release_binding()
        if self.backend == "xqa" and self.captures and self.captures[-1]["profiling"]:
            self.captures[-1].update(task_plan_bindings_released=True, persistent_plan_buffers_retained=True)

    def restore(self):
        super().restore()
        if getattr(self, "fusions", None) is not None:
            self.fusions.restore()
