"""Native FULL graph and cache lifecycle ownership."""

import logging
from .attention_kv_fullgraph import input_key


def current_decode_lengths(batch, *, rows=16, allow_padding=False):
    """Current CPU upper bounds only; no CUDA reads or synthetic lengths."""
    actual = batch.num_reqs
    if (
        not 1 <= actual <= rows
        or (not allow_padding and actual != rows)
        or batch.num_reqs_after_padding != rows
        or batch.num_tokens != actual
        or batch.num_tokens_after_padding != rows
        or batch.has_prefill
        or any(batch.is_prefilling_np)
        or tuple(batch.num_scheduled_tokens) != (1,) * actual
        or tuple(batch.query_start_loc_np[: rows + 1])
        != tuple(range(actual + 1)) + (actual,) * (rows - actual)
        or batch.seq_lens_cpu_upper_bound.device.type != "cpu"
    ):
        return None
    lengths = tuple(int(n) for n in batch.seq_lens_cpu_upper_bound[:rows].tolist())
    if len(lengths) != rows or any(n <= 0 for n in lengths[:actual]) or any(lengths[actual:]):
        return None
    return lengths


def metadata_binding(metadata, fa_names, *, rows=16):
    """Deduplicate native shared builder metadata; validate the dense ABI."""
    groups = {}
    for name, item in metadata.items():
        group = groups.setdefault(id(item), [item, []])
        group[1].append(name)
    bindings = []
    for item, names in groups.values():
        names = tuple(sorted(names))
        if set(names) <= fa_names:
            raise ValueError("FA bindings are owned by the decode adapter")
        else:
            if (
                set(names) & fa_names
                or item.num_actual_tokens != rows
                or item.num_decodes != rows
                or item.num_decode_tokens != rows
                or item.num_prefills != 0
                or item.num_prefill_tokens != 0
                or item.num_spec_decodes != 0
                or item.num_spec_decode_tokens != 0
                or any(
                    getattr(item, field, None) is not None
                    for field in (
                        "has_initial_state",
                        "chunk_indices",
                        "chunk_offsets",
                        "prefill_query_start_loc",
                        "prefill_state_indices",
                        "prefill_has_initial_state",
                        "spec_query_start_loc",
                        "spec_state_indices_tensor",
                        "spec_sequence_masks",
                        "spec_token_indx",
                        "non_spec_token_indx",
                        "num_accepted_tokens",
                        "nums_dict",
                        "batch_ptr",
                        "token_chunk_offset_ptr",
                    )
                )
                or item.non_spec_state_indices_tensor is None
                or item.non_spec_query_start_loc is None
                or item.non_spec_state_indices_tensor.numel() != rows
                or item.non_spec_query_start_loc.numel() != rows + 1
            ):
                raise ValueError("Current native GDN/Conv FULL ABI changed")
            key = ("GDN", input_key((item.non_spec_state_indices_tensor, item.non_spec_query_start_loc)))
        bindings.append((names, key))
    return tuple(sorted(bindings))


def cache_binding(registry, names):
    rows = []
    for name in sorted(names):
        layer = registry[name]
        cache = layer.kv_cache
        tensors = tuple(cache) if isinstance(cache, (list, tuple)) else (cache,)
        rows.append((name, id(layer), tuple(id(t) for t in tensors), input_key(tensors)))
    return tuple(rows)


class WorkerFullEntry:
    """Current native staging feeds one fixed private Graph, never PW."""

    rows = 16

    def __init__(self, runner, adapters, *, observe=None):
        self.runner, self.manager = runner, runner.cudagraph_manager
        self.adapters, self.observe = tuple(adapters), observe
        self.fa_names = {a.attention.layer_name for a in adapters}
        self.current, self.graph, self.hooks = None, None, []
        self.capture_receipt = None

    def hook(self, owner, name, replacement):
        if name in vars(owner):
            raise RuntimeError("Do not replace an existing instance hook: " + name)
        setattr(owner, name, replacement)
        self.hooks.append((owner, name, replacement))

    def install(self):
        native_capture = self.manager.capture
        native_prepare = self.runner.model_state.prepare_attn
        self.native_replay = self.manager.run_fullgraph

        def captured(*args, **kwargs):
            native_capture(*args, **kwargs)
            self.capture()

        def prepared(batch, mode, *args, **kwargs):
            self.current = None
            metadata = native_prepare(batch, mode, *args, **kwargs)
            if mode.name == "FULL" and not kwargs.get("for_capture", False):
                lengths = self.decode_lengths(batch)
                if lengths is not None:
                    self.current = (metadata, lengths)
            return metadata

        try:
            self.hook(self.manager, "capture", captured)
            self.hook(self.runner.model_state, "prepare_attn", prepared)
            self.hook(self.manager, "run_fullgraph", self.replay)
        except Exception:
            self.restore()
            raise

    def capture(self):
        from .attention_kv_full_capture import capture_entry

        capture_entry(self)

    def decode_lengths(self, batch):
        return current_decode_lengths(batch)

    def bind(self, graph, descriptor, metadata, plans, inputs, receipt):
        if self.graph is not None or self.manager.hidden_states is None:
            raise RuntimeError("FULL entry already bound or native output missing")
        self.descriptor, self.plans = descriptor, tuple(plans)
        self.binding = self.metadata_key(metadata)
        self.names = frozenset(metadata)
        self.cache_key = cache_binding(self.runner.compilation_config.static_forward_context, self.names)
        self.inputs_key = input_key(inputs)
        self.output, self.output_key = self.manager.hidden_states, input_key((self.manager.hidden_states,))
        self.native_graph = self.manager.graphs.get(descriptor)
        self.graph, self.capture_receipt = graph, receipt

    def prepare_forward(self):
        """Strategy prelude captured once before the original model forward."""

    def restore(self):
        while self.hooks:
            owner, name, replacement = self.hooks[-1]
            if vars(owner).get(name) != replacement:
                raise RuntimeError("FULL hook ownership changed: " + name)
            delattr(owner, name)
            self.hooks.pop()
        self.current, self.graph, self.plans = None, None, ()


class FullWorkerLifecycle:
    """Install before profiling; release profile bindings before native teardown."""

    def __init__(self, runner, attentions, *, observe=None, backend_observer=None):
        self.runner, self.attentions = runner, tuple(attentions)
        self.observe, self.backend_observer = observe, backend_observer
        self.active, self.adapters, self.hooks, self.captures = None, [], [], []

    def install(self):
        initialize, capture = self.runner.initialize_kv_cache, self.runner.capture_model

        def initialized(*args, **kwargs):
            self.release()
            result = initialize(*args, **kwargs)
            try:
                for attention in self.attentions:
                    observer = self.backend_observer(attention.layer_name) if self.backend_observer else None
                    adapter = self.make_adapter(attention, observer)
                    adapter.install()
                    self.adapters.append(adapter)
                self.active = self.make_entry()
                self.active.install()
                self.profiling = bool(kwargs.get("is_profiling", False))
            except ValueError as error:
                # Layout/model mismatch is discovered after native cache
                # allocation, before compile/capture. Restore every owned hook
                # and model fusion together, then retain native execution.
                self.restore()
                callback = getattr(self, "on_unsupported", None)
                if callback is not None:
                    callback(str(error))
                logging.getLogger(__name__).warning("qwen-infra uses native attention: %s", error)
            except Exception:
                self.release()
                raise
            return result

        def captured(*args, **kwargs):
            profile_only = bool(kwargs.get("profile_only", False))
            try:
                measured = capture(*args, **kwargs)
                if self.active is not None and self.active.capture_receipt is not None:
                    receipt = dict(
                        self.active.capture_receipt,
                        profiling=profile_only,
                        native_total_graph_memory_bytes=int(measured),
                    )
                    self.captures.append(receipt)
                return measured
            finally:
                if profile_only:
                    self.release()
                    if self.captures and self.captures[-1]["profiling"]:
                        self.captures[-1]["private_refs_released_before_native_teardown"] = True

        for name, replacement in (("initialize_kv_cache", initialized), ("capture_model", captured)):
            if name in vars(self.runner):
                self.restore()
                raise RuntimeError("Runner instance hook already installed: " + name)
            setattr(self.runner, name, replacement)
            self.hooks.append((name, replacement))

    def release(self):
        if self.active is not None:
            self.active.restore()
            self.active = None
        while self.adapters:
            self.adapters[-1].restore()
            self.adapters.pop()

    def restore(self):
        self.release()
        while self.hooks:
            name, replacement = self.hooks[-1]
            if vars(self.runner).get(name) is not replacement:
                raise RuntimeError("Runner FULL lifecycle hook ownership changed")
            delattr(self.runner, name)
            self.hooks.pop()
