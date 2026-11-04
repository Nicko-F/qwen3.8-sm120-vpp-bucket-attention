"""Native-cache adapter and CUDA compiler discovery."""

from contextlib import contextmanager
from qwen_infra.runtime_paths import compiler_inputs as compiler_inputs


def cuda_metadata_key(metadata, rows):
    from .attention_kv_fullgraph import input_key

    decode = metadata.decode
    if (
        metadata.num_actual_tokens != rows
        or metadata.num_decodes != rows
        or metadata.num_decode_tokens != rows
        or metadata.num_prefill_tokens != 0
        or decode.q_len_per_req != 1
        or decode.kernel.value != "xqa"
        or decode.block_tables.shape[0] != rows
        or decode.seq_lens.shape != (rows,)
    ):
        raise ValueError("Expected original native padded q1 XQA metadata")
    return input_key((decode.block_tables, decode.seq_lens))


class NativeCUDAAdapter:
    """Compute-only callback; native writer, Graph staging and cache ownership stay."""

    def __init__(self, torch, attention, workspace, *, observer=None, workspaces=None, backend="legacy"):
        from vllm.v1.attention.backends.flashinfer import _nvfp4_kv_views

        self.torch, self.attention, self.impl = torch, attention, attention.impl
        self.workspace, self.observer = workspace, observer
        if backend not in ("legacy", "xqa"):
            raise ValueError("Expected legacy/xqa callback backend")
        self.backend, self.route = backend, None
        self.workspaces = workspaces if workspaces is not None else {workspace.batch: workspace}
        self.plan_bindings = {}
        self.cache = attention.kv_cache
        self.data, self.scales, layout = _nvfp4_kv_views(self.cache, 1)
        if (
            not self.impl.nvfp4_fa2
            or self.impl.scale != 1 / 16
            or layout != "NHD"
            or self.impl.num_heads != 6
            or self.impl.num_kv_heads != 1
            or self.impl.head_size != 256
            or self.impl.window_left != -1
            or self.impl.sinks is not None
            or self.impl.logits_soft_cap not in (None, 0.0)
            or tuple(self.data[0].shape[1:]) != (32, 1, 128)
            or any(
                t.untyped_storage().data_ptr() != self.cache.untyped_storage().data_ptr()
                for t in (*self.data, *self.scales)
            )
        ):
            raise ValueError("Expected unchanged native NVFP4 TP4 XQA cache/attention contract")
        self.native, self.installed = self.impl.forward, self.call
        self.enabled, self.md, self.lens, self.table = False, None, None, None
        self.assign_tasks = False
        self.tail = None
        self.calls = 0

    def install(self):
        if self.impl.forward != self.native:
            raise RuntimeError("Attention callback already owned")
        self.impl.forward = self.installed

    def prepare_plan(self, metadata, *, workspace=None, route=None):
        rows = metadata.num_actual_tokens
        if self.backend == "xqa":
            if route not in ("aligned", "persistent") or workspace is not self.workspaces.get((rows, route)):
                raise ValueError("Expected the captured XQA row/route workspace")
            self.workspace, self.route = workspace, route
        else:
            self.workspace = self.workspaces[rows]
        key = cuda_metadata_key(metadata, rows)
        if (
            metadata.q_data_type_decode != self.torch.bfloat16
            or metadata.decode.mask is not None
            or metadata.decode.block_tables.dtype != self.torch.int32
            or metadata.decode.seq_lens.dtype != self.torch.int32
        ):
            raise ValueError("Expected original BF16 q1 dense decode metadata")
        owner = (rows, route) if self.backend == "xqa" else rows
        if owner in self.plan_bindings and self.plan_bindings[owner] != key:
            raise RuntimeError("A captured metadata binding cannot be replaced")
        self.plan_bindings[owner] = key
        self.md, self.lens, self.table = metadata, metadata.decode.seq_lens, metadata.decode.block_tables
        if self.backend == "xqa":
            self.workspace.plan.prepare(
                self.lens, table_width=self.table.shape[1], aligned=route == "aligned"
            )
        else:
            self.workspace.prepare_plan(self.lens)
        return key

    @contextmanager
    def capture(self, plan):
        if self.enabled or self.md is None or cuda_metadata_key(self.md, self.workspace.batch) != plan:
            raise RuntimeError("Expected one prepared CUDA capture binding")
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False

    def call(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if not self.enabled:
            if self.observer is not None:
                self.observer("native")
            return self.native(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        if (
            layer is not self.attention
            or kv_cache is not self.cache
            or attn_metadata is not self.md
            or output_scale is not None
            or output_block_scale is not None
            or query.dtype != self.torch.bfloat16
            or query.shape != (self.workspace.batch, 6, 256)
        ):
            raise ValueError("CUDA callback received an unexpected live native binding")
        # This is also the original XQA path's staging, not a BF16 cache expansion.
        query = query.contiguous()
        bmm1 = self.impl.get_xqa_bmm1_scale(layer, attn_metadata.q_data_type_decode)
        bmm2 = self.impl.bmm2_scale
        if bmm2 is None:
            # Native initializes this lazily from the same immutable loaded scale.
            bmm2 = layer._v_scale_float
        if self.backend == "xqa":
            gate, scale, tail_output = self.tail if self.tail is not None else (None, None, output)
            self.workspace.bind(
                query,
                self.data,
                self.scales,
                self.table,
                self.lens,
                qscale=bmm1,
                vscale=bmm2,
                output=tail_output,
                aligned=self.route == "aligned",
                gate=gate,
                output_scale=scale,
            )
        else:
            gate, scale, tail_output = self.tail if self.tail is not None else (None, None, output)
            self.workspace.bind(
                query,
                self.data,
                self.scales,
                self.table,
                self.lens,
                qscale=bmm1,
                vscale=bmm2,
                output=tail_output,
                gate=gate,
                output_scale=scale,
            )
        self.workspace.run(assign=self.assign_tasks)
        self.calls += 1
        if self.observer is not None:
            self.observer("xqa_" + self.route if self.backend == "xqa" else "cuda_bucket")
        return output

    def restore(self):
        if self.impl.forward != self.installed:
            raise RuntimeError("Foreign Attention callback owner")
        self.impl.forward = self.native
        self.md = self.table = self.lens = None
        self.plan_bindings.clear()
