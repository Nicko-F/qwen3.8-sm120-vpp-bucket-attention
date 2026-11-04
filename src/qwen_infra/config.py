"""CPU-only launch configuration shared by the CLI and Python examples."""

from dataclasses import dataclass
from qwen_infra.owners import OWNER_ABI
from qwen_infra.limits import MAX_SEQUENCE_LENGTH

SERVE_PROFILES = ("native", "decode-bucket", "decode-fused")
PREFILL_PROFILES = ("cpp", "vpp")


def decode_options(profile):
    if profile not in SERVE_PROFILES:
        raise ValueError(f"Unknown serving profile: {profile}")
    if profile == "native":
        return {}
    fused = profile == "decode-fused"
    additional = {"qwen_infra_attention_fusions": fused}
    if fused:
        # Stamp before model construction and compiler cache configuration.
        additional["qwen_infra_attention_owner_abi"] = OWNER_ABI
    return dict(
        worker_cls="qwen_infra.worker.DecodeWorker",
        dtype="bfloat16",
        kv_cache_dtype="nvfp4",
        attention_config={"backend": "FLASHINFER"},
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        async_scheduling=False,
        enable_prefix_caching=False,
        compilation_config={
            "mode": 3,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "compile_ranges_endpoints": [32, 2048],
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32],
        },
        additional_config=additional,
    )


@dataclass(frozen=True)
class PrefillConfig:
    model: str
    profile: str = "vpp"
    chunk_size: int = 2048
    max_model_len: int = 32768
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.85
    transport: str = "segment-dma"

    def engine_options(self):
        if not self.model or self.profile not in PREFILL_PROFILES:
            raise ValueError("A model and cpp/vpp profile are required")
        if not 1 <= self.chunk_size <= 16384 or self.max_model_len <= 1:
            raise ValueError("Use a positive chunk budget up to 16384 and a positive context capacity")
        if not 1 <= self.max_num_seqs or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("Invalid sequence or GPU memory budget")
        if self.transport not in ("segment-dma", "nccl"):
            raise ValueError("Expected segment-dma or nccl transport")
        options = dict(
            model=self.model,
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            dtype="bfloat16",
            kv_cache_dtype="nvfp4",
            attention_config={"backend": "FLASHINFER"},
            max_num_batched_tokens=self.chunk_size,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=False,
            async_scheduling=False,
            mamba_cache_mode="none",
            compilation_config={"mode": 3, "cudagraph_mode": "PIECEWISE"},
        )
        if self.profile == "vpp":
            fast = self.transport == "segment-dma"
            options.update(
                worker_cls="qwen_infra.vpp.worker.FoldedWorker",
                distributed_executor_backend="qwen_infra.vpp.executor.FoldedExecutor",
                additional_config=dict(
                    qwen_infra_vpp_rank_local=True,
                    qwen_infra_vpp_transport="segment_dma" if fast else "native",
                    qwen_infra_vpp_direct_inputs=fast,
                    qwen_infra_vpp_direct_outputs=fast,
                    qwen_infra_vpp_static_scales=fast,
                ),
            )
        return options


def decode_contract(text, *, tp, pp, dtype, cache_dtype, max_model_len, fused=False):
    """Return a native-fallback reason using actual geometry, before allocation."""
    if pp != 1 or tp < 1:
        return "bucket decode requires one pipeline stage"
    if str(dtype).split(".")[-1] != "bfloat16" or cache_dtype != "nvfp4":
        return "bucket decode requires BF16 queries and NVFP4 KV cache"
    if not 1 <= max_model_len <= MAX_SEQUENCE_LENGTH:
        return "context capacity exceeds the signed-32-bit page-table arithmetic bound"
    q = getattr(text, "num_attention_heads", 0)
    kv = getattr(text, "num_key_value_heads", 0)
    dim = getattr(text, "head_dim", 0)
    if q != 6 * tp or kv != tp or dim != 256:
        return "bucket decode requires per-rank Q6/KV1/head dimension 256"
    if fused and (getattr(text, "hidden_size", 0) != 5120 or getattr(text, "rms_norm_eps", None) != 1e-6):
        return "the fused norm producer requires H5120 and epsilon 1e-6"
    return None
