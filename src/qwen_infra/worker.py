"""Thin native-vLLM worker integration; no alternate request scheduler."""

import logging

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_worker import Worker

from qwen_infra.config import decode_contract

LOG = logging.getLogger(__name__)


class DecodeWorker(Worker):
    """Install automatic bucket attention before native profiling and capture."""

    def load_model(self, *, load_dummy_weights=False):
        cfg = self.vllm_config
        fused = bool(cfg.additional_config.get("qwen_infra_attention_fusions", False))
        self.full_lifecycle = None
        self.eligible_fa_count = 0
        self.service_fusions = None
        self.fallback_reason = decode_contract(
            cfg.model_config.hf_text_config,
            tp=cfg.parallel_config.tensor_parallel_size,
            pp=cfg.parallel_config.pipeline_parallel_size,
            dtype=cfg.model_config.dtype,
            cache_dtype=cfg.cache_config.cache_dtype,
            max_model_len=cfg.model_config.max_model_len,
            fused=fused,
        )
        if not self.fallback_reason and (
            not self.use_v2_model_runner
            or cfg.speculative_config is not None
            or cfg.lora_config is not None
            or cfg.parallel_config.enable_dbo
            or cfg.parallel_config.prefill_context_parallel_size != 1
            or cfg.parallel_config.decode_context_parallel_size != 1
            or cfg.model_config.enable_sleep_mode
            or cfg.weight_transfer_config is not None
            or cfg.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
        ):
            self.fallback_reason = (
                "bucket decode requires the sequential V2 runner without speculation, LoRA or DBO"
            )
        if not self.fallback_reason and torch.cuda.get_device_capability(self.device) != (12, 0):
            self.fallback_reason = "the extension targets SM120; this device uses native attention"
        if not self.fallback_reason:
            try:
                from qwen_infra.kernels.nvfp4_bucket_cuda import compiler_inputs
                from qwen_infra.kernels.nvfp4_bucket_xqa import native_sources

                compiler_inputs()
                native_sources()
                from vllm.v1.attention.backends.flashinfer import _nvfp4_kv_views  # noqa: F401
            except (ImportError, FileNotFoundError, RuntimeError) as error:
                self.fallback_reason = str(error)
        if self.fallback_reason:
            LOG.warning("qwen-infra uses native attention: %s", self.fallback_reason)
            return super().load_model(load_dummy_weights=load_dummy_weights)
        if fused:
            from qwen_infra.cache import build_directory
            from qwen_infra.owners import OWNER_ABI
            from qwen_infra.kernels.service_fusions import ServiceFusions

            cfg.additional_config = dict(cfg.additional_config, qwen_infra_attention_owner_abi=OWNER_ABI)
            self.kernel_cache = build_directory("fusion")
            self.service_fusions = ServiceFusions(self)
        value = super().load_model(load_dummy_weights=load_dummy_weights)
        from qwen_infra.kernels.nvfp4_kv_bucket_worker import CUDAFullLifecycle

        attentions = [
            layer
            for layer in self.compilation_config.static_forward_context.values()
            if getattr(getattr(layer, "impl", None), "nvfp4_fa2", False)
        ]
        self.eligible_fa_count = len(attentions)
        if not attentions:
            self._disable_fusions()
            LOG.warning("qwen-infra uses native attention: no eligible NVFP4 attention layers")
            return value
        try:
            self.full_lifecycle = CUDAFullLifecycle(self.model_runner, attentions)
            self.full_lifecycle.on_unsupported = self._native_fallback
            self.full_lifecycle.install()
            if self.service_fusions is not None:
                self.service_fusions.bind_and_warm()
                self.service_fusions.observers["enabled"] = False
                for owner in self.full_lifecycle.fusions.owners.values():
                    owner.observe_calls = False
        except ValueError as error:
            if self.full_lifecycle is not None:
                self.full_lifecycle.restore()
            self._native_fallback(str(error))
            LOG.warning("qwen-infra uses native attention: %s", error)
        except Exception:
            if self.full_lifecycle is not None:
                self.full_lifecycle.restore()
                self.full_lifecycle = None
            self._disable_fusions()
            raise
        return value

    def _native_fallback(self, reason=None):
        self._disable_fusions()
        self.full_lifecycle = None
        if reason:
            self.fallback_reason = reason

    def qwen_infra_status(self):
        """Cold metadata RPC; no kernel timing hooks or tensor copies."""
        lifecycle = self.full_lifecycle
        entry = lifecycle.active if lifecycle is not None else None
        captured = (
            [
                dict(rows=member.rows, route=member.route)
                for member in entry.entries.values()
                if member.graph is not None
            ]
            if entry
            else []
        )
        return dict(
            rank=self.rank,
            fallback_reason=self.fallback_reason,
            eligible_fa_count=self.eligible_fa_count,
            bucket_installed=lifecycle is not None,
            captured=captured,
            last_selected_route=getattr(entry, "last_selected_route", None),
            last_padded_rows=getattr(entry, "last_padded_rows", None),
            scratch_bytes=sum(workspace.scratch.numel() for workspace in lifecycle.workspaces.values())
            if lifecycle is not None
            else 0,
            native_graphs_preserved=all(
                member.manager.graphs.get(member.descriptor) is member.native_graph
                for member in entry.entries.values()
                if member.graph is not None
            )
            if entry
            else True,
            attention_fusions=lifecycle is not None and lifecycle.fusions is not None,
            owner_abi=self.vllm_config.additional_config.get("qwen_infra_attention_owner_abi"),
        )

    def _disable_fusions(self):
        owner = getattr(self, "service_fusions", None)
        if owner is not None:
            config = self.compilation_config.inductor_compile_config
            if config.get("post_grad_custom_pre_pass") is owner.prepass:
                del config["post_grad_custom_pre_pass"]
            self.service_fusions = None

    def reload_weights(self, *args, **kwargs):
        if self.full_lifecycle is not None:
            raise RuntimeError("Restart the engine before replacing weights owned by captured bucket graphs")
        return super().reload_weights(*args, **kwargs)

    def shutdown(self):
        if getattr(self, "full_lifecycle", None) is not None:
            self.full_lifecycle.restore()
        self._disable_fusions()
        return super().shutdown()
