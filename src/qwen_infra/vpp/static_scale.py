# SPDX-License-Identifier: Apache-2.0
"""Scoped host-validation hoist for vLLM a9fe30e9 / FlashInfer 0.7.1.

The native writer still owns quantization, group16 scales, cache
mutation and the device positive/finite check. No dependency is monkeypatched.
Only read-only global scales without a KV quant method are eligible here.
"""

import math


def _native_api():
    import torch
    from flashinfer.page import (
        _nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_kernel as writer,
    )
    from flashinfer.utils import TensorLayout
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl, _nvfp4_kv_views

    return torch, Attention, FlashInferImpl, _nvfp4_kv_views, writer, TensorLayout


class _Writer:
    def __init__(self, owner, layer, api):
        torch, _, _, self.views, self.kernel, self.layouts = api
        self.owner, self.layer = owner, layer
        self.impl, self.cache = layer.impl, layer.kv_cache
        self.k, self.v = layer._k_scale, layer._v_scale
        for value in (self.k, self.v):
            if (
                value.dtype != torch.float32
                or not value.is_cuda
                or value.device != self.cache.device
                or value.numel() != 1
                or not value.is_contiguous()
            ):
                raise ValueError("Static KV scales require contiguous CUDA FP32 scalars")
        # Same native scalar rule, once after load/cache binding, not each write.
        self.values = torch.stack([self.k.reshape(()), self.v.reshape(())]).detach().cpu().tolist()
        if any(not math.isfinite(v) or v <= 0 for v in self.values):
            raise ValueError("Static KV scales must be positive and finite")
        self.epoch = owner.epoch
        self.original = self.impl.do_kv_cache_update
        self.had_override = "do_kv_cache_update" in self.impl.__dict__
        self.callback = self.write

    def write(self, layer, key, value, cache, slots):
        if (
            not self.owner.enabled
            or self.epoch != self.owner.epoch
            or layer is not self.layer
            or cache is not self.cache
            or layer._k_scale is not self.k
            or layer._v_scale is not self.v
        ):
            return self.original(layer, key, value, cache, slots)
        data, scales, layout = self.views(cache, self.impl.num_kv_heads)
        # The public wrapper's dtype/device/shape checks are repeated by this
        # native C++ entry. Keep all four cache writes and the outer vLLM op.
        return self.kernel(key, value, slots, *data, *scales, self.k, self.v, self.layouts[layout].value)

    def restore(self):
        if self.impl.do_kv_cache_update is not self.callback:
            raise RuntimeError("KV backend changed outside its scale-handle owner")
        if self.had_override:
            self.impl.do_kv_cache_update = self.original
        else:
            del self.impl.do_kv_cache_update


class StaticScaleWriters:
    """Cold lifecycle owner; refresh only while the runner is drained.

    In-place mutation is NOT detected by Tensor identity or _version. Native
    reload/cache rebuild hooks invalidate before mutation and refresh afterwards;
    arbitrary external scale mutation is outside this read-only experiment.
    Native CUDA still loads actual scalars and checks their values every write.
    """

    def __init__(self):
        self.epoch = 0
        self.enabled = False
        self.writers = []

    def invalidate(self):
        self.epoch += 1
        if self.enabled:
            for writer in self.writers:
                writer.restore()
        self.writers.clear()

    def set_enabled(self, enabled):
        if enabled == self.enabled:
            return
        for writer in self.writers:
            if enabled:
                writer.impl.do_kv_cache_update = writer.callback
            else:
                writer.restore()
        self.enabled = enabled

    def refresh(self, model):
        if self.writers:
            raise RuntimeError("Invalidate handles before refreshing")
        api = _native_api()
        torch, Attention, FlashInferImpl, *_ = api
        prepared = []
        for layer in model.modules():
            if type(layer) is not Attention or type(layer.impl) is not FlashInferImpl:
                continue
            impl, cache = layer.impl, layer.kv_cache
            if (
                not impl.nvfp4_fa2
                or impl.kv_sharing_target_layer_name is not None
                or getattr(layer, "quant_method", None) is not None
                or not cache.is_cuda
                or cache.dtype != torch.uint8
                or cache.ndim != 4
            ):
                continue  # Unknown/dynamic/placeholder backend retains native validation.
            prepared.append(_Writer(self, layer, api))
        # Validate all before installing any; a failed cold refresh leaves native.
        if self.enabled:
            for writer in prepared:
                writer.impl.do_kv_cache_update = writer.callback
        self.writers = prepared

    def receipt(self):
        return dict(
            enabled=self.enabled,
            epoch=self.epoch,
            layers={w.layer.layer_name: dict(epoch=w.epoch, values=w.values) for w in self.writers},
        )
