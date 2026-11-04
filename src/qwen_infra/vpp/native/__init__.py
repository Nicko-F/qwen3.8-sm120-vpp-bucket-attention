"""Lazy native CUDA DMA/NCCL binding; C++ owns completion and slot lifetime."""

import os
from pathlib import Path

_MODULE = None


def load():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load as load_extension
    from qwen_infra.cache import build_directory
    from qwen_infra.runtime_paths import compiler_inputs, nccl_inputs

    cuda, cccl = compiler_inputs()
    nccl_include, nccl_library = nccl_inputs()
    os.environ.setdefault("MAX_JOBS", "2")
    if not os.environ["MAX_JOBS"].isdigit() or int(os.environ["MAX_JOBS"]) < 1:
        raise ValueError("MAX_JOBS must be a positive compiler job count")
    source = Path(__file__).resolve().parent
    sources = [
        str(source / name)
        for name in (
            "interface.cpp",
            "slot_pool.cpp",
            "host_region.cpp",
            "device_pool.cpp",
            "nccl_edge.cpp",
            "dma_edge.cpp",
            "kv_copy.cpp",
            "bindings.cpp",
        )
    ]
    _MODULE = load_extension(
        "qwen_infra_vpp_native",
        sources=sources,
        extra_cflags=["-O2", "-Wall", "-Wextra"],
        extra_include_paths=[str(cuda / "include"), str(nccl_include), str(cccl)],
        extra_ldflags=[str(nccl_library), "-ldl", f"-Wl,-rpath,{nccl_library.parent}"],
        with_cuda=True,
        build_directory=str(build_directory("vpp-native")),
        verbose=True,
    )
    return _MODULE
