"""CUDA producer for existing Qwen QK/Gate and native NVFP4 cache consumers."""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .nvfp4_bucket_cuda import compiler_inputs


class Front(ct.Structure):
    _fields_ = (
        [
            (n, ct.c_void_p)
            for n in (
                "qkv",
                "qw",
                "kw",
                "cos_sin",
                "positions",
                "slots",
                "k_scale",
                "v_scale",
                "k",
                "v",
                "ks",
                "vs",
                "q",
                "gate",
            )
        ]
        + [
            (n, ct.c_int)
            for n in (
                "rows",
                "qheads",
                "kvheads",
                "rotary",
                "slot_rows",
                "pages",
                "page_size",
                "cos_float",
                "pos_long",
                "slot_long",
                "mrope_h",
                "mrope_w",
            )
        ]
        + [
            (n, ct.c_int64)
            for n in (
                "input_row",
                "cos_row",
                "pos_m",
                "pos_t",
                "data_page",
                "data_row",
                "data_head",
                "sf_page",
                "sf_row",
                "sf_head",
            )
        ]
        + [("epsilon", ct.c_float)]
    )


def rotary_pairs(rotary):
    """CPU-only layout check for128threads, no extra rotary shared buffer."""
    if rotary not in (64, 128, 256):
        raise ValueError("Expected warp-aligned rotary pairs")
    half = rotary // 2
    return [(t, t + half) if t < half else (2 * t, 2 * t + 1) for t in range(128)]


def build(run, *, query_resources=True):
    if type(query_resources) is not bool:
        raise ValueError("Resource query must be an explicit boolean")
    import fcntl
    from qwen_infra.cache import ROOT, guard

    guard()
    cuda, cccl = compiler_inputs()
    source = Path(__file__).with_suffix(".cu")
    flags = [
        "-std=c++17",
        "-O3",
        "-gencode=arch=compute_120f,code=sm_120f",
        "--shared",
        "-Xcompiler=-fPIC",
        "-Xptxas=-v",
        "-lineinfo",
        "-I" + str(cccl),
    ]
    version = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True, timeout=10)
    digest = hashlib.sha256(source.read_bytes() + repr(flags).encode() + version.encode()).hexdigest()
    cache = ROOT / "cache/nvfp4/attention_front" / digest[:16]
    cache.mkdir(parents=True, exist_ok=True)
    library, partial, log = cache / "front.so", cache / "front.part.so", cache / "build.log"
    argv = [str(cuda / "bin/nvcc"), *flags, str(source), "-o", str(partial)]
    recipe = dict(
        identity=digest, argv=argv, compiler_version=version, library=str(library), build_log=str(log)
    )
    (Path(run) / "front_build.json").write_text(json.dumps(recipe, indent=2) + "\n")
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with log.open("a") as stream:
                child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    status = child.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                    raise
            if status:
                raise RuntimeError("Front CUDA build failed: " + str(log))
            os.replace(partial, library)
    if not query_resources:
        return None, dict(recipe, compiled=True, gpu_resources_queried=False)
    lib = ct.CDLL(str(library))
    lib.nvfp4_attention_front.argtypes = [Front, ct.c_void_p]
    lib.nvfp4_attention_front.restype = ct.c_int
    lib.nvfp4_attention_front_resources.argtypes = [ct.POINTER(ct.c_int)]
    lib.nvfp4_attention_front_resources.restype = ct.c_int
    values = (ct.c_int * 5)()
    if lib.nvfp4_attention_front_resources(values) or values[0] != ct.sizeof(Front):
        raise RuntimeError("Front compiled parameter ABI mismatch")
    resources = dict(
        registers=values[1],
        shared_bytes=values[2],
        local_bytes=values[3],
        resident_ctas=values[4],
        threads=128,
        argv=argv,
        library=str(library),
        build_log=str(log),
    )
    (Path(run) / "front_resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    return lib, resources


def launch(
    torch,
    lib,
    qkv,
    positions,
    qw,
    kw,
    cos,
    slots,
    data,
    scales,
    k_global,
    v_global,
    *,
    qheads,
    kvheads,
    rotary,
    epsilon=1e-6,
    mrope_section=None,
):
    rotary_pairs(rotary)
    rows = qkv.shape[0]
    k, v = data
    ks, vs = scales
    if (
        qkv.dtype != torch.bfloat16
        or qkv.ndim != 2
        or qkv.stride(-1) != 1
        or qkv.shape[1] != (qheads * 2 + kvheads * 2) * 256
        or any(t.dtype != torch.bfloat16 or t.shape != (256,) for t in (qw, kw))
        or cos.dtype not in (torch.bfloat16, torch.float32)
        or cos.shape[1] != rotary
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.shape[-1] != rows
        or positions.ndim not in (1, 2)
        or (positions.ndim == 2 and positions.shape[0] != 3)
        or slots.dtype not in (torch.int32, torch.int64)
        or slots.ndim != 1
        or not slots.is_contiguous()
        or slots.numel() > rows
        or k.dtype != torch.uint8
        or v.shape != k.shape
        or k.shape[2:] != (kvheads, 128)
        or ks.dtype != torch.float8_e4m3fn
        or vs.shape != ks.shape
        or ks.shape != (*k.shape[:3], 16)
        or k.stride() != v.stride()
        or ks.stride() != vs.stride()
        or any(t.stride(-1) != 1 for t in (k, v, ks, vs))
        or any(
            t.device != qkv.device for t in (positions, qw, kw, cos, slots, k, v, ks, vs, k_global, v_global)
        )
        or any(
            t.dtype != torch.float32 or t.numel() != 1 or not t.is_contiguous() for t in (k_global, v_global)
        )
    ):
        raise ValueError("Expected native BF16 QKV/Gemma/D256 and NHD-view NVFP4 byte ABI")
    mrope = positions.ndim == 2
    if mrope and (mrope_section is None or len(mrope_section) != 3 or sum(mrope_section) != rotary // 2):
        raise ValueError("Expected original three-axis interleaved MRoPE sections")
    q = qkv.new_empty((rows, qheads * 256))
    gate = torch.empty_like(q)
    values = (qkv, qw, kw, cos, positions, slots, k_global, v_global, k, v, ks, vs, q, gate)
    param = Front(
        *[t.data_ptr() for t in values],
        rows,
        qheads,
        kvheads,
        rotary,
        slots.numel(),
        k.shape[0],
        k.shape[1],
        int(cos.dtype == torch.float32),
        int(positions.dtype == torch.int64),
        int(slots.dtype == torch.int64),
        mrope_section[1] if mrope else 0,
        mrope_section[2] if mrope else 0,
        qkv.stride(0),
        cos.stride(0),
        positions.stride(0) if mrope else 0,
        positions.stride(-1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        ks.stride(0),
        ks.stride(1),
        ks.stride(2),
        epsilon,
    )
    status = lib.nvfp4_attention_front(param, torch.cuda.current_stream(qkv.device).cuda_stream)
    if status:
        raise RuntimeError("Fused front launch status " + str(status))
    return q, gate
