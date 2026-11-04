"""Opt-in mixed-dtype Gemma/FP4 producer; original GEMMs and SUM unchanged."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess


def scale_offset(row, group):
    if type(row) is not int or type(group) is not int or row < 0 or not 0 <= group < 320:
        raise ValueError("Expected a logical H5120/group16 scale coordinate")
    return (row // 128) * 40960 + (group // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4


def build(run):
    import fcntl
    from qwen_infra.cache import ROOT, guard
    from .nvfp4_bucket_cuda import compiler_inputs

    guard()
    cuda, cccl = compiler_inputs()
    source = Path(__file__).with_suffix(".cu")
    flags = [
        "-std=c++17",
        "-O3",
        "--fmad=false",
        "-arch=sm_120f",
        "--shared",
        "-Xcompiler=-fPIC",
        "-Xptxas=-v",
        "-lineinfo",
        "-I" + str(cccl),
    ]
    version = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True, timeout=10)
    digest = hashlib.sha256(source.read_bytes() + repr(flags).encode() + version.encode()).hexdigest()
    cache = ROOT / "cache/nvfp4/norm_pack" / digest[:16]
    cache.mkdir(parents=True, exist_ok=True)
    library, log, receipt = cache / "norm_pack.so", cache / "build.log", cache / "build.json"
    argv = [str(cuda / "bin/nvcc"), *flags, str(source), "-o", str(library)]
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            with log.open("x") as output:
                process = subprocess.Popen(
                    argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    status = process.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    raise
            if status:
                raise RuntimeError("Preserved norm/FP4 build failed: " + str(log))
            receipt.write_text(
                json.dumps(
                    dict(
                        identity=digest,
                        argv=argv,
                        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    )
                )
                + "\n"
            )
        if not receipt.is_file() or json.loads(receipt.read_text())["identity"] != digest:
            raise RuntimeError("Library lacks its successful immutable build receipt")
    lib = ctypes.CDLL(str(library))
    lib.nvfp4_norm_pack_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_void_p]
    lib.nvfp4_norm_pack_launch.restype = ctypes.c_int
    lib.nvfp4_norm_pack_metadata.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.nvfp4_norm_pack_metadata.restype = ctypes.c_int
    values = (ctypes.c_int * 3)()
    if lib.nvfp4_norm_pack_metadata(values):
        raise RuntimeError("Cannot read compiled producer resources")
    metadata = dict(
        registers=values[0],
        shared_bytes=values[1],
        local_bytes=values[2],
        threads_per_cta=320,
        ctas_per_row=1,
        identity=digest,
        library=str(library),
        build_log=str(log),
        weight_math="float(raw BF16 gamma)+1, no FP32-effective-gamma downcast",
    )
    return lib, metadata


def register_producer(
    torch,
    run,
    *,
    observer=None,
    permit=None,
    native_fallback=None,
    route_observer=None,
    observer_control=None,
):
    """Cold registration shared by isolated module and native compiler adapter."""
    if (permit is None) != (native_fallback is None) or (
        permit is not None and (not callable(permit) or not callable(native_fallback))
    ):
        raise ValueError("A conditional producer requires its paired native fallback")
    lib, metadata = build(run)

    @torch.library.custom_op("qwen_sm120::gemma_nvfp4_pack", mutates_args=(), device_types="cuda")
    def producer(
        x: torch.Tensor, residual: torch.Tensor, raw_gamma: torch.Tensor, global_inverse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        if (
            not 1 <= rows <= 8192
            or tuple(x.shape) != (rows, 5120)
            or residual.shape != x.shape
            or x.dtype != torch.bfloat16
            or residual.dtype != x.dtype
            or raw_gamma.dtype != torch.bfloat16
            or tuple(raw_gamma.shape) != (5120,)
            or global_inverse.dtype != torch.float32
            or global_inverse.numel() != 1
            or any(
                t.device != x.device or not t.is_contiguous() or t.data_ptr() % 16
                for t in (x, residual, raw_gamma)
            )
            or global_inverse.device != x.device
        ):
            raise ValueError("Original disjoint BF16/Gemma/FP32 global scale contract changed")
        fused = permit is None or bool(permit())
        observing = observer_control is None or observer_control.get("enabled", False)
        if observing:
            if observer is not None:
                observer(rows, raw_gamma.data_ptr())
            if route_observer is not None:
                route_observer("fused" if fused else "native", rows)
        if not fused:
            return native_fallback(x, residual, raw_gamma, global_inverse)
        q = torch.empty((rows, 2560), dtype=torch.uint8, device=x.device)
        scale = torch.empty((((rows + 127) // 128) * 128, 80), dtype=torch.int32, device=x.device)
        updated = torch.empty_like(residual)
        status = lib.nvfp4_norm_pack_launch(
            *[
                ctypes.c_void_p(t.data_ptr())
                for t in (x, residual, raw_gamma, global_inverse, q, scale, updated)
            ],
            rows,
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status:
            raise RuntimeError("Norm/FP4 CUDA launch status " + str(status))
        return q, scale, updated

    @producer.register_fake
    def fake(x, residual, raw_gamma, global_inverse):
        return (
            x.new_empty((x.shape[0], 2560), dtype=torch.uint8),
            x.new_empty((((x.shape[0] + 127) // 128) * 128, 80), dtype=torch.int32),
            torch.empty_like(residual),
        )

    return producer, metadata
