"""Explicit native-XQA task backend; its normalized-partial ABI is not U/m/l."""

import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from qwen_infra.runtime_paths import package_directory
from qwen_infra.limits import MAX_TABLE_WIDTH, MAX_SEQUENCE_LENGTH


TILE = 256
# Physical-block rounding may make native kernel-page capacity exceed4096.
# Bound the ABI's signed32-bit width*32+TILE-1 arithmetic, not live KV length.


def task_plan(lengths, *, ctas=170, cyclic=False, aligned=False, table_width=None):
    """CPU interval/cyclic/request-aligned oracle, outside timed Graph replay."""
    if (
        not 1 <= len(lengths) <= 32
        or not len(lengths) <= ctas <= 340
        or any(type(n) is not int or not 0 <= n <= MAX_SEQUENCE_LENGTH for n in lengths)
        or type(cyclic) is not bool
        or type(aligned) is not bool
        or (cyclic and aligned)
    ):
        raise ValueError("Expected bounded TP4 q1 lengths and CTA budget")
    width = max(1, (max(lengths) + 31) // 32) if table_width is None else table_width
    if type(width) is not int or not 1 <= width <= MAX_TABLE_WIDTH or max(lengths) > width * 32:
        raise ValueError("Native page-table width must cover actual lengths")
    counts = [(n + TILE - 1) // TILE for n in lengths]
    budget = max(2, (sum(counts) + ctas - 1) // ctas)
    if aligned:
        # Deliberately exhaustive CPU oracle; the GPU uses bounded binary search.
        budget = next(
            limit
            for limit in range(1, max(counts, default=0) + 2)
            if sum((n + limit - 1) // limit for n in counts) <= ctas
        )
    splits = min(max(1, ctas // len(lengths)), (width * 32 + TILE - 1) // TILE)
    tasks, requests, prefix = [], [], 0
    buckets = [[0, 0] for _ in range(ctas)]
    for r, n in enumerate(counts):
        begin = len(tasks)
        if cyclic:
            tasks.extend([r, s, n, splits] for s in range(splits))
        elif aligned:
            parts = (n + budget - 1) // budget
            tasks.extend([r, s * n // parts, (s + 1) * n // parts, 1] for s in range(parts))
        elif n:
            for bucket in range(prefix // budget, (prefix + n - 1) // budget + 1):
                lo = max(prefix, bucket * budget) - prefix
                hi = min(prefix + n, (bucket + 1) * budget) - prefix
                if buckets[bucket][0] == buckets[bucket][1]:
                    buckets[bucket][0] = len(tasks)
                tasks.append([r, lo, hi, 1])
                buckets[bucket][1] = len(tasks)
        requests.append([begin, len(tasks) - begin])
        prefix += n
    if aligned:
        buckets[: len(tasks)] = [[slot, slot + 1] for slot in range(len(tasks))]
    active = len(tasks) if cyclic or aligned else (prefix + budget - 1) // budget
    return dict(
        tasks=tasks,
        requests=requests,
        buckets=buckets,
        header=[splits if cyclic else budget, len(tasks), active, max(n for _, n in requests)],
    )


class Params(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_void_p)
            for name in (
                "q",
                "k",
                "v",
                "ks",
                "vs",
                "table",
                "lengths",
                "tasks",
                "requests",
                "buckets",
                "header",
                "semaphores",
                "scratch",
                "output",
            )
        ]
        + [(name, ct.c_int) for name in ("batch", "ctas", "table_width", "mode")]
        + [(name, ct.c_int64) for name in ("kv_page", "kv_token", "sf_page", "sf_token")]
        + [("qscale", ct.c_float), ("vscale", ct.c_float)]
        + [(name, ct.c_void_p) for name in ("gate", "output_scale", "fp8_output")]
    )


def native_sources(source_root=None):
    """Require the actual header/source API, without a version or source-hash gate."""
    package = package_directory("flashinfer")
    selected = source_root or os.environ.get("QWEN_INFRA_XQA_SOURCE")
    if selected is None and package is None:
        raise FileNotFoundError("FlashInfer XQA headers are missing; install flashinfer-python")
    native = Path(selected or package / "data/csrc/xqa").resolve()
    required = (
        "cuda_hint.cuh",
        "defines.h",
        "ldgsts.cuh",
        "mha.h",
        "mhaUtils.cuh",
        "mha_components.cuh",
        "mma.cuh",
        "utils.cuh",
        "utils.h",
        "mha_stdheaders.cuh",
        "platform.h",
        "barriers.cuh",
        "hostUtils.h",
    )
    missing = [name for name in required if not (native / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Native XQA source/header API missing in " + str(native) + ": " + ", ".join(missing)
        )
    return native


def build(run, *, source_root=None, pdl=True, persistent=False, tail=False, query_resources=True):
    """Cold private build; installed XQA headers/source stay read-only."""
    if any(type(option) is not bool for option in (pdl, persistent, tail, query_resources)):
        raise ValueError("PDL, persistent and tail selections must be explicit booleans")
    import fcntl
    from qwen_infra.cache import ROOT, guard
    from .nvfp4_bucket_cuda import compiler_inputs

    guard()
    native = native_sources(source_root)
    generated = Path(__file__).with_name("xqa_task_impl.cuh").read_text()
    changes = "maintained NVFP4_XQA_PERSISTENT/TAIL specializations"
    bridge = Path(__file__).with_suffix(".cu")
    cuda, cccl = compiler_inputs()
    # Match native gen_jit_spec(use_fast_math=True), in addition to cpp_ext's
    # relaxed constexpr support. This aligns the native build configuration.
    flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-use_fast_math",
        "-O3",
        "-gencode=arch=compute_120f,code=sm_120f",
        "--shared",
        "-Xcompiler=-fPIC",
        "-Xptxas=-v",
        "-lineinfo",
        "-I" + str(cccl),
        "-I" + str(native),
        "-DNDEBUG=1",
        "-DBEAM_WIDTH=1",
        "-DUSE_INPUT_KV=0",
        "-DUSE_CUSTOM_BARRIER=1",
        "-DHEAD_ELEMS=256",
        "-DHEAD_GRP_SIZE=6",
        "-DINPUT_FP16=0",
        "-DCACHE_ELEM_ENUM=3",
        "-DTOKENS_PER_PAGE=32",
        "-DSLIDING_WINDOW=0",
        "-DLOW_PREC_OUTPUT=0",
        "-DSPEC_DEC=0",
        "-DENABLE_PDL=" + str(int(pdl)),
        "-DNVFP4_XQA_PERSISTENT=" + str(int(persistent)),
        "-DNVFP4_XQA_TAIL=" + str(int(tail)),
    ]
    version = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True, timeout=10)
    headers = sorted(p for p in native.iterdir() if p.suffix in (".h", ".cuh"))
    identity = hashlib.sha256(
        generated.encode()
        + bridge.read_bytes()
        + repr(flags).encode()
        + version.encode()
        + b"".join(p.read_bytes() for p in headers)
    ).hexdigest()
    cache = ROOT / "cache/nvfp4/kv_xqa_task" / identity[:16]
    cache.mkdir(parents=True, exist_ok=True)
    library, log = cache / "xqa.so", cache / "build.log"
    argv = [str(cuda / "bin/nvcc"), *flags, "-I" + str(cache), str(bridge), "-o", str(cache / "xqa.part.so")]
    recipe = dict(
        identity=identity,
        argv=argv,
        compiler_version=version,
        native_source=str(native),
        source_adaptations=changes,
        normalized_partial_abi="native BF16 O + FP32 row max/sum",
        one_cta_per_task=not persistent,
        persistent_bucket_ownership=persistent,
        pdl=pdl,
        final_fp8_tail=tail,
        tail_math="BF16 attention -> BF16 sigmoid/product -> static E4M3",
        library=str(library),
        build_log=str(log),
    )
    suffix = ("_persistent" if persistent else "") + ("_tail" if tail else "")
    (Path(run) / ("xqa_build" + suffix + ".json")).write_text(json.dumps(recipe, indent=2) + "\n")
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (cache / "xqa_task_impl.cuh").write_text(generated)
        if not library.exists():
            with log.open("a") as output:
                process = subprocess.Popen(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                try:
                    status = process.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    raise
            if status:
                raise RuntimeError("Native-XQA task build failed; retained log: " + str(log))
            os.replace(cache / "xqa.part.so", library)
    if not query_resources:
        # Compile/link preparation while devices are busy: do not load CUDA
        # libraries or call cudaFuncGetAttributes/occupancy (which create a context).
        return None, dict(recipe, compiled=True, gpu_resources_queried=False)
    lib = ct.CDLL(str(library))
    for name in ("assign", "compute"):
        method = getattr(lib, "nvfp4_xqa_" + name)
        method.argtypes, method.restype = [Params, ct.c_void_p], ct.c_int
    lib.nvfp4_xqa_resources.argtypes = [ct.POINTER(ct.c_int)]
    lib.nvfp4_xqa_resources.restype = ct.c_int
    lib.nvfp4_xqa_scratch_bytes.argtypes = [ct.c_int]
    lib.nvfp4_xqa_scratch_bytes.restype = ct.c_uint64
    values = (ct.c_int * 13)()
    if lib.nvfp4_xqa_resources(values):
        raise RuntimeError("Native-XQA task resource query failed")
    if (values[0], values[10], values[11], values[12]) != (ct.sizeof(Params), 16, 8, 8):
        raise RuntimeError("Native-XQA task C/Python ABI mismatch")
    resources = dict(
        **recipe,
        registers=values[1],
        shared_bytes=values[2],
        local_bytes=values[3],
        resident_ctas=values[4],
        threads=256,
        scalar_partial_bytes=values[5],
        output_partial_bytes=values[6],
        kv_head_bytes=values[8],
        sf_head_bytes=values[9],
        bucket_range_bytes=values[12],
    )
    lib._nvfp4_xqa_persistent = persistent
    lib._nvfp4_xqa_tail = tail
    (Path(run) / ("xqa_resources" + suffix + ".json")).write_text(json.dumps(resources, indent=2) + "\n")
    return lib, resources


class TaskPlan:
    """One immutable metadata binding; GPU contents refresh once per forward.

    It owns no per-layer Q/cache/output references or normalized partials.
    Sequential FA layers may read the same tasks with independent scratch.
    """

    def __init__(self, torch, lib, batch, *, ctas=170, device="cuda"):
        task_plan([0] * batch, ctas=ctas)
        self.torch, self.lib, self.batch, self.ctas = torch, lib, batch, ctas
        self.capacity = ctas + batch - 1
        self.tasks = torch.empty((self.capacity, 4), dtype=torch.int32, device=device)
        self.requests = torch.empty((batch, 2), dtype=torch.int32, device=device)
        self.buckets = torch.empty((ctas, 2), dtype=torch.int32, device=device)
        self.header = torch.empty(4, dtype=torch.int32, device=device)
        self.binding = None

    @property
    def logical_bytes(self):
        return sum(
            t.numel() * t.element_size() for t in (self.tasks, self.requests, self.buckets, self.header)
        )

    def prepare(self, lengths, *, table_width, cyclic=False, aligned=False):
        if (
            type(cyclic) is not bool
            or type(aligned) is not bool
            or (cyclic and aligned)
            or ((cyclic or aligned) and getattr(self.lib, "_nvfp4_xqa_persistent", False))
            or type(table_width) is not int
            or not 1 <= table_width <= MAX_TABLE_WIDTH
            or lengths.device != self.tasks.device
            or lengths.dtype != self.torch.int32
            or tuple(lengths.shape) != (self.batch,)
            or not lengths.is_contiguous()
        ):
            raise ValueError("Expected one native device-length buffer and fixed page-table capacity")
        key = (
            lengths.data_ptr(),
            tuple(lengths.shape),
            tuple(lengths.stride()),
            str(lengths.dtype),
            str(lengths.device),
            table_width,
            cyclic,
            aligned,
        )
        if self.binding is not None:
            if key != self.binding:
                raise RuntimeError("A captured shared task-plan binding cannot be replaced")
            return self
        self.lengths, self.binding = lengths, key
        self.params = Params()
        for name in ("lengths", "tasks", "requests", "buckets", "header"):
            setattr(self.params, name, getattr(self, name).data_ptr())
        self.params.batch, self.params.ctas = self.batch, self.ctas
        self.params.table_width, self.params.mode = table_width, (1 if cyclic else 2 if aligned else 0)
        return self

    def assign(self):
        if self.binding is None:
            raise RuntimeError("Shared task-plan metadata generation is released")
        stream = self.torch.cuda.current_stream(self.tasks.device).cuda_stream
        error = self.lib.nvfp4_xqa_assign(self.params, stream)
        if error:
            raise RuntimeError("Native-XQA shared task assign launch error " + str(error))

    def release_binding(self):
        """Cold lifecycle only, after every graph using this plan is restored."""
        for name in ("lengths", "params"):
            if hasattr(self, name):
                delattr(self, name)
        self.binding = None


class Workspace:
    """One stable native partial/counter generation; sequential Graph replay only."""

    tile = TILE

    def __init__(self, torch, lib, batch, *, ctas=170, device="cuda", plan=None, tail_library=None):
        task_plan([0] * batch, ctas=ctas)
        self.torch, self.lib, self.batch, self.ctas = torch, lib, batch, ctas
        if getattr(lib, "_nvfp4_xqa_tail", False):
            raise ValueError("The plan/default compute library must retain BF16 output")
        self.tail_library = tail_library
        if tail_library is not None and (
            not getattr(tail_library, "_nvfp4_xqa_tail", False)
            or getattr(tail_library, "_nvfp4_xqa_persistent", False)
            != getattr(lib, "_nvfp4_xqa_persistent", False)
        ):
            raise ValueError("Tail library must match the default task ownership specialization")
        self.capacity = ctas + batch - 1
        self.plan = plan if plan is not None else TaskPlan(torch, lib, batch, ctas=ctas, device=device)
        if (
            self.plan.torch is not torch
            or self.plan.lib is not lib
            or self.plan.batch != batch
            or self.plan.ctas != ctas
        ):
            raise ValueError("Shared plan must match backend, batch and CTA capacity")
        self.tasks, self.requests, self.buckets, self.header = (
            self.plan.tasks,
            self.plan.requests,
            self.plan.buckets,
            self.plan.header,
        )
        self.semaphores = torch.zeros(batch, dtype=torch.int32, device=device)
        size = lib.nvfp4_xqa_scratch_bytes(self.capacity)
        if not size:
            raise ValueError("Native compact scratch capacity is unsupported")
        self.scratch = torch.empty(size, dtype=torch.uint8, device=device)
        self.output = torch.empty((batch, 6, 256), dtype=torch.bfloat16, device=device)
        if self.tasks.device != self.output.device:
            raise ValueError("Shared plan and layer workspace must use the same device")

    @property
    def logical_bytes(self):
        return self.plan.logical_bytes + self.exclusive_bytes

    @property
    def exclusive_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.semaphores, self.scratch, self.output))

    @property
    def pointers(self):
        return tuple(
            t.data_ptr()
            for t in (
                self.tasks,
                self.requests,
                self.buckets,
                self.header,
                self.semaphores,
                self.scratch,
                self.output,
            )
        )

    def bind(
        self,
        q,
        data,
        scales,
        table,
        lengths,
        *,
        qscale,
        vscale,
        cyclic=False,
        aligned=False,
        output=None,
        gate=None,
        output_scale=None,
    ):
        torch = self.torch
        k, v = data
        ks, vs = scales
        out = self.output if output is None else output
        tail = gate is not None
        out_dtype = torch.float8_e4m3fn if tail else q.dtype
        if tail and (
            self.tail_library is None
            or gate.device != q.device
            or gate.dtype != q.dtype
            or gate.shape != q.shape
            or not gate.is_contiguous()
            or output_scale is None
            or output_scale.device != q.device
            or output_scale.dtype != torch.float32
            or output_scale.numel() != 1
            or not output_scale.is_contiguous()
        ):
            raise ValueError("Tail requires a matching compiled library, BF16 gate and static FP32 scale")
        if not tail and output_scale is not None:
            raise ValueError("OUT scale requires a gate")
        tensors = (q, k, v, ks, vs, table, lengths, out)
        if (
            type(cyclic) is not bool
            or type(aligned) is not bool
            or (cyclic and aligned)
            or any(t.device != q.device for t in tensors)
            or q.device != self.output.device
            or q.device.type != "cuda"
            or q.dtype != torch.bfloat16
            or tuple(q.shape) != (self.batch, 6, 256)
            or not q.is_contiguous()
            or out.dtype != out_dtype
            or out.shape != q.shape
            or not out.is_contiguous()
            or k.dtype != torch.uint8
            or v.dtype != k.dtype
            or k.shape != v.shape
            or k.stride() != v.stride()
            or tuple(k.shape[1:]) != (32, 1, 128)
            or k.stride(-1) != 1
            or ks.dtype != torch.float8_e4m3fn
            or vs.dtype != ks.dtype
            or ks.shape != vs.shape
            or tuple(ks.shape) != (*k.shape[:3], 16)
            or ks.stride() != vs.stride()
            or ks.stride(-1) != 1
            or table.dtype != torch.int32
            or table.ndim != 2
            or table.shape[0] != self.batch
            or not table.is_contiguous()
            or not 1 <= table.shape[1] <= MAX_TABLE_WIDTH
            or lengths.dtype != torch.int32
            or tuple(lengths.shape) != (self.batch,)
            or not lengths.is_contiguous()
        ):
            raise ValueError("Expected native packed page32/Q6/KV1/D256 views and stable dense metadata")
        if not all(math.isfinite(float(s)) for s in (qscale, vscale)):
            raise ValueError("Expected finite independent BMM1/BMM2 scales")
        self.plan.prepare(lengths, table_width=table.shape[1], cyclic=cyclic, aligned=aligned)
        self.bound = (q, data, scales, table, lengths, out)
        self.tail_bindings = (gate, output_scale)
        self.compute_library = self.tail_library if tail else self.lib
        self.params = Params(
            *[
                t.data_ptr()
                for t in (
                    q,
                    k,
                    v,
                    ks,
                    vs,
                    table,
                    lengths,
                    self.tasks,
                    self.requests,
                    self.buckets,
                    self.header,
                    self.semaphores,
                    self.scratch,
                    self.output if tail else out,
                )
            ],
            self.batch,
            self.ctas,
            table.shape[1],
            (1 if cyclic else 2 if aligned else 0),
            k.stride(0),
            k.stride(1),
            ks.stride(0),
            ks.stride(1),
            qscale,
            vscale,
            gate.data_ptr() if tail else 0,
            output_scale.data_ptr() if tail else 0,
            out.data_ptr() if tail else 0,
        )
        return self

    def _call(self, name):
        stream = self.torch.cuda.current_stream(self.output.device).cuda_stream
        error = getattr(self.compute_library, "nvfp4_xqa_" + name)(self.params, stream)
        if error:
            raise RuntimeError("Native-XQA task " + name + " launch error " + str(error))

    def assign(self):
        self.plan.assign()

    def compute(self):
        self._call("compute")

    def run(self, *, assign=True):
        if assign:
            self.assign()
        self.compute()
        return self.bound[-1]

    def inspect_plan(self):
        count = int(self.header[1].item())
        return dict(
            tasks=self.tasks[:count].cpu().tolist(),
            requests=self.requests.cpu().tolist(),
            buckets=self.buckets.cpu().tolist(),
            header=self.header.cpu().tolist(),
        )

    def release_bindings(self):
        for name in ("bound", "params", "tail_bindings", "compute_library"):
            if hasattr(self, name):
                delattr(self, name)
