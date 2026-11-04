"""Discover dependencies in virtual environments, system packages or overrides."""

import importlib.util
import os
from pathlib import Path
import shutil
import hashlib
import sys


def package_directory(name):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return Path(spec.origin).parent if spec.origin else None


def compiler_inputs(*, prefix=None):
    configured = os.environ.get("QWEN_INFRA_CUDA_HOME") or os.environ.get("CUDA_HOME")
    compiler = shutil.which("nvcc")
    private = Path(prefix or sys.prefix) / "cuda-toolkit/nvidia/cu13"
    packaged = package_directory("nvidia.cu13")
    if configured:
        cuda = Path(configured).expanduser().resolve()
    elif (private / "bin/nvcc").is_file():
        cuda = private.resolve()
    elif packaged is not None and (packaged / "bin/nvcc").is_file():
        cuda = packaged.resolve()
    elif compiler:
        cuda = Path(compiler).resolve().parent.parent
    else:
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except ImportError:
            CUDA_HOME = None
        if not CUDA_HOME:
            raise RuntimeError("Set CUDA_HOME to a CUDA toolkit containing bin/nvcc")
        cuda = Path(CUDA_HOME).resolve()
    if not (cuda / "bin/nvcc").is_file():
        raise FileNotFoundError("CUDA compiler missing: " + str(cuda / "bin/nvcc"))
    explicit = os.environ.get("QWEN_INFRA_CCCL_INCLUDE")
    candidates = (
        [Path(explicit).expanduser().resolve()] if explicit else [cuda / "include/cccl", cuda / "include"]
    )
    for name, relative in (
        ("flashinfer", "data/cccl/libcudacxx/include"),
        ("nvidia.cu13", "include/cccl"),
        ("nvidia.cuda_cccl", "include"),
    ):
        package = package_directory(name)
        if package is not None:
            candidates.append(package / relative)
    cccl = next((path for path in candidates if (path / "nv/target").is_file()), None)
    if cccl is None:
        raise FileNotFoundError(
            "CCCL headers missing; set QWEN_INFRA_CCCL_INCLUDE to the directory containing nv/target"
        )
    return cuda, cccl


def nccl_inputs():
    configured = os.environ.get("NCCL_HOME")
    root = Path(configured).expanduser().resolve() if configured else package_directory("nvidia.nccl")
    if root is None:
        raise FileNotFoundError("NCCL not discovered; install nvidia-nccl or set NCCL_HOME")
    libraries = [root / sub / "libnccl.so.2" for sub in ("lib", "lib64")]
    library = next((path for path in libraries if path.is_file()), None)
    if not (root / "include/nccl.h").is_file() or library is None:
        raise FileNotFoundError("NCCL_HOME must provide include/nccl.h and lib[/64]/libnccl.so.2")
    return root / "include", library


def prepare_jit_environment():
    """Expose only cuBLAS headers/libs beside an independently chosen toolkit.

    CUDA compiler headers must come from CUDA_HOME. Adding a newer Torch CUDA
    package's entire include directory mixes toolchain ABIs, so expose only
    cuBLAS through a private cache overlay. Existing user paths are preserved.
    """
    from qwen_infra.cache import build_directory

    cuda, _ = compiler_inputs()
    os.environ["CUDA_HOME"] = str(cuda)
    configured = os.environ.get("CUBLAS_HOME")
    roots = (
        [Path(configured).expanduser().resolve()]
        if configured
        else [package_directory("nvidia.cublas"), package_directory("nvidia.cu13")]
    )
    root = next((path for path in roots if path is not None and (path / "include/cublas.h").is_file()), None)
    if root is None:
        return  # A system toolkit may already supply cuBLAS itself.
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    overlay = build_directory("cublas-" + identity)
    sources = [
        (root / "include" / name, overlay / "include" / name)
        for name in ("cublas.h", "cublasLt.h", "cublasXt.h", "cublas_api.h", "cublas_v2.h")
    ]
    for name in ("libcublas.so", "libcublasLt.so"):
        choices = [root / "lib" / name, *sorted((root / "lib").glob(name + ".*"))]
        library = next((path for path in choices if path.is_file()), None)
        if library is not None:
            sources.append((library, overlay / "lib" / name))
    for source, destination in sources:
        if not source.is_file():
            raise FileNotFoundError("Missing cuBLAS dependency: " + str(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError("Unexpected existing compiler overlay alias: " + str(destination))
        elif not destination.exists():
            destination.symlink_to(source)
        else:
            raise RuntimeError("Refusing to replace a non-symlink compiler overlay file: " + str(destination))
    for variable, directory in (
        ("CPATH", overlay / "include"),
        ("LIBRARY_PATH", overlay / "lib"),
        ("LD_LIBRARY_PATH", root / "lib"),
    ):
        existing = os.environ.get(variable)
        os.environ[variable] = str(directory) + (os.pathsep + existing if existing else "")
