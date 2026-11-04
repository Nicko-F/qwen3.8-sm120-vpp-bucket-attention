import os
from pathlib import Path
import subprocess
import sys


def _run(program, environment):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), **environment)
    return subprocess.check_output([sys.executable, "-c", program], text=True, env=env, timeout=15)


def test_dependency_discovery_uses_explicit_toolkit_without_site_packages(tmp_path):
    cuda = tmp_path / "toolkit"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").touch()
    cccl = tmp_path / "external-cccl"
    (cccl / "nv").mkdir(parents=True)
    (cccl / "nv/target").touch()
    program = 'from qwen_infra.runtime_paths import compiler_inputs; print(*compiler_inputs(), sep="\\n")'
    output = _run(program, {"CUDA_HOME": str(cuda), "QWEN_INFRA_CCCL_INCLUDE": str(cccl)})
    assert output.splitlines() == [str(cuda.resolve()), str(cccl.resolve())]


def test_nccl_home_supports_external_lib64_without_global_install(tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "lib64").mkdir()
    (tmp_path / "include/nccl.h").touch()
    (tmp_path / "lib64/libnccl.so.2").touch()
    output = _run(
        'from qwen_infra.runtime_paths import nccl_inputs; print(*nccl_inputs(), sep="\\n")',
        {"NCCL_HOME": str(tmp_path)},
    )
    assert output.splitlines() == [str(tmp_path / "include"), str(tmp_path / "lib64/libnccl.so.2")]


def test_private_toolkit_is_rediscovered_after_bootstrap_environment_ends(tmp_path):
    cuda = tmp_path / "cuda-toolkit/nvidia/cu13"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "include/nv").mkdir(parents=True)
    (cuda / "bin/nvcc").touch()
    (cuda / "include/nv/target").touch()
    program = (
        'import os\nos.environ.pop("CUDA_HOME",None)\nos.environ.pop("QWEN_INFRA_CUDA_HOME",None)\n'
        "from qwen_infra.runtime_paths import compiler_inputs\nprint(*compiler_inputs(prefix="
        + repr(str(tmp_path))
        + '),sep="\\n")'
    )
    output = _run(program, {"QWEN_INFRA_CCCL_INCLUDE": str(cuda / "include")})
    assert output.splitlines() == [str(cuda.resolve()), str((cuda / "include").resolve())]
