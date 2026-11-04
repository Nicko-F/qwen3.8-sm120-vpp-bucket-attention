import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native transport uses Linux shared memory")


def test_host_allocation_has_no_server_memory_or_numa_node_lock(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required")
    repo = Path(__file__).resolve().parents[1]
    source = repo / "src/qwen_infra/vpp/native"
    executable = tmp_path / "host-region"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I" + str(source),
            str(repo / "tests/host_region.cpp"),
            str(source / "host_region.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=60,
    )
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=15)
    if result.returncode == 77:
        pytest.skip(result.stderr)
    assert result.returncode == 0, result.stderr


def test_failed_first_norm_compile_can_retry_same_cache(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("A C compiler is required for the CPU-only library stub")
    toolkit = tmp_path / "toolkit"
    (toolkit / "bin").mkdir(parents=True)
    cccl = toolkit / "include"
    (cccl / "nv").mkdir(parents=True)
    (cccl / "nv/target").touch()
    stub = tmp_path / "stub.c"
    stub.write_text(
        "int nvfp4_norm_pack_launch(void){return 0;}\n"
        "int nvfp4_norm_pack_metadata(int *v){v[0]=12;v[1]=0;v[2]=0;return 0;}\n"
    )
    compiler = toolkit / "bin/nvcc"
    compiler.write_text(
        "#!" + sys.executable + "\n"
        "import os,pathlib,subprocess,sys\n"
        'if "--version" in sys.argv: print("CPU cache-retry test compiler");sys.exit(0)\n'
        'state=pathlib.Path(os.environ["CACHE_RETRY_STATE"])\n'
        'output=pathlib.Path(sys.argv[sys.argv.index("-o")+1])\n'
        "count=int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(count+1))\n"
        'if count==0: output.write_text("interrupted partial library");sys.exit(1)\n'
        'subprocess.run(["cc","-shared","-fPIC",os.environ["CACHE_RETRY_STUB"],"-o",str(output)],check=True)\n'
    )
    compiler.chmod(0o755)
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        QWEN_INFRA_CACHE_DIR=str(tmp_path / "cache"),
        CUDA_HOME=str(toolkit),
        QWEN_INFRA_CCCL_INCLUDE=str(cccl),
        CACHE_RETRY_STATE=str(tmp_path / "attempts"),
        CACHE_RETRY_STUB=str(stub),
    )
    program = (
        "from pathlib import Path\nfrom qwen_infra.kernels.nvfp4_norm_pack import build\n"
        'try: build(Path.cwd())\nexcept RuntimeError: pass\nelse: raise AssertionError("first build must fail")\n'
        'lib,metadata=build(Path.cwd())\nassert metadata["registers"]==12\n'
        "build(Path.cwd())\n"
    )
    subprocess.run([sys.executable, "-c", program], cwd=tmp_path, env=env, check=True, timeout=60)
    assert (tmp_path / "attempts").read_text() == "2"
