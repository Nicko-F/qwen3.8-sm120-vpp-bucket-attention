"""Exercise bootstrap forwarding and prefetch policy without installing packages."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="deployment bootstrap targets Linux")


@pytest.mark.parametrize("prefetch,fail", [(True, False), (False, False), (True, True)])
def test_bootstrap_prefetch_is_gpu_hidden_optional_and_failure_falls_back(tmp_path, prefetch, fail):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required for the deployment bootstrap")
    venv = tmp_path / "private-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").touch()
    log = tmp_path / "commands.jsonl"
    body = (
        "#!" + sys.executable + "\nimport json,os,sys\n"
        'with open(os.environ["BOOTSTRAP_TEST_LOG"],"a") as stream:\n'
        ' stream.write(json.dumps({"args":sys.argv[1:],"cuda":os.environ.get("CUDA_VISIBLE_DEVICES")})+"\\n")\n'
        'sys.exit(1 if "flashinfer" in sys.argv and os.environ.get("BOOTSTRAP_TEST_FAIL")=="1" else 0)\n'
    )
    for name in ("python", "qwen-infra"):
        path = venv / "bin" / name
        path.write_text(body)
        path.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/bootstrap.sh"
    command = [bash, str(script), "--venv", str(venv)]
    if not prefetch:
        command.append("--no-prefetch")
    command.extend(["--", "serve", "--model", "example/model"])
    env = dict(
        os.environ,
        BOOTSTRAP_TEST_LOG=str(log),
        BOOTSTRAP_TEST_FAIL="1" if fail else "0",
        CUDA_VISIBLE_DEVICES="user-selection",
    )
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    native = [row for row in calls if row["args"][:2] == ["-m", "flashinfer"]]
    assert len(native) == (2 if prefetch else 0)
    if prefetch:
        assert native[0]["args"] == ["-m", "flashinfer", "install-cubin-wheel"]
        assert native[1]["args"] == [
            "-m",
            "flashinfer",
            "install-jit-cache-wheel",
            "--mode",
            "minimal",
            "--sm",
            "sm120f",
        ]
        assert all(row["cuda"] == "" for row in native)
    assert calls[-1] == {"args": ["serve", "--model", "example/model"], "cuda": "user-selection"}
    assert ("prefetch failed" in result.stderr) == fail
