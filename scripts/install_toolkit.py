"""Install a checkout-private CUDA compiler when --with-toolkit is requested."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    cuda = target / "nvidia/cu13"
    if not (cuda / "bin/nvcc").is_file():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--target",
                str(target),
                "nvidia-cuda-nvcc==13.0.88",
                "nvidia-nvvm==13.0.88",
                "nvidia-cuda-crt==13.0.88",
                "nvidia-cuda-runtime==13.0.96",
            ],
            check=True,
            timeout=1200,
        )
    for alias, value in ((cuda / "lib64", "lib"), (cuda / "lib/libcudart.so", "libcudart.so.13")):
        if not alias.exists() and not alias.is_symlink():
            alias.symlink_to(value)
    from qwen_infra.runtime_paths import package_directory

    package = package_directory("nvidia.curand") or package_directory("nvidia.cu13")
    if package is not None:
        for header in (package / "include").glob("curand*.h"):
            alias = cuda / "include" / header.name
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to(header)
    # Output is a path, not an evaluated shell fragment.
    print(cuda)
    if not (cuda / "bin/nvcc").is_file():
        raise RuntimeError("Private CUDA installation did not provide nvcc")
    os.environ["CUDA_HOME"] = str(cuda)
    subprocess.run([str(cuda / "bin/nvcc"), "--version"], check=True, timeout=15)


if __name__ == "__main__":
    main()
