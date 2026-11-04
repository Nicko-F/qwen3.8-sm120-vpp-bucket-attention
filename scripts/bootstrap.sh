#!/usr/bin/env bash
# Private checkout-local environment. Native vLLM's wheel remains intact.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${QWEN_INFRA_VENV:-$repo/.venv}"
python="${PYTHON:-python3}"
wheel="${QWEN_INFRA_VLLM_WHEEL:-}"
runtime=1
toolkit=0
while (($#)); do
  case "$1" in
    --help|-h)
      printf '%s\n' 'Usage: bash scripts/bootstrap.sh [--venv PATH] [--python PYTHON] [--vllm-wheel URL] [--with-toolkit] [--package-only] -- [doctor|serve|prefill ...]'
      exit 0 ;;
    --venv) venv="$2"; shift 2 ;;
    --python) python="$2"; shift 2 ;;
    --vllm-wheel) wheel="$2"; shift 2 ;;
    --package-only) runtime=0; shift ;;
    --with-toolkit) toolkit=1; shift ;;
    --) shift; break ;;
    *) printf 'Unknown bootstrap option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! -x "$venv/bin/python" ]]; then
  "$python" -m venv "$venv"
fi
if [[ ! -f "$venv/pyvenv.cfg" ]]; then
  printf 'Refusing to install into a non-venv prefix: %s\n' "$venv" >&2
  exit 2
fi
"$venv/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix, "A private virtual environment is required"'
"$venv/bin/python" -m pip install --upgrade pip
if ((runtime)); then
  if [[ -n "$wheel" ]]; then
    "$venv/bin/python" -m pip install "$wheel"
    "$venv/bin/python" -m pip install 'flashinfer-python>=0.7.1,<0.8' 'nvidia-cutlass-dsl[cu13]>=4.8,<5' ninja
  else
    "$venv/bin/python" -m pip install -r "$repo/requirements-tested.txt"
  fi
  "$venv/bin/python" -m pip install "$repo"
else
  "$venv/bin/python" -m pip install "$repo"
fi
if ((toolkit)); then
  "$venv/bin/python" "$repo/scripts/install_toolkit.py" --target "$venv/cuda-toolkit"
  export CUDA_HOME="$venv/cuda-toolkit/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-2}"
if (($# == 0)); then set -- doctor; fi
exec "$venv/bin/qwen-infra" "$@"
