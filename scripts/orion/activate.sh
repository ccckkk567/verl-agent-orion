#!/bin/bash
# Source only inside a Slurm allocation, after loading the same Python module.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Activate this environment inside a Slurm compute job." >&2
    return 1
fi
export ORION_ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
export ORION_VENV="$ORION_ROOT/envs/qwen3vl-py312-v1"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ORION_ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$ORION_ROOT/.cache/pip"
export XDG_CACHE_HOME="$ORION_ROOT/.cache"
export TORCH_HOME="$ORION_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$ORION_ROOT/.cache/triton"
export TMPDIR="$ORION_ROOT/tmp/${SLURM_JOB_ID}"
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"
if [[ ! -f "$ORION_VENV/bin/activate" ]]; then
    echo "Environment missing: submit scripts/orion/install.sbatch first." >&2
    return 1
fi
source "$ORION_VENV/bin/activate"
