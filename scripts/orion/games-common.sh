#!/bin/bash
# Source inside a compute job; GAME must be alfworld or webshop.
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'A Slurm allocation is required.' >&2; return 1; }
case "${GAME:-}" in alfworld|webshop) ;; *) echo 'Set GAME=alfworld or webshop.' >&2; return 1 ;; esac
export ORION_ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
export ORION_REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export GAME_VENV="$ORION_ROOT/envs/${GAME}-py310-v1"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$ORION_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ORION_ROOT/.cache"
export PIP_CACHE_DIR="$ORION_ROOT/.cache/pip"
export UV_CACHE_DIR="$ORION_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ORION_ROOT/runtimes/python"
export UV_PYTHON_BIN_DIR="$ORION_ROOT/runtimes/bin"
export UV_NO_MODIFY_PATH=1
export TORCH_HOME="$ORION_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$ORION_ROOT/.cache/triton"
export TMPDIR="$ORION_ROOT/tmp/$SLURM_JOB_ID"
export ALFWORLD_DATA="$ORION_ROOT/data/alfworld"
export WEBSHOP_DATA="$ORION_ROOT/data/webshop"
export JAVA_HOME="$ORION_ROOT/runtimes/jdk-11.0.28+6"
export PIP_REQUIRE_VIRTUALENV=true PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ORION_ROOT/envs" "$ORION_ROOT/logs" "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_BIN_DIR"
if [[ "$GAME" == webshop ]]; then
    export PATH="$JAVA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$JAVA_HOME/lib/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export JAVA_TOOL_OPTIONS='-Xmx16g -XX:ActiveProcessorCount=4'
fi
