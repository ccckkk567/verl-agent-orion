#!/bin/bash
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run this script inside a Slurm compute job.' >&2; return 1; }
export ORION_ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
export ORION_REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the experiment checkout}}"
export PILOT_VENV="$ORION_ROOT/envs/dep-prep-py312-v1"
export PILOT_OUTPUT_ROOT="$ORION_ROOT/logs/dependency_pilot"
export MODEL_PATH="${MODEL_PATH:-$ORION_ROOT/data/models/Qwen2.5-1.5B-Instruct}"
export HF_HOME="$ORION_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ORION_ROOT/.cache"
export PIP_CACHE_DIR="$ORION_ROOT/.cache/pip"
export TORCH_HOME="$ORION_ROOT/.cache/torch"
export TMPDIR="$ORION_ROOT/tmp/dep-prep-$SLURM_JOB_ID"
export JAVA_HOME="$ORION_ROOT/runtimes/jdk-11.0.28+6"
export PATH="$JAVA_HOME/bin:$PATH"
export ALFWORLD_DATA="$ORION_ROOT/data/alfworld"
export ALFWORLD_PYTHON="$ORION_ROOT/envs/alfworld-py310-v1/bin/python"
export APPWORLD_ROOT="$ORION_ROOT/data/appworld-dep-prep"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PIP_REQUIRE_VIRTUALENV=true PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR" "$PILOT_OUTPUT_ROOT" "$HF_HOME" "$PIP_CACHE_DIR"
if [[ -f "$ORION_ROOT/dep-prep-credentials.env" ]]; then
    set -a
    source "$ORION_ROOT/dep-prep-credentials.env"
    set +a
fi
cd "$ORION_REPO"
