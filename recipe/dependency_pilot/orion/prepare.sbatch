#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-dep-prepare
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-prepare-%j.out
set -euo pipefail
# Python 3.12.5 raised SIGILL on ioqueue; setup passed on testqueue in job 159634.
ulimit -c 0
module purge
module load python/3.12.5
source recipe/dependency_pilot/orion/common.sh
trap 'echo "PILOT_PREPARE_FAILED job=$SLURM_JOB_ID line=$LINENO" >&2' ERR
exec 9>"$PILOT_VENV.install.lock"
flock -n 9 || { echo 'Another pilot setup job owns the install lock.' >&2; exit 1; }
PILOT_BASE_PYTHON="$(command -v python3.12)"
"$PILOT_BASE_PYTHON" -I -c 'import sys, platform; print("BASE_PYTHON", sys.executable, sys.version, platform.machine(), flush=True); assert sys.version_info[:2] == (3, 12), "Expected Python 3.12 from the loaded module"'
if [[ -f "$ORION_ROOT/envs/dep-prep-py312-v1/pyvenv.cfg" ]]; then
    echo 'Previous v1 environment Python version (read-only diagnostic):'
    sed -n '/^version = /p' "$ORION_ROOT/envs/dep-prep-py312-v1/pyvenv.cfg"
fi
if [[ ! -x "$PILOT_VENV/bin/python" ]]; then
    "$PILOT_BASE_PYTHON" -I -m venv "$PILOT_VENV"
fi
source "$PILOT_VENV/bin/activate"
python -I -c 'import sys; print("VENV_PYTHON", sys.executable, sys.version, flush=True); assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix, "Refusing to install outside a venv"'
rm -f "$PILOT_VENV/.pilot-setup-complete"
# Ignore inherited package-index configuration only in this job, not shared home.
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS
python -m pip install --index-url https://pypi.org/simple 'pip==25.3'
python -m pip --version
python -m pip install --index-url https://download.pytorch.org/whl/cu126 --only-binary=:all: 'torch==2.8.0'
python -m pip install --index-url https://pypi.org/simple --only-binary=torch,numpy,pyarrow -r recipe/dependency_pilot/requirements-p0.txt
python -m pip check
python -c 'import torch; print("TORCH_BUILD", torch.__version__, "CUDA_WHEEL", torch.version.cuda, flush=True); assert torch.__version__.split("+")[0] == "2.8.0"; assert torch.version.cuda == "12.6"'
python -m pip freeze > "$PILOT_OUTPUT_ROOT/pip-freeze-setup-$SLURM_JOB_ID.txt"
java -version
python -m recipe.dependency_pilot.run inspect > "$PILOT_OUTPUT_ROOT/inspect-setup-$SLURM_JOB_ID.json"
python -m recipe.dependency_pilot.download_model "$MODEL_PATH" --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python -m recipe.dependency_pilot.run reference-check --env scienceworld --max-tasks 2
if ! python -m recipe.dependency_pilot.run prepare --env alfworld --profile smoke; then
    echo 'ALFWorld preparation failed; the ScienceWorld path remains available. See the generated report.'
fi
python -m pytest -q recipe/dependency_pilot/tests
printf '%s\n' "$SLURM_JOB_ID" > "$PILOT_VENV/.pilot-setup-complete"
echo "PILOT_PREPARE_OK job=$SLURM_JOB_ID venv=$PILOT_VENV"
