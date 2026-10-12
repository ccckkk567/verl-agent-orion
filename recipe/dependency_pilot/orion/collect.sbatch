#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-dep-collect
#SBATCH --partition=testqueue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-collect-%j.out
set -euo pipefail
module purge
module load miniforge/24.11.3-2
source recipe/dependency_pilot/orion/common.sh
source "$PILOT_VENV/bin/activate"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
PILOT_ENV="${1:-scienceworld}"
case "$PILOT_ENV" in scienceworld|alfworld|appworld) ;; *) echo 'Unknown pilot environment' >&2; exit 2;; esac
if [[ "$PILOT_ENV" == appworld ]]; then
    export APPWORLD_PYTHON="$ORION_ROOT/envs/dep-prep-appworld-py311-v1/bin/python"
fi
nvidia-smi
python -m recipe.dependency_pilot.run collect --env "$PILOT_ENV" --profile smoke --swanlab-mode "${SWANLAB_MODE}"
