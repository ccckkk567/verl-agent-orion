#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-dep-appworld
#SBATCH --partition=ioqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-appworld-%j.out
set -euo pipefail
module purge
module load python/3.11.9
source recipe/dependency_pilot/orion/common.sh
APP_VENV="$ORION_ROOT/envs/dep-prep-appworld-py311-v1"
exec 9>"$APP_VENV.install.lock"
flock -n 9 || exit 1
[[ -x "$APP_VENV/bin/python" ]] || python -m venv "$APP_VENV"
source "$APP_VENV/bin/activate"
python -m pip install 'appworld==0.1.3.post1'
mkdir -p "$APPWORLD_ROOT"
cd "$APPWORLD_ROOT"
appworld install
appworld download data
cd "$ORION_REPO"
python -m pip freeze > "$PILOT_OUTPUT_ROOT/appworld-pip-freeze-$SLURM_JOB_ID.txt"
python -m recipe.dependency_pilot.run prepare --env appworld --profile smoke
python -m recipe.dependency_pilot.run interface-check --env appworld --profile smoke
