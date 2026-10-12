#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-dep-prepare
#SBATCH --partition=ioqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-prepare-%j.out
set -euo pipefail
module purge
module load miniforge/24.11.3-2
source recipe/dependency_pilot/orion/common.sh
exec 9>"$ORION_ROOT/envs/dep-prep-py312-v1.install.lock"
flock -n 9 || { echo 'Another pilot setup job owns the install lock.' >&2; exit 1; }
if [[ ! -x "$PILOT_VENV/bin/python" ]]; then
    python -m venv "$PILOT_VENV"
fi
source "$PILOT_VENV/bin/activate"
python -m pip install -r recipe/dependency_pilot/requirements-p0.txt
python -m pip freeze > "$PILOT_OUTPUT_ROOT/pip-freeze-setup-$SLURM_JOB_ID.txt"
java -version
python -m recipe.dependency_pilot.run inspect > "$PILOT_OUTPUT_ROOT/inspect-setup-$SLURM_JOB_ID.json"
python -m recipe.dependency_pilot.download_model "$MODEL_PATH" --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python -m recipe.dependency_pilot.run reference-check --env scienceworld --max-tasks 2
if ! python -m recipe.dependency_pilot.run prepare --env alfworld --profile smoke; then
    echo 'ALFWorld preparation failed; the ScienceWorld path remains available. See the generated report.'
fi
python -m pytest -q recipe/dependency_pilot/tests
