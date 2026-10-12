#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-dep-postp0
#SBATCH --partition=testqueue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-postp0-%j.out
set -euo pipefail
trap 'echo "POST_P0_FAILED job=$SLURM_JOB_ID line=$LINENO"' ERR
module purge
module load python/3.12.5
source recipe/dependency_pilot/orion/common.sh
[[ -f "$PILOT_VENV/.pilot-setup-complete" ]]
source "$PILOT_VENV/bin/activate"
python -I -c 'import sys; print("VENV_PYTHON",sys.executable,sys.version,flush=True); assert sys.version_info[:2] == (3,12)'
PILOT_PHASE="${1:?Specify alf-p1 or science-rep0}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
nvidia-smi
case "$PILOT_PHASE" in
  alf-p1)
    PILOT_OLD="$PILOT_OUTPUT_ROOT/alfworld-collect-20261012T095303Z-d56e91"
    PILOT_RESULT="$PILOT_OUTPUT_ROOT/alfworld-p1-$SLURM_JOB_ID"
    python -u -m recipe.dependency_pilot.post_p0 alf-p1 --source-run "$PILOT_OLD" --run-dir "$PILOT_RESULT" --swanlab-mode "$SWANLAB_MODE"
    ;;
  science-rep0)
    PILOT_OLD="$PILOT_OUTPUT_ROOT/scienceworld-collect-20261012T094001Z-8ecf63"
    PILOT_SMOKE="$PILOT_OUTPUT_ROOT/scienceworld-interface-smoke-$SLURM_JOB_ID"
    PILOT_LOCK="$PILOT_OUTPUT_ROOT/scienceworld-interface-lock-$SLURM_JOB_ID.json"
    PILOT_RESULT="$PILOT_OUTPUT_ROOT/scienceworld-rep0-$SLURM_JOB_ID"
    PILOT_COMPARE="$PILOT_OUTPUT_ROOT/scienceworld-old-new-$SLURM_JOB_ID"
    python -m recipe.dependency_pilot.post_audit audit-interface --old "$PILOT_OLD" --output "$PILOT_OUTPUT_ROOT/scienceworld-interface-audit-$SLURM_JOB_ID"
    python -u -m recipe.dependency_pilot.post_p0 science-smoke --source-run "$PILOT_OLD" --run-dir "$PILOT_SMOKE" --swanlab-mode "$SWANLAB_MODE"
    python -m recipe.dependency_pilot.post_audit lock-interface --smoke "$PILOT_SMOKE" --output "$PILOT_LOCK"
    python -u -m recipe.dependency_pilot.post_p0 science-rep0 --source-run "$PILOT_OLD" --run-dir "$PILOT_RESULT" --lock-file "$PILOT_LOCK" --swanlab-mode "$SWANLAB_MODE"
    python -m recipe.dependency_pilot.post_audit compare --old "$PILOT_OLD" --new "$PILOT_RESULT" --output "$PILOT_COMPARE"
    python -m recipe.dependency_pilot.post_audit plot-compare --output "$PILOT_COMPARE"
    ;;
  *) echo 'Unknown phase. Use alf-p1 or science-rep0.' >&2; exit 2;;
esac
echo "POST_P0_COMPLETE job=$SLURM_JOB_ID phase=$PILOT_PHASE output=$PILOT_RESULT"
