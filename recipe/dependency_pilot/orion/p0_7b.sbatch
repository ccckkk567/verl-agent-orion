#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-7b-p0
#SBATCH --partition=testqueue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-7b-p0-%j.out
set -euo pipefail
trap 'echo "P0_7B_FAILED job=$SLURM_JOB_ID line=$LINENO"' ERR
module purge
module load python/3.12.5
export MODEL_PATH=/scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70
source recipe/dependency_pilot/orion/common.sh
[[ -f "$PILOT_VENV/.pilot-setup-complete" ]]
source "$PILOT_VENV/bin/activate"
python -I -c 'import sys,torch; print("EXISTING_PYTHON",sys.executable,sys.version,flush=True); assert torch.cuda.is_available(), "GPU required for 7B P0"'
nvidia-smi
PILOT_ENV="${1:?Specify alfworld or scienceworld}"
case "$PILOT_ENV" in
  alfworld) PILOT_OLD="$PILOT_OUTPUT_ROOT/alfworld-collect-20261012T095303Z-d56e91";;
  scienceworld) PILOT_OLD="$PILOT_OUTPUT_ROOT/scienceworld-rep0-159710";;
  *) echo 'Unknown environment' >&2; exit 2;;
esac
PILOT_RESULT="$PILOT_OUTPUT_ROOT/$PILOT_ENV-p0-7b-$SLURM_JOB_ID"
PILOT_COMPARE="$PILOT_OUTPUT_ROOT/$PILOT_ENV-1p5b-vs-7b-$SLURM_JOB_ID"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
PILOT_EXIT=0
python -u -m recipe.dependency_pilot.post_p0 root-p0-7b --source-run "$PILOT_OLD" --run-dir "$PILOT_RESULT" --model-path "$MODEL_PATH" --swanlab-mode "$SWANLAB_MODE" || PILOT_EXIT=$?
if [[ -f "$PILOT_RESULT/episodes.jsonl" ]]; then
  python -m recipe.dependency_pilot.post_audit compare --old "$PILOT_OLD" --new "$PILOT_RESULT" --output "$PILOT_COMPARE" || PILOT_EXIT=$?
  if [[ -f "$PILOT_COMPARE/plot_data.json" ]]; then
    python -m recipe.dependency_pilot.post_audit plot-compare --output "$PILOT_COMPARE" || PILOT_EXIT=$?
  fi
fi
# Preserve partial evidence too; unique job-based paths never replace 1.5B results.
PILOT_PACK="$PILOT_OUTPUT_ROOT/$PILOT_ENV-7b-results-$SLURM_JOB_ID.tar"
PILOT_ITEMS=("$(basename "$PILOT_RESULT")")
[[ ! -d "$PILOT_COMPARE" ]] || PILOT_ITEMS+=("$(basename "$PILOT_COMPARE")")
if [[ -d "$PILOT_RESULT" ]]; then
  tar -cf "$PILOT_PACK" -C "$PILOT_OUTPUT_ROOT" "${PILOT_ITEMS[@]}"
  sha256sum "$PILOT_PACK" > "$PILOT_PACK.sha256"
fi
echo "P0_7B_END job=$SLURM_JOB_ID env=$PILOT_ENV exit=$PILOT_EXIT output=$PILOT_RESULT archive=$PILOT_PACK"
exit "$PILOT_EXIT"
