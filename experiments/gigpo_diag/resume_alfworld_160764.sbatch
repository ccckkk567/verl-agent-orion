#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=ggp-alf-resume
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=17:50:00
set -euo pipefail
: "${GIGPO_RESUME_HELPER:?must point to standalone select_alfworld_resume.py}"
: "${GIGPO_EXPECTED_COMMIT:?must pin existing training checkout}"
base=/scratch/prjsnnn0000269/Students/shuming/cjk
parent="$base/logs/gigpo-reproduction/alfworld/alfworld-qwen25-15b-gigpo-160764-20261013T123704Z"
test "$(git rev-parse HEAD)" = "$GIGPO_EXPECTED_COMMIT"
test "$GIGPO_EXPECTED_COMMIT" = f261c3079388f776b7835c5fed329d85bfd235d9
export GIGPO_RESUME_FROM
GIGPO_RESUME_FROM=$("$base/runtimes/bin/python3.10" "$GIGPO_RESUME_HELPER" "$parent")
step=${GIGPO_RESUME_FROM##*global_step_}
if (( step >= 150 )); then
  echo "Parent already saved the full 150-update schedule; no continuation needed."
  exit 0
fi
export GIGPO_RESUME_MODE=resume_path
export GIGPO_RUN_ID="alfworld-qwen25-15b-gigpo-resume160764-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
unset SWANLAB_RUN_ID SWANLAB_RESUME
export SWANLAB_MODE=cloud
echo "RESUME_PARENT=160764 CHECKPOINT=$GIGPO_RESUME_FROM NEXT_UPDATE=$((step+1)) TARGET_UPDATE=150 SWANLAB=new_run"
bash experiments/gigpo_diag/train.sbatch
