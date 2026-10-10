#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=kvm-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=kvm-check-%j.out

set -euo pipefail
ulimit -c 0
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ -f "$REPO/scripts/orion/check_kvm.py" ]]
module load python/3.12.5
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs"
python "$REPO/scripts/orion/check_kvm.py" \
    --output "$ROOT/logs/kvm-check-${SLURM_JOB_ID}.json"
