#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=orion-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=orion-preflight-%j.out

set -euo pipefail

ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
[[ -f "$REPO/setup.py" && -d "$REPO/verl" ]] || {
    echo "Submit from the verl-agent-orion repository root, or set ORION_REPO." >&2
    exit 1
}

# Only inspect the allocation. Environment installation is a separate job.
module -t avail python cuda gcc 2>&1 || true
module load python/3.12.5
if ! module load cuda/12.6.2; then
    echo "CUDA 12.6.2 module unavailable; recording the available toolchain instead."
fi
module list 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs"
python "$REPO/scripts/orion/probe.py" \
    --repo "$REPO" \
    --output "$ROOT/logs/orion-preflight-${SLURM_JOB_ID}.json"
