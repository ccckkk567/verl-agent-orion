#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=env-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=env-check-%j.out

set -euo pipefail
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
module load python/3.12.5
source "$REPO/scripts/orion/activate.sh"
[[ -f "$ORION_VENV/.install-complete" ]] || {
    echo "Installation has not completed successfully." >&2
    exit 1
}
python -m pip check
python "$REPO/scripts/orion/smoke.py"
