#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=game-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=game-check-%j.out

set -euo pipefail
ulimit -c 0
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?}}"
source "$REPO/scripts/orion/games-activate.sh" "${1:?Specify alfworld or webshop}"
cd "$ORION_REPO"
python -m pip check
timeout 45m python scripts/orion/games_smoke.py "$GAME"
