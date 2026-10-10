#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=web-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=web-prepare-%j.out

# Resume after package/Java installation; no package installation or upgrades.
set -euo pipefail
ulimit -c 0
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
source "$REPO/scripts/orion/games-activate.sh" webshop
cd "$ORION_REPO"
exec 9>"$ORION_ROOT/envs/webshop-py310-v1.install.lock"
flock -n 9 || { echo 'WebShop installation/preparation is already running.' >&2; exit 1; }
trap 'echo "FAILED: WebShop data/index/check; inspect this log before retrying." >&2' ERR
echo "host=$(hostname) commit=$(git rev-parse HEAD)"
[[ -x "$JAVA_HOME/bin/java" ]] || { echo 'Install Java via games-install.sbatch first.' >&2; exit 1; }
java -version
python -m pip check
rm -f "$GAME_VENV/.install-complete"
python scripts/orion/games_prepare.py webshop
timeout 45m python scripts/orion/games_smoke.py webshop
printf '%s\n' "$SLURM_JOB_ID" > "$GAME_VENV/.install-complete"
echo 'GAME_INSTALL_OK: webshop'
