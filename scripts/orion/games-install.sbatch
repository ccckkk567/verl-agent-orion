#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=game-install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=game-install-%j.out

set -euo pipefail
ulimit -c 0
GAME="${1:?Usage: sbatch scripts/orion/games-install.sbatch alfworld|webshop}"
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?}}"
source "$REPO/scripts/orion/games-common.sh"
cd "$ORION_REPO"
trap 'echo "FAILED: $GAME setup; inspect this log before retrying." >&2' ERR
exec 9>"$ORION_ROOT/envs/${GAME}-py310-v1.install.lock"
flock -n 9 || { echo 'This environment is already being installed.' >&2; exit 1; }
echo "game=$GAME host=$(hostname) commit=$(git rev-parse HEAD)"

# Use the documented uv module to obtain Python 3.10 in Scratch, then venv.
# Shared runtime setup is locked when the two game jobs start concurrently.
module load uv/0.11.14
mkdir -p "$ORION_ROOT/runtimes"
exec 8>"$ORION_ROOT/runtimes/games-bootstrap.lock"
flock 8
uv python install 3.10.19
PYTHON_BIN="$(uv python find --managed-python 3.10.19)"
"$PYTHON_BIN" -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 10)'
flock -u 8
"$PYTHON_BIN" -m venv "$GAME_VENV"
rm -f "$GAME_VENV/.install-complete"
source "$GAME_VENV/bin/activate"
if [[ "$GAME" == alfworld ]]; then
    for tool in gcc g++ make; do
        command -v "$tool" >/dev/null || { echo "Missing build tool: $tool" >&2; exit 1; }
    done
fi
python -m pip install --index-url https://pypi.org/simple 'pip==25.3' 'setuptools==79.0.1' wheel 'numpy==1.26.4' Cython
if [[ "$GAME" == alfworld ]]; then
    python -m pip install --index-url https://pypi.org/simple 'cmake==3.31.6'
fi
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install --index-url https://pypi.org/simple --no-build-isolation \
    --only-binary=vllm,xformers,torch,torchvision,torchaudio,nmslib \
    -c scripts/orion/games-constraints.txt \
    -r "scripts/orion/${GAME}-requirements.txt" vllm==0.11.0 -e .
python -m pip check
python -m pip freeze > "$ORION_ROOT/logs/${GAME}-packages-${SLURM_JOB_ID}.txt"

if [[ "$GAME" == webshop ]]; then
    # No sudo or system Java changes. Verify the upstream archive's SHA-256.
    if [[ ! -x "$JAVA_HOME/bin/java" ]]; then
        JDK_FILE=OpenJDK11U-jdk_x64_linux_hotspot_11.0.28_6.tar.gz
        JDK_URL=https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.28%2B6
        JDK_STAGE="$(mktemp -d "$ORION_ROOT/runtimes/java-stage.XXXXXX")"
        curl --fail --location --retry 3 "$JDK_URL/$JDK_FILE" -o "$JDK_STAGE/$JDK_FILE"
        printf '%s  %s\n' '7dfd551795a8884b26cbb02e0301da95db40160bb194f48271dc2ef9367f50c2' "$JDK_FILE" > "$JDK_STAGE/$JDK_FILE.sha256.txt"
        (cd "$JDK_STAGE" && sha256sum -c "$JDK_FILE.sha256.txt")
        tar --no-same-owner -xzf "$JDK_STAGE/$JDK_FILE" -C "$JDK_STAGE"
        mv "$JDK_STAGE/jdk-11.0.28+6" "$JAVA_HOME"
        rm -f "$JDK_STAGE/$JDK_FILE" "$JDK_STAGE/$JDK_FILE.sha256.txt"
        rmdir "$JDK_STAGE"
    fi
    java -version
fi
python scripts/orion/games_prepare.py "$GAME"
timeout 45m python scripts/orion/games_smoke.py "$GAME"
printf '%s\n' "$SLURM_JOB_ID" > "$GAME_VENV/.install-complete"
echo "GAME_INSTALL_OK: $GAME"
