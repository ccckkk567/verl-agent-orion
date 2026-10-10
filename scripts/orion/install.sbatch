#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=env-install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=env-install-%j.out

set -euo pipefail
# The provided Python raised SIGILL on ioqueue (job 158256), but ran on
# testqueue during preflight (job 158237). Install on the target GPU nodes.
ulimit -c 0
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ -f "$REPO/setup.py" && -f "$REPO/scripts/orion/constraints.txt" ]]
module load python/3.12.5
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
hash -r
PYTHON_BIN="$(type -P python)"
echo "Installation host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Module Python: $PYTHON_BIN"
module list 2>&1
lscpu || true
if ! "$PYTHON_BIN" -c 'import sys, venv; print(sys.version); print(sys.executable); assert sys.version_info[:2] == (3, 12)'; then
    echo "Module Python cannot run here; no package installation attempted." >&2
    exit 1
fi
VENV="$ROOT/envs/qwen3vl-py312-v1"
mkdir -p "$ROOT/envs" "$ROOT/logs"
# Avoid two installation jobs modifying the same environment simultaneously.
exec 9>"$ROOT/envs/qwen3vl-py312-v1.install.lock"
flock -n 9 || { echo "Another installation is running." >&2; exit 1; }
trap 'echo "Installation failed or interrupted; inspect this log before retrying." >&2' ERR
rm -f "$VENV/.install-complete"
"$PYTHON_BIN" -m venv "$VENV"
source "$REPO/scripts/orion/activate.sh"
python -c 'import sys; assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix; print(sys.executable)'

python -m pip install --index-url https://pypi.org/simple \
    'pip==25.3' 'setuptools==79.0.1' wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install --index-url https://pypi.org/simple \
    --no-build-isolation --only-binary=vllm,xformers,torch,torchvision,torchaudio \
    -c "$REPO/scripts/orion/constraints.txt" vllm==0.11.0 -e "$REPO"
# Older installs pulled Decord via the default project dependencies. Its 0.6.0
# Linux wheel has contradictory Python tags (dmlc/decord#356). The screenshot
# baseline needs no Decord; torchvision remains available for video decoding.
python - <<'PY'
from importlib.metadata import distribution, PackageNotFoundError
try:
    decord = distribution("decord")
except PackageNotFoundError:
    print("Optional Decord is not installed.")
else:
    print("Removing optional Decord; original WHEEL metadata:")
    print(decord.read_text("WHEEL"))
PY
python -m pip uninstall -y decord
python -m pip check

python -m pip freeze > "$ROOT/logs/orion-packages-${SLURM_JOB_ID}.txt"
python -m pip inspect > "$ROOT/logs/orion-packages-${SLURM_JOB_ID}.json"
{
    echo "repository_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "hostname=$(hostname)"
    echo "python=$VENV/bin/python"
    if [[ -e /dev/kvm ]]; then
        ls -l /dev/kvm
        if [[ -r /dev/kvm && -w /dev/kvm ]]; then
            echo "node_kvm_read_write=true"
        else
            echo "node_kvm_read_write=false"
        fi
    else
        echo "node_kvm_exists=false"
    fi
} | tee "$ROOT/logs/orion-install-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$VENV/.install-complete"
echo "INSTALL_OK: submit scripts/orion/smoke.sbatch next."
# FlashAttention training extensions and the AndroidWorld runtime are separate
# setup steps. This job installs the model/inference base without model weights.
