#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --job-name=orion-install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=orion-install-%j.out

set -euo pipefail
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ -f "$REPO/setup.py" && -f "$REPO/scripts/orion/constraints.txt" ]]
module load python/3.12.5
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
VENV="$ROOT/envs/qwen3vl-py312-v1"
mkdir -p "$ROOT/envs" "$ROOT/logs"
# Avoid two installation jobs modifying the same environment simultaneously.
exec 9>"$ROOT/envs/qwen3vl-py312-v1.install.lock"
flock -n 9 || { echo "Another installation is running." >&2; exit 1; }
trap 'echo "Installation failed or interrupted; inspect this log before retrying." >&2' ERR
rm -f "$VENV/.install-complete"
/apps/python/3.12.5/bin/python -m venv "$VENV"
source "$REPO/scripts/orion/activate.sh"
python -c 'import sys; assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix; print(sys.executable)'

python -m pip install --index-url https://pypi.org/simple \
    'pip==25.3' 'setuptools==79.0.1' wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install --index-url https://pypi.org/simple \
    --no-build-isolation --only-binary=vllm,xformers,torch,torchvision,torchaudio \
    -c "$REPO/scripts/orion/constraints.txt" vllm==0.11.0 -e "$REPO"
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
            echo "cpu_node_kvm_read_write=true"
        else
            echo "cpu_node_kvm_read_write=false"
        fi
    else
        echo "cpu_node_kvm_exists=false"
    fi
} | tee "$ROOT/logs/orion-install-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$VENV/.install-complete"
echo "INSTALL_OK: submit scripts/orion/smoke.sbatch next."
# FlashAttention training extensions and the AndroidWorld runtime are separate
# setup steps. This job installs the model/inference base without model weights.
