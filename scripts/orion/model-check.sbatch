#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=model-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=model-check-%j.out

set -euo pipefail
ulimit -c 0
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
module load python/3.12.5
source "$REPO/scripts/orion/activate.sh"
[[ -f "$ORION_VENV/.install-complete" ]]
python -m pip check

# Use the Python already verified on testqueue; it raised SIGILL on ioqueue.
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
MODEL_DIR="$ORION_ROOT/models/Qwen3-VL-8B-Instruct-${REVISION:0:12}"
RESULT_DIR="$ORION_ROOT/logs/model-check-${SLURM_JOB_ID}"
mkdir -p "$MODEL_DIR" "$RESULT_DIR"
exec 9>"$MODEL_DIR/.model-check.lock"
flock -n 9 || { echo "Another model check is using this snapshot." >&2; exit 1; }
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
echo "Downloading $MODEL_ID at $REVISION into $MODEL_DIR"
hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$MODEL_DIR" --max-workers 4
echo "MODEL_DOWNLOAD_OK"

# Loading the pinned local snapshot should not fetch more model files.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
git -C "$REPO" rev-parse HEAD > "$RESULT_DIR/repository-commit.txt"
python "$REPO/scripts/orion/model_check.py" \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" --revision "$REVISION" \
    --output-dir "$RESULT_DIR"
