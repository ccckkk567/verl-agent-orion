#!/bin/bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=cjk-7b-model
#SBATCH --partition=testqueue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-7b-model-%j.out
set -euo pipefail
trap 'echo "MODEL_7B_FAILED job=$SLURM_JOB_ID line=$LINENO"' ERR
module purge
module load python/3.12.5
export MODEL_PATH=/scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70
source recipe/dependency_pilot/orion/common.sh
[[ -f "$PILOT_VENV/.pilot-setup-complete" ]]
source "$PILOT_VENV/bin/activate"
python -I -c 'import sys; print("EXISTING_PYTHON",sys.executable,sys.version,flush=True); assert sys.version_info[:2]==(3,12)'
# Reuse the GET downloader that worked on Orion's proxy, now with sharded safetensors support.
# No environment installation, upgrade, smoke, or model sampling in this job.
python -u -m recipe.dependency_pilot.download_model "$MODEL_PATH" --repo Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28
echo "MODEL_7B_READY job=$SLURM_JOB_ID model=$MODEL_PATH"
