#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --job-name=kvm-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=kvm-cpu-%j.out

# Start with ioqueue; if it cannot provide KVM, probe a cpup allocation with:
# sbatch --partition=cpup scripts/orion/cpu-kvm.sbatch
set -euo pipefail
ulimit -c 0
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ -f "$REPO/scripts/orion/check_kvm.py" ]]
unset PYTHONHOME PYTHONPATH
mkdir -p "$ROOT/logs"
echo "KVM CPU probe: host=$(hostname) partition=${SLURM_JOB_PARTITION:-unknown}"

# /apps/python/3.12.5 raised SIGILL on ioqueue in job 158256. The OS Python
# suffices for this stdlib-only probe; an existing managed Python is a fallback.
PYTHON_BIN=''
for candidate in /usr/bin/python3 "$ROOT/envs/alfworld-py310-v1/bin/python"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -I -B -c 'import fcntl, sys; assert sys.version_info >= (3, 8); print(sys.executable, sys.version)'; then
        PYTHON_BIN="$candidate"
        break
    fi
    echo "Python startup failed: $candidate; trying the next existing interpreter." >&2
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo 'No working Python interpreter; KVM has NOT been tested.' >&2
    exit 1
fi
"$PYTHON_BIN" -I -B "$REPO/scripts/orion/check_kvm.py" \
    --output "$ROOT/logs/kvm-cpu-${SLURM_JOB_ID}.json"
