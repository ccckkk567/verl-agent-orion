#!/bin/bash
# Run from the existing post-P0 checkout on the login node; only submit jobs here.
set -euo pipefail
[[ "$(git branch --show-current)" == exp/dep-prep-p1-sci-rep0-20261012 ]] || { echo 'Use the existing 1.5B post-P0 experiment branch.' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked files changed: preserve and inspect them before submitting.' >&2; exit 2; }
PILOT_PREP=$(sbatch --parsable recipe/dependency_pilot/orion/prepare_7b.sbatch)
PILOT_PREP=${PILOT_PREP%%;*}
echo "MODEL_DOWNLOAD_JOB=$PILOT_PREP"
for PILOT_ENV in alfworld scienceworld; do
  PILOT_JOB=$(sbatch --parsable --dependency="afterok:$PILOT_PREP" --kill-on-invalid-dep=yes recipe/dependency_pilot/orion/p0_7b.sbatch "$PILOT_ENV")
  echo "$PILOT_ENV P0_JOB=${PILOT_JOB%%;*} dependency=$PILOT_PREP"
done
