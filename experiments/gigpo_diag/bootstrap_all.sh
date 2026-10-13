#!/usr/bin/env bash
set -euo pipefail
orion_code=/scratch/prjsnnn0000269/Students/shuming/cjk/code
mkdir -p "$orion_code/gigpo-reproduction" /scratch/prjsnnn0000269/Students/shuming/cjk/logs
alf_code="$(cd "$(dirname "$0")/../.." && pwd)"
status=0
if (cd "$alf_code" && bash experiments/gigpo_diag/submit.sh); then
  echo 'ALFWorld queued from existing verl-agent-orion repository.'
else
  echo 'ALFWorld submission failed; continuing other projects.' >&2
  status=1
fi
for env in scienceworld appworld; do
  name=scienceworld-digitalhuman
  relative=RLVMR/code
  if [[ "$env" == appworld ]]; then name=appworld-in-context-explorer; relative=.; fi
  checkout="$orion_code/gigpo-reproduction/$name"
  branch="exp/gigpo-train-diagnostics-$env-20261013"
  url="https://github.com/ccckkk567/gigpo-$env-diagnostics.git"
  if (
    if [[ ! -e "$checkout" ]]; then
      git clone --single-branch --branch "$branch" "$url" "$checkout"
    else
      test "$(git -C "$checkout" remote get-url origin)" = "$url"
      test "$(git -C "$checkout" branch --show-current)" = "$branch"
      test -z "$(git -C "$checkout" status --porcelain)"
      git -C "$checkout" pull --ff-only origin "$branch"
    fi
    cd "$checkout/$relative"
    bash experiments/gigpo_diag/submit.sh
  ); then
    echo "$env queued independently."
  else
    echo "$env setup/submission failed; other projects remain independent." >&2
    status=1
  fi
done
exit "$status"
