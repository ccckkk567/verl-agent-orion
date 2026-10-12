# Dependency and preparation pilot (P0)

Work only on `exp/dep-prep-pilot-20261012`. Baseline: `24305ac85b5c7456ed2c0f851b12d70bda45823d`. Code, masks, prompts and local outputs are independent of SwanLab availability. See `SOURCES.md`, `metrics_dictionary.md`, and `DECISIONS.md` for semantics and limitations.

## Supported now

- `inspect`, `prepare`, `collect`, `analyze`, `plot`, `report`, ScienceWorld `reference-check`, AppWorld `interface-check`.
- Existing `TrajectoryCollector`, fixed-instance groups, original GiGPO diagnostics, existing SwanLab Tracking. Stable step IDs survive collector flattening. The hook is dormant unless a pilot observer is attached.
- Raw resets, decisions saved **before** environment execution, completed steps and episodes; original and tokenized actual prompt; raw/parsed response; stop reason; raw score/reward; once-only terminal reward; metrics, PNG/SVG figures and report.
- ALFWorld reuses its installed official text environment and upstream parser/templates. An external Python bridge supports the existing HPC Python 3.10 environment. AppWorld enumerates train/dev only and runs each world in its own child process on Orion.

One bounded ScienceWorld prefix probe is implemented and was run after P0. It replays an actual observed prefix, then compares the original goal with one local goal hint using the same remaining budget and endpoint. The source run lacks full state/RNG snapshots, so this is explicitly approximate replay and excluded from strict causal paired analysis. Training and other P2 branches are not implemented. `reference-check`/`interface-check` are **not model samples**. See `LOCAL_EXECUTION.md` for actual outcomes.

## Local commands

Run from the experiment checkout, in a compatible Python environment:

```bash
python -m recipe.dependency_pilot.run inspect
python -m recipe.dependency_pilot.run reference-check --env scienceworld --max-tasks 2 --output-root /absolute/output
python -m recipe.dependency_pilot.run collect --env scienceworld --profile smoke --model-path /absolute/model --output-root /absolute/output --swanlab-mode offline
python -m recipe.dependency_pilot.run report --run-dir /absolute/output/existing-run
python -m pytest -q recipe/dependency_pilot/tests
```

Every preparation/collection invocation creates a new unique directory. `--run-dir` for these commands must name a nonexistent directory. Analysis/report commands accept an existing directory. Raw data is never overwritten; derived metrics/figures can be regenerated.

Smoke: 2 task instances × 4 trajectories, 50 turns, 512 response tokens, 4096 prompt tokens. Discovery: ScienceWorld 8×4, ALFWorld 12×4. AppWorld smoke is 2×2, 30 turns, 1024/8192 tokens. `--max-tasks`, `--max-steps`, `--response-tokens`, `--rollouts-per-task`, etc. are explicit recorded overrides. All results retain task/instance/split identity; no outcome-based task selection. Confirmation dev tasks are enumerated but not collected.

For a 4GB local GPU a **separate connectivity protocol** can use `--max-tasks 2 --rollouts-per-task 2 --max-steps 12 --response-tokens 128 --gpu-memory-gib 3`. It is not pooled with the 50-turn profile. Actual execution records are in the delivered output folder and `LOCAL_EXECUTION.md` after runs finish.

## Orion handoff

HPC repository: `/scratch/prjsnnn0000269/Students/shuming/cjk/code/verl-agent-orion`. The assistant cannot connect; the user executes all commands. Push is explicitly left to the user. On the local experiment checkout:

```bash
git push -u origin exp/dep-prep-pilot-20261012
```

On Orion, obtain an independent checkout without changing its existing master worktree:

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/code/verl-agent-orion
git fetch origin exp/dep-prep-pilot-20261012
git worktree add -b exp/dep-prep-pilot-20261012 ../verl-agent-dep-prep-pilot origin/exp/dep-prep-pilot-20261012
cd /scratch/prjsnnn0000269/Students/shuming/cjk/code/verl-agent-dep-prep-pilot
sbatch recipe/dependency_pilot/orion/prepare.sbatch
```

These branch/worktree commands are for first creation; inspect an existing experiment worktree rather than force-resetting it. After the setup job completes successfully:

```bash
sbatch recipe/dependency_pilot/orion/collect.sbatch scienceworld
sbatch recipe/dependency_pilot/orion/collect.sbatch alfworld
sbatch recipe/dependency_pilot/orion/appworld-prepare.sbatch
# After AppWorld's independent setup succeeds:
sbatch recipe/dependency_pilot/orion/collect.sbatch appworld
```

CPU setup, downloads and JVM reference paths run on `ioqueue`; frozen generation uses one GPU on `testqueue`. All jobs explicitly use `prjsnnn0000269`, names beginning `cjk-dep-`, bounded two-hour allocations, private temp paths and Scratch outputs. No account-wide cancellation, Ray shutdown or process kill.

Credentials may be provided via `HF_TOKEN` and `SWANLAB_API_KEY`, or upload the separately supplied credentials file to `/scratch/prjsnnn0000269/Students/shuming/cjk/dep-prep-credentials.env`. Jobs source it only into their own environment. The file is separate from Git/config/results. SDK 0.10.2 login defaults to `save=False`.

Outputs: `/scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/<run-id>/`. Batch stdout prints `RUN_DIR=` and `RESULT=`. Each run writes a report even if infrastructure fails. Copy the run directory back locally for review; cloud summary visibility is recorded in `tracking.json`, never inferred from offline files.

Optional `SWANLAB_MODE=offline` can be exported before submitting if the compute node has no internet. SwanLab receives scalars and PNGs only; task/episode CSVs and raw traces stay local in P0. For Qwen downloads, `download_model` pins the official revision and verifies weight hashes, with resumable files outside Git.
