# 7B root P0 — 2026-10-12

Status at local handoff: **7B sampling not yet run**. Local code/config checks and historical 1.5B reanalysis are complete. The user submits the compute jobs on Orion; Codex has no direct HPC access.

Continue in `exp/dep-prep-p1-sci-rep0-20261012`, the existing post-P0 checkout. No new branch/worktree, environment installation, P1, probe, training, or cost comparison. SwanLab remains the tracker; raw traces stay local. Training updates = 0, policy updated = false.

Model: `Qwen/Qwen2.5-7B-Instruct`, pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`, 7,615,616,512 parameters. New path: `/scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70`. Download verifies all shards and tokenizer files against Hub content hashes; each sampling job rechecks the manifest, hashes and actual loaded model. No 1.5B SHA is used as an active 7B setting. Historical configs/locks are archived separately.

Run on the HPC login node (submission only):

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/code/verl-agent-dep-prep-postp0 && git pull --ff-only origin exp/dep-prep-p1-sci-rep0-20261012 && bash recipe/dependency_pilot/orion/submit_7b.sh
```

The helper submits one download/verification job on GPU testqueue, using the already working Python environment, then two independent dependent sampling jobs. Each P0 requests one GPU, 64 GiB host memory, up to four hours. Download has up to two hours. If download fails, dependent jobs are cancelled, not run with old weights. The helper prints all three job IDs. Do not resubmit while these jobs are pending/running.

The two actual collection commands inside Slurm are:

```bash
python -u -m recipe.dependency_pilot.post_p0 root-p0-7b --source-run "$PILOT_OLD" --run-dir "$PILOT_RESULT" --model-path "$MODEL_PATH" --swanlab-mode "$SWANLAB_MODE"
python -m recipe.dependency_pilot.post_audit compare --old "$PILOT_OLD" --new "$PILOT_RESULT" --output "$PILOT_COMPARE"
python -m recipe.dependency_pilot.post_audit plot-compare --output "$PILOT_COMPARE"
```

These are **prepared remote commands**, not a claim they have already run. Each new run records its actual invocation/status in `RUN_COMMANDS.md` and `run_manifest.json`.

Fixed baselines and tasks:

| Environment | Historical root P0 | New sampling |
|---|---|---|
| ALFWorld | `alfworld-collect-20261012T095303Z-d56e91` | Exact CreditCard→Sofa and Clean Cloth→Toilet manifest assets, 4 each, original reset |
| ScienceWorld | `scienceworld-rep0-159710` / `scienceworld-rep0-20261012T132744Z-7d238f` | `power-component/7`, `power-component-renewable-vs-nonrenewable-energy/5`, 4 each, original reset |

Both: 50 turns, temperature .7, top_p 1, prompt 4096, response 512, history 4, original seeds. ScienceWorld: unchanged `reference_interface_v1`, `p0_original_unchanged`, empty simplifications, verified 1.2.3 package/wrapper/JAR. Actor prompts, parsing, terminal reward and GiGPO core are unchanged. Electrical state collection is enabled for the new phase, observer-only. GiGPO uses step-weighted episode mean, mean_norm, gamma .95, step weight 1, exact pre-anchors, no filtering.

Outputs below `/scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/`:

- `<env>-p0-7b-<job>`: resolved config, model identity, raw reset/steps/episodes, native facts, annotations, offline AE/AS, metrics, figures, tracker status and code snapshot.
- `<env>-1p5b-vs-7b-<job>`: separate old/new reanalysis, per-task CSV, actual cases, comparison figures, config differences and SUMMARY.
- `<env>-7b-results-<job>.tar` and `.sha256`: download these after completion. Partial evidence is also packed when possible.
- Job logs are one directory above, `cjk-7b-model-<job>.out` and `cjk-7b-p0-<job>.out`.

`P0_7B_END ... exit=0` plus `RESULT=completed_exploratory` indicates the sampling process succeeded; verify eight complete episodes per environment in the artifact. A failed process or partial group remains an infrastructure/partial outcome, not a task failure. Tracking errors preserve local results.

Local validation completed before handoff:

- 26 fixtures pass, including unchanged collector behavior with logging off, parser behavior, native fact serialization, 7B config inheritance, correct source selection, N/A Pick cleaning labels, native cleaning and strict `connect` word matching.
- All three Slurm/submission scripts pass `bash -n`; no HPC jobs were submitted locally.
- Both historical root datasets reanalyzed using the new same-rule metrics and unchanged upstream GiGPO; input hashes verified unchanged. Outputs live outside the historical directories.
- 1.5B Pick 2/4 complete, Clean 0/4 complete. Clean: pickup 4/4, any sink 1/4, holding cloth at sink 0/4, real cleaning 0/4, immediate unclean target placement after first pickup 3/4 eligible episodes.
- Interface-adjusted 1.5B ScienceWorld: both tasks 0/4 complete; strict connect attempts 5 (power) and 113 (energy), zero new connections. Partial scores and focus are recorded separately.

Metric interpretation: cleaning keys are null/N/A for Pick. Missing facts are unknown. The first-pickup statistic excludes no-pickup, no-next-decision and unknown-fact cases, recording each reason. Alternative action support uses parser-usable commands in complete groups; its normal-anchor denominator excludes generic pre-feedback. The original GiGPO grouping, rewards and advantage calculations are not changed by these analysis labels. Task-relevant electrical connection remains unknown without sufficient evidence.

Next two actions: run the two 7B root P0 jobs; then inspect their genuine preparation events and compare per task against these exact historical baselines. Any P1 or skill check is a subsequent decision, not automatically scheduled.
