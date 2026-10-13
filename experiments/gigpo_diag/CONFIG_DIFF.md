# alfworld: reference recipe and execution differences

Reference: `examples/gigpo_trainer/run_alfworld.sh`, upstream commit in SOURCE.json. This run follows the locked source recipe, not an exact reproduction of a historical paper table.

- Initialization: Qwen/Qwen2.5-1.5B-Instruct + tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Immutable named directory; full file hashes checked before loading. No old pilot/RL checkpoint, no SFT.
- Preserve train batch 16, group 8, val batch 128, native prompt/parser/history, native reward and invalid penalty 0.1, KL loss 0.01, learning rate 1e-6, gamma .95, step weight 1, native normalization, 150 epochs. Dynamic filtering disabled; exact anchors.
- Resource adaptation: 2 Slurm GPUs, TP from reference; actor and logprob microbatch 2. No hardcoded CUDA devices. Separate Python 3.12.5 venv; vLLM version from each source recipe, pinned dependencies in requirements-runtime.txt.
- Native checkpoint every 10 updates and at native final boundary; keep all native saved checkpoints, including optimizer/extra/tokenizer. Initial model is a pinned reference. A walltime interruption may lose work since the latest complete checkpoint; no claim of automatic final checkpoint on SIGKILL.
- Initial eval enabled, native eval every 5 updates. Native ALF/Sci holdout evaluated in aggregate; full holdout text excluded. Fixed train/dev study slices and prefix probes are not yet implemented and do not block native training.
- SwanLab uses the existing Tracking owner once; local process-sharded JSONL always saved. Failed network logging is nonfatal. Raw trace/model artifacts are not uploaded to GitHub or tracker.
- Diagnostics detached/read-only; explicit step identity before reordering. Native invalid penalty is captured before/after mutation; actual AE/AS/joint captured within each native core. Original sampled tokens remain the loss target. Local parameter samples and successful native weight-loading calls are logged separately from driver update count.
- ScienceWorld-only: parent samples original L0 pool with replacement, repeats explicit task/variation/reset seed, worker loads it; same task/observation/state-summary verified before training. Python sampler state has a checkpoint sidecar. Original easy preset and reference `10 * (done and score>0)` retained; no score-delta/milestone shaping. Strict uses Java completion and unrounded score>=1 only as a diagnostic. Two groups x two live resets precede training.
- AppWorld-only: native one-response code turn, train/dev split, max_interactions aligned to max_steps. `multi_episode_rollout=False`, success_coef=0 and prompt_type=vanilla; the latter two are not read by the AppWorld ordinary prompt/reward path (ALF/WebShop shaping is a separate branch). No claim that AppWorld's previous default success_coef caused shaping. 16*8 + 16*1 = 144 services derived from resolved config. Unique job/worker/reset experiments preserve native API traces. No global process cleanup. AppWorld library pinned to upstream Git commit 42b5bcf3cd334fee33f0c37c02070a9f5807add5 per Git-install recipe (PyPI 0.1.3.post1 requires pydantic1 and conflicts with vLLM).

Native schedules and environment RNG may not resume bitwise identically (ALFWorld/TextWorld and AppWorld worker state not checkpointed). ScienceWorld task sampler sidecar restores its newly introduced independent RNG. Unknown labels remain null. No actual training has run locally; only HPC results can establish updates or learning.

## Installation repair after first HPC attempt (2026-10-13)

Observed: ALFWorld job 160221 built/installed flash-attn, then pip rejected
`ray[default]` in the constraints file. AppWorld job 160225 failed resolution:
vLLM 0.8.5 requires OpenTelemetry SDK 1.26.x but Ray[default] 2.49.0 requires >=1.30.
ScienceWorld job 160223 received an unknown Content-Type for the datasets index;
the same datasets 3.6.0 was found in the other jobs. Module pip/setuptools also
leaked from /apps/spack into the activated venv in all logs.

Separate constraints-runtime.txt strips extras; requirements-runtime.txt retains
the requested extras. Clear module PYTHONPATH/PYTHONHOME in both setup and train,
invoke installation using the absolute venv Python with -I, and verify imported
pip + distribution metadata reside in that venv. Retry only recognized transient
transport/index errors, at most three attempts. Each project gets its own pip cache.
Existing venvs and installed flash-attn are reused. No directory/cache purge or
global pip installation. Training recipe, model, reward and algorithm are unchanged.

## ALFWorld job 160510: verified upstream wheel metadata defects

Installation completed, then pip 25.3 check rejected textworld 1.7.0 and decord
0.6.0. The exact official PyPI Linux x86_64 wheels were downloaded locally and
SHA256-verified against PyPI JSON. See WHEEL_TAG_EVIDENCE.json for URLs, hashes,
original headers and corrected headers. TextWorld puts a blank line before Tag,
so pip's email header parser finds no tags. decord's filename says py3-none but
its internal header says cp36-cp36m (upstream issue dmlc/decord#356); its Python
wrapper loads libdecord via ctypes. This error is a metadata check, not a failed
GPU kernel or an actual runtime test.

Pin these two already installed versions. In the existing dedicated venv only,
repair the exact known WHEEL bytes (TextWorld header separator; decord tag to
match its published filename), update the corresponding RECORD hash, and keep
original WHEEL/RECORD backups under the venv's gigpo-wheel-tag-backups directory.
Unknown versions/metadata or unsupported target tags fail closed. No package
code, native library or other dependency is changed. Repeated setup is idempotent.
No new venv, dependency purge, or pip downgrade. Full pip check still must pass.

The same GPU prepare job then executes a native decord array allocation/copy
and the repository's actual ALFWorld worker reset + one look action on the train
pool. This is an installation check only: no model, optimizer, policy rollout or
experimental result. Evidence is saved in the job TMPDIR as wheel-tag-repair.json
and alfworld-runtime-check.json. Any failure blocks the dependent training job.
Local checks exercise the published headers, RECORD update/backups, idempotence,
unknown-metadata rejection, platform rejection and venv boundary. Actual Orion
runtime compatibility remains unverified until this prepare job succeeds.

Retry only ALFWorld with experiments/gigpo_diag/submit.sh after pulling this
branch. Do not rerun bootstrap_all.sh or change ScienceWorld/AppWorld checkouts
while their latest jobs have unknown status. Job 160510 failed before training;
its paired training job ID was not supplied and must not be guessed.

## Shared data stack: ALFWorld prepare job 160563

ALFWorld's full pip check, native decord call, actual environment reset/look,
torch CUDA availability and pinned model preparation now passed. The final data
preparation import then failed: installed PyArrow 26.0.0 requires NumPy >=2.0,
while these reference runtimes pin NumPy 1.26.4. Dependency metadata resolution
and pip check alone did not detect this runtime restriction. Training has not
started in the supplied log.

Keep NumPy 1.26.4 and datasets 3.6.0, explicitly pin PyArrow 19.0.1 in requirements
and constraints. All three complete Linux/Python3.12 dependency graphs resolve.
Actual local Python3.12 checks passed datasets mapping, nested prompt/extra_info
Parquet write/read, pandas/pyarrow reads and NumPy conversion. Add the same small
check after pip check in the existing prepare job; outputs are dependency checks
under the job TMPDIR, not experimental rollouts. Do not change native prepare.py,
placeholder sizes, task distribution, reward or trainer. Existing environments,
model files and native libraries are reused. Other projects receive the same
version constraint to prevent this shared failure; their jobs are not restarted
or cancelled automatically. HPC data preparation/real training remain unverified.
