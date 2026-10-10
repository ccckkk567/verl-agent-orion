# Orion: ALFWorld and full WebShop

This setup creates two independent Python 3.10 venvs with the editable local
`verl-agent` package, Torch 2.8/CUDA 12.8 and vLLM 0.11. NumPy is held at 1.26.4
for the legacy games. It does not change the Qwen3-VL Python 3.12 venv.

All persistent paths are under `/scratch/prjsnnn0000269/Students/shuming/cjk`:

| Content | Relative path |
|---|---|
| ALFWorld venv | `envs/alfworld-py310-v1` |
| WebShop venv | `envs/webshop-py310-v1` |
| Managed Python / Java 11 | `runtimes/` |
| ALFWorld text games and logic | `data/alfworld/` |
| Full WebShop JSON and Lucene index | `data/webshop/` |
| Package snapshots and check reports | `logs/` |

Python 3.10.19 is installed by the cluster's `uv/0.11.14` module into Scratch,
then used to create standard venvs. WebShop's pinned Temurin 11 archive is
verified against its upstream SHA-256. No system installation is needed.

## Submit

Pull the updated repository in a short I/O job, following the existing workflow.
Wait for that job to finish successfully before submitting installation jobs.

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/code/verl-agent-orion
sbatch -A prjsnnn0000269 -p ioqueue --job-name=sync-code \
  --cpus-per-task=1 --mem=1G --time=00:10:00 \
  --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/sync-code-%j.out \
  --wrap='git pull --ff-only origin master'
```

After the pull succeeds:

```bash
sbatch --job-name=alf-install --mem=32G scripts/orion/games-install.sbatch alfworld
sbatch --job-name=web-install scripts/orion/games-install.sbatch webshop
```

Both use account `prjsnnn0000269`, `testqueue`, one GPU and eight CPUs. This
also lets the final check exercise the actual H200 stack. WebShop requests
96 GiB for loading the full catalog and building the index. Each job has an
eight-hour limit. They can be submitted independently; a per-venv lock prevents
overlapping installs of the same environment. Do not run experiments in a venv
while its installation job is running.

Output files are in the repository directory: `game-install-<jobid>.out`.
An empty `squeue` only means a job is no longer queued/running; inspect its log
and accounting status:

```bash
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
tail -n 100 game-install-JOBID.out
```

Success requires both `GAME_SMOKE_OK: <game>` and `GAME_INSTALL_OK: <game>`.
The environment's `.install-complete` marker is written only after checks pass.
For an interrupted download or installation, resubmit the same command. Complete
downloads and a matching validated full index are reused. Existing unrelated
repository data directories are not overwritten: inspect any path conflict
reported by the job. Failed index staging directories remain available for
diagnosis; do not launch repeated jobs blindly if storage is tight.

### Resume WebShop after a data download failure

If package installation, `pip check` and Java succeeded (as in job 158322),
update the checkout and submit only the remaining data/index/check work:

```bash
sbatch scripts/orion/webshop-prepare.sbatch
```

Read `web-prepare-<jobid>.out`. This job uses the installed WebShop venv and Java
without installing or upgrading packages. It holds the same lock as installation
and writes the completion marker only after the interaction check passes.

The original Google Drive link failed to yield a public download URL on Orion.
Downloads now use a pinned public [community mirror](https://huggingface.co/datasets/sparklabutah/timewarp-env-data),
with expected byte sizes and SHA-256 checks before publishing each file. The two
large-file hashes agree with two independent HF mirrors; the human-instruction
file matches the upstream WebShop GitHub copy. No login or token is used.
The files total about 5.67 GB, before index/intermediate files.
Interrupted mirror downloads resume from `.hf.part` files; old Google Drive
`.part` files are not reused. Existing completed files are verified too. A size
or hash mismatch stops the job rather than indexing unverified data.

## Validation and later use

- ALFWorld: download train, seen-validation and unseen-validation games plus
  PDDL/grammar; exercise reset and one legal action in each split through the
  repository's Ray wrapper. This is `AlfredTWEnv` text mode.
- WebShop: download the **full** product/attribute files and human instructions;
  build a matching full Lucene index, check its document count, then exercise
  reset, search, product click and purchase through the repository's Ray wrapper
  for train/eval goals. The check follows the repository default of synthetic
  goals (`human_goals=false`). A nonzero task success rate is not required: this
  scripted interaction tests the environment, not an agent's capability.
- Both: `pip check`, vLLM import and a GPU BF16 operation. This does **not** yet
  validate model generation, FlashAttention training extensions or an RL update.

To repeat checks without reinstalling:

```bash
sbatch --job-name=alf-check --mem=32G scripts/orion/games-check.sbatch alfworld
sbatch --job-name=web-check scripts/orion/games-check.sbatch webshop
```

Inside a future Slurm experiment script, from the repository root:

```bash
source scripts/orion/games-activate.sh alfworld
# Or: source scripts/orion/games-activate.sh webshop
```

For WebShop training **set `env.webshop.use_small=False`**. The repository
defaults to small data, which this full-data setup does not download. Full data
and the index are linked into the repository's legacy expected paths. Avoid
copying these links when making another checkout; run setup for that checkout.
Start with a small number of environment workers: the current wrapper loads a
full catalog per worker, so total RAM must be measured before scaling up.

The text modes need neither Android/KVM nor a browser service. Original setup
scripts that create Conda environments are not used. Installation and all checks
run on compute nodes; the local Windows checks cannot certify Linux execution.

Upstream references: [ALFWorld assets](https://github.com/alfworld/alfworld/blob/master/scripts/alfworld-download),
[WebShop setup/data IDs](https://github.com/princeton-nlp/WebShop/blob/master/setup.sh),
[uv managed Python](https://docs.astral.sh/uv/guides/install-python/),
[Temurin 11 release](https://github.com/adoptium/temurin11-binaries/releases/tag/jdk-11.0.28%2B6).
