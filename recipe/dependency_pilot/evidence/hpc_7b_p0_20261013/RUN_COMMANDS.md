# 实际运行命令

远端采样由用户提交，HPC 的 run_manifest/RUN_COMMANDS 记录如下；本地复核只读取结果。

## alfworld / 159779

```bash
python -m recipe.dependency_pilot.post_p0 root-p0-7b --source-run /scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/alfworld-collect-20261012T095303Z-d56e91 --run-dir /scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/alfworld-p0-7b-159779 --model-path /scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70 --swanlab-mode cloud
```

Observed status: completed_exploratory.
Slurm job: 159779
Policy updated: false; training updates: 0.

## scienceworld / 159780

```bash
python -m recipe.dependency_pilot.post_p0 root-p0-7b --source-run /scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/scienceworld-rep0-159710 --run-dir /scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/scienceworld-p0-7b-159780 --model-path /scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70 --swanlab-mode cloud
```

Observed status: completed_exploratory.
Slurm job: 159780
Policy updated: false; training updates: 0.


## 本地已完成的复核

```text
C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-local\Scripts\python.exe -m recipe.dependency_pilot.post_audit compare --old C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\hpc-20261012\alfworld-collect-20261012T095303Z-d56e91 --new C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\hpc-159779-159780\alfworld-p0-7b-159779 --output C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\review-159779-159780\alfworld
```

退出码 0。

```text
C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-local\Scripts\python.exe -m recipe.dependency_pilot.post_audit plot-compare --output C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\review-159779-159780\alfworld
```

退出码 0。

```text
C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-local\Scripts\python.exe -m recipe.dependency_pilot.post_audit compare --old C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\post-p0-20261012\hpc-159709-159710\scienceworld-rep0-159710 --new C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\hpc-159779-159780\scienceworld-p0-7b-159780 --output C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\review-159779-159780\scienceworld
```

退出码 0。

```text
C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-local\Scripts\python.exe -m recipe.dependency_pilot.post_audit plot-compare --output C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\review-159779-159780\scienceworld
```

退出码 0。

附加审计实际执行脚本为本地 tmp/audit_7b_results.py，全部检查通过；汇总作图实际执行 tmp/build_7b_review.py。对应源码保存在 helpers/。脚本读取四个指定原始目录、本地哈希相同的 tokenizer 和当前实验实现，不创建模型，不调用环境。

## 输出定位

本地原始解包目录：`C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\hpc-159779-159780`。本地复核目录：`C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012\review-159779-159780`。

HPC 根目录为 `/scratch/prjsnnn0000269/Students/shuming/cjk/logs/dependency_pilot/`，新子目录为 `alfworld-p0-7b-159779` 和 `scienceworld-p0-7b-159780`。各自含 resolved_config.json、model_identity.json、raw steps/episodes、离线诊断、SwanLab 状态和实现快照。
