# 本轮执行入口

分支 `exp/dep-prep-p1-sci-rep0-20261012`，从包含真实 P0 分析的 `a3614fab` 创建。继续用 SwanLab。旧运行只读，新结果另存。没有训练更新、算法/奖励/严格成功变更，也没有扩展 AppWorld。

## HPC 上执行（助手无法直连）

从旧 checkout 取实验分支，建独立 worktree。显式远端 refspec 避免旧 clone 的窄 fetch 配置丢失 remote-tracking ref：

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/code/verl-agent-orion && git fetch origin refs/heads/exp/dep-prep-p1-sci-rep0-20261012:refs/remotes/origin/exp/dep-prep-p1-sci-rep0-20261012 && git worktree add -b exp/dep-prep-p1-sci-rep0-20261012 ../verl-agent-dep-prep-postp0 origin/exp/dep-prep-p1-sci-rep0-20261012 && cd ../verl-agent-dep-prep-postp0
sbatch recipe/dependency_pilot/orion/post_p0.sbatch alf-p1
sbatch recipe/dependency_pilot/orion/post_p0.sbatch science-rep0
```

只在用户个人 cjk 目录工作，沿用 account prjsnnn0000269、testqueue 单 GPU、Python3.12.5 和已完成安装的 dep-prep-py312-v2；ALFWorld 沿用现有 py310 RPC 环境。两项可独立提交，脚本不安装依赖。不要在 login 节点直接运行 Python 采样、分析或安装。

日志：`/scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-postp0-JOBID.out`。

```bash
tail -n 80 -f /scratch/prjsnnn0000269/Students/shuming/cjk/logs/cjk-dep-postp0-JOBID.out
```

将 JOBID 换成提交返回编号。`COMPLETED_P1_GROUP` 表示一个 prefix×arm 的两条后缀已有本地结果。`POST_P0_COMPLETE` 表示该作业预定流程跑完；`Submitted batch job` 只代表已提交。

## ALFWorld

旧源目录：`$PILOT_OUTPUT_ROOT/alfworld-collect-20261012T095303Z-d56e91`。四个前缀 39/35/47/34 剩余轮数，先 p3，然后 p1/p2/p4；各组三个条件按固定种子打乱，两个新 continuation seeds 10000/10001。实际每步随机种子 = continuation_seed + 100003 × absolute_step_index。native reset seed 保持原值 0。

精确重放 source parsed_action，逐步核对 typed facts、反馈、菜单、moves、score 和任务，最终核对 history hash、持布状态；全可观测记录一致才进入 matched 后缀。未保存隐藏 RNG 快照，报告明确匹配范围。重放记录与新 rollout 分离，原历史仍为 actor 实际历史；hint 只在首步 prompt 尾部追加，之后不进入 history。tokenized input 核验 hint 可见。

输出 `alfworld-p1-JOBID/`：prefix_manifest、replay_checks、continuations、steps、episodes、diagnostics、event_annotations、metrics、analysis、图和报告。清洁谓词为原生 isclean，cleanable 不算清洁；同时记录 cloth 4 和任意合法 cloth。GiGPO 按 prefix×arm×batch 隔离，每组仅两个新后缀，重放不计入。

## ScienceWorld

旧源目录：`$PILOT_OUTPUT_ROOT/scienceworld-collect-20261012T094001Z-8ecf63`。

同一个 science-rep0 作业依次执行：实际旧 prompt/解析审查 → 两个任务各一条 12-turn smoke → token可见性/无输入截断/观察者不增加 moves 检查 → 写 immutable lock → 原始 reset 开始的 8 条完整 re-P0 → 离线 GiGPO → 同版规则重算旧数据 → 新旧指标与图。

完整协议强制 2 个原任务各 4 条、50 轮、4096/512 token、history4、temperature0.7、top_p1、seed0+i+100003×turn、空 simplifications、同模型 revision。profile 为 reference_interface_v1，parser 原样保留；详情见 SCIENCEWORLD_INTERFACE_DIFF.md。完整运行使用新的 scienceworld-rep0 run_id，不把 smoke、手工命令或旧重分析算作 re-P0。

输出同一 logs/dependency_pilot 下：scienceworld-interface-audit-JOBID、scienceworld-interface-smoke-JOBID、scienceworld-interface-lock-JOBID.json、scienceworld-rep0-JOBID、scienceworld-old-new-JOBID。原始 reset/steps/episodes、原生分数尺度、电气边和观察者错误都保存；unknown 不填成 0。原生 score 满分且 native completed 才是严格成功。

新旧比较检查模型/采样/奖励/JAR/wrapper/package/Torch/Transformers 差异；完整性另检查 run_kind 和样本数。图含按任务执行分类、连接/完成率、通用反馈与正常 anchor 支持。

## 本地分析与追踪

`python -m recipe.dependency_pilot.post_p0 --help` 和 `python -m recipe.dependency_pilot.post_audit --help` 是实际入口。直接 runner 调用示例（必须在计算节点）：

```bash
python -m recipe.dependency_pilot.post_p0 alf-p1 --source-run "$ALF_P0_DIR"
python -m recipe.dependency_pilot.post_p0 science-smoke --source-run "$SCI_P0_DIR" --run-dir "$SMOKE_DIR"
python -m recipe.dependency_pilot.post_audit lock-interface --smoke "$SMOKE_DIR" --output "$LOCK_FILE"
python -m recipe.dependency_pilot.post_p0 science-rep0 --source-run "$SCI_P0_DIR" --run-dir "$NEW_DIR" --lock-file "$LOCK_FILE"
python -m recipe.dependency_pilot.post_audit compare --old "$SCI_P0_DIR" --new "$NEW_DIR" --output "$COMPARE_DIR"
python -m recipe.dependency_pilot.post_audit plot-compare --output "$COMPARE_DIR"
```

每个采样组完成后写本地派生分析，完整 raw 每步即时保存。图在独立进程生成，避开本地 Windows Torch/matplotlib OpenMP 冲突。云追踪独立进程，失败不丢本地结果；SwanLab 上传带 prefix/arm/task 的汇总和图，不上传完整 prompt/facts/密钥。`tracking.json` 保存实际 backend、URL/失败原因。本地短测为 disabled，HPC 默认 cloud，均记录 training_updates=0。手工命令属于独立 interface fixture，不计模型成功。
