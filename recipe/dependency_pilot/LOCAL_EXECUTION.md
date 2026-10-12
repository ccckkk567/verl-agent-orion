# 本地实跑记录：2026-10-12

实验分支：`exp/dep-prep-pilot-20261012`。基线 `24305ac85b5c7456ed2c0f851b12d70bda45823d`；P0 实跑代码 commit `190815d8`。后续前缀代码在该 commit 上开发、实跑并另行提交，原始 manifest 保留当时的 dirty 状态。没有 push、merge、训练更新或 HPC 执行。

## 已运行与尚未运行

| 项目 | 实际运行 | 结果与边界 |
|---|---|---|
| ScienceWorld 官方参考动作 | 2 个实例、21 个环境动作 | 两条均达到 100 分；用于接口核验，不是模型能力成绩。首次选择为同一任务族，之后 P0 改为预先声明的两任务族轮询。 |
| 冻结 Qwen2.5-1.5B-Instruct | 2 个 train 实例 × 2 条轨迹 × 12 轮，48 次交互 | 完整成功 0/4，全部轮数耗尽；两个任务实例的宏平均也是 0。 |
| 原目标 / 局部目标提示续接 | 1 个真实前缀 × 2 条件 × 2 次，32 次新模型交互 | 两条件接线子目标均 0/2，完整任务均 0/2；近似重放，不能作严格配对因果比较。 |
| ALFWorld | 本地执行 prepare，未采样 | 未配置本地 ALFWORLD_DATA；复用 HPC 已有 `alfworld-py310-v1` 的脚本已备妥，等待远端实际执行。 |
| AppWorld | 本地执行 prepare，未交互 | 本机未安装 AppWorld；独立 Python 3.11/ioqueue 安装及 train/dev 小型交互检查脚本已备妥，依赖任务语义尚未确认。 |
| GiGPO | 用原 core 对 P0 离线计算 | 48 步的 episode/local advantage 都为 0；这是冻结采样诊断，不是训练效果。 |

模型 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。ScienceWorld 1.2.3，无 simplifications。模型在本地 RTX 3050 Laptop 4GB 上以 FP16 加载到 CUDA；无量化、无参数更新。实际本地 torch 为 2.10.0+cu126；HPC 安装清单为 2.8.0，二者不能冒称完全相同运行环境。完整环境版本和模型 SHA 保存在 manifest 中。

本地协议显式缩减为每题 2 条、12 轮、128 响应 token；温度 0.7、top_p 1、prompt 4096、历史 4 轮。HPC 默认仍为每题 4 条、50 轮、512 响应 token。P0 没有输入/输出 token 截断，所有 episode 都触及轮数预算；前缀提示组有 1/16 次响应达到 128 token 上限。不能排除轮数预算、提示表达、环境操作能力或更大响应预算的影响。

## 三个主要观察

1. **格式正确不代表命令可执行。** P0 的 `<think>/<action>` 解析成功 47/48，但 41/48 返回原生 `No known action matches that input.`。可见最直接的问题发生在环境命令表达；尚不能排除基本能力不足来支持 H1。
2. **有局部行为，未形成电路。** `5e7f0a99-4ff4-4ac3-9e6e-6613360dde3f:3` 的 `focus on green light bulb` 被接受，score 从 5 升到 38。下一步 `connect green light bulb to battery` 被告知必须指定 anode/cathode 等接点。聚焦已证实，但这不是接线准备条件齐备的证据。提示续接有一次成功激活核能发电机，仍没有接线成功。
3. **这批数据没有非零回报比较证据。** 研究奖励只在首次严格完成时给 +10，因此全部为 0；GiGPO episode/local 项也为 0。8 个精确观察 anchor 组中 4 个只有一条独特轨迹。不能据此声称 GiGPO 错估了准备动作，更不能宣称算法优于或劣于其他算法。

原目标续接 16 步中 11 步返回“不认识命令”、2 步要求指定接点；局部提示续接分别为 2 步与 10 步。提示改变了动作类型，但没有达成预先选择的接线终点。样本只有一个前缀、每条件两次。

## 前缀的可比性

来源为上面的真实轨迹，包含从 reset 到第 4 次动作（零基 step 3）的全部真实输出。提示为 `Make the green light bulb electrically connected to at least one wire.`，只描述一个条件，未提供动作序列。两条件保留原目标、同样历史、同样剩余 8 轮及 128 token/轮，使用匹配的采样种子。

局部终点是原生 `GoalObjectConnectedToWire` 两个条件中任一个成为 true；开始时两个均 false。该接口只供观察者评价，不进入模型输入。各对重放后的对象树/目标进度/可见状态哈希相同，原始观察与分数也相同；原 P0 没有保存完整环境状态和 RNG，故仍标为 **approximate**，`strict_paired_analysis_eligible=false`。没有为了得到效果而更换前缀或提示。

## 输出与 SwanLab

完整本地交付目录：`C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\local-20261012`。

原始运行目录位于 `C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-results`，交付目录另存这些已完成结果的副本：

- `scienceworld-collect-20261012T074122Z-02b488`：resolved config、manifest、原始 decisions/steps/episodes、metrics、GiGPO diagnostics、3 组 PNG/SVG 图、自动报告。
- `prefix-probe-20261012T075403Z-d0a4e7`：四组续接原始记录、开始状态快照、重放比较、共同终点与报告。续接目前仅本地记录，未上传 SwanLab。
- `scienceworld-reference-check-20261012T072551Z-72b81a`：官方动作接口核验。
- `alfworld-prepare-20261012T075525Z-1f867e` / `appworld-prepare-20261012T075526Z-ed19ba`：实际失败原因及报告，无虚构任务或模型结果。
- `case_annotations.jsonl`：两条以原生反馈为证据的人工补充标签；原始记录保持原样，未确认条件保留 null。
- `action_feedback.png` / `.svg`：真实 80 次模型动作的反馈分类图。
- `raw_checksums.json`：交付副本原始日志校验值。

SwanLab **P0 指标和三张图已补传完成**：[实际运行](https://swanlab.cn/@KANSEI/gigpo-dependency-pilot/runs/g1y15ziq)。身份为原 P0 的独立 analysis run，配置保存 parent_run_id 和源 steps SHA256。Windows 首次日志编码、DLL 初始化和 OpenMP 冲突导致的失败记录仍保存；以该链接和 `tracking.json` 的 submitted/completed 为准。没有使用忽略 OpenMP 冲突的环境变量。

## 实际运行命令

在本地实验 worktree 的 PowerShell 中运行；这些是复现命令，重新执行会新建结果目录：

```powershell
Set-Location 'C:\Users\19096\.codex\worktrees\dep-prep-pilot\verl-agent'
$PilotPython='C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-local\Scripts\python.exe'
$PilotModel='C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-models\Qwen2.5-1.5B-Instruct'
$PilotOutput='C:\Users\19096\Desktop\verl-agent\tmp\dep-prep-results'
$env:JAVA_HOME='C:\Program Files\Java\jdk-11'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:TOKENIZERS_PARALLELISM='false'
$env:PYTHONDONTWRITEBYTECODE='1'
& $PilotPython -m recipe.dependency_pilot.run reference-check --env scienceworld --max-tasks 2 --output-root $PilotOutput
& $PilotPython -u -m recipe.dependency_pilot.run collect --env scienceworld --model-path $PilotModel --max-tasks 2 --rollouts-per-task 2 --max-steps 12 --response-tokens 128 --gpu-memory-gib 3 --output-root $PilotOutput --swanlab-mode cloud
& $PilotPython -u -m recipe.dependency_pilot.probes --run-dir "$PilotOutput\scienceworld-collect-20261012T074122Z-02b488" --traj-uid 5e7f0a99-4ff4-4ac3-9e6e-6613360dde3f --after-step 3 --hint 'Make the green light bulb electrically connected to at least one wire.' --repeats 2
& $PilotPython -m recipe.dependency_pilot.run prepare --env alfworld --output-root $PilotOutput
& $PilotPython -m recipe.dependency_pilot.run prepare --env appworld --output-root $PilotOutput
& $PilotPython -m recipe.dependency_pilot.run track --run-dir "$PilotOutput\scienceworld-collect-20261012T074122Z-02b488" --swanlab-mode cloud
```

`track` 使用已保存指标和图；凭据由进程环境或独立文件提供，不包含在上述代码中。官方参考动作第一次实跑时使用旧的同族选择次序；目前复现 reference-check 将选择两个不同任务族，原始结果未重写。

测试：7 passed；覆盖未知值分母、只发一次终局奖励、GiGPO 数值重构/组隔离、collector observer 开关结果相同和稳定 step ID、正分且 done 不算完整成功、关闭云日志、局部终点的 unknown/false 区分。HPC 脚本仅本地语法检查，实际调度与安装未验证。

## 下一轮只做两件事

1. 在 Orion 跑已备妥的 ScienceWorld 50 轮协议、ALFWorld 文字参照与 AppWorld 小型接口检查，记录实际资源与成功/失败；AppWorld 先看真实 train/dev 指令再确认依赖子集。
2. 选择**独立操作能力诊断**分支：少量固定电路状态检查端点命名/接线操作。在有明确局部目标后仍不会操作时，再做少量 Qwen2.5-3B 对照；保持数据和协议分开。暂不进入优势算法修改或大规模 P2，也不从这一批结果推断 H1/H2/H3 成立。

同步、凭据上传、创建远端 worktree 和 sbatch 命令见 `README.md`。用户实际执行 HPC 操作；禁止登录节点重计算、全局配置修改或共享账号范围的进程清理。
