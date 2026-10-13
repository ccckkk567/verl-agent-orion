# 7B P0 结果与下一步｜2026-10-13

**两环境共 16 条、800 步真实 P0 已完成，完整成功为 0/16。当前最突出的瓶颈是输出格式与既定解析接口不匹配；这轮数据没有支持“直接换 7B 就更适合研究准备性行动”。样本仅四个固定实例、每题四条，不能据此推断模型总体能力排序。**

| 任务 | 1.5B 完整成功 | 7B 完整成功 | 1.5B 可解析动作/步 | 7B 可解析动作/步 |
|---|---:|---:|---:|---:|
| ALFWorld Pick CreditCard→Sofa | 2/4 | 0/4 | 182/182 | 33/200 |
| ALFWorld Clean Cloth→Toilet | 0/4 | 0/4 | 197/200 | 29/200 |
| ScienceWorld power-component/7 | 0/4 | 0/4 | 102/185 | 4/200 |
| ScienceWorld energy/5 | 0/4 | 0/4 | 157/200 | 0/200 |

实际模型为 `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`，7,615,616,512 参数、FP16、GPU 0。每题四条原始 reset、50 轮，history=4、temperature=.7、top_p=1、4096/512 token 上限。ScienceWorld 沿用 reference_interface_v1、原 parser、1.2.3/JAR、空 simplifications。`policy_updated=false`、`training_updates=0`。原始输入解码、回答解码、解析器执行记录均逐步核对了 800/800；新旧各任务首个 prompt 一致，输入和回答均未触及截断上限。

1. **大量轮次停在动作格式层。** 7B 的 ALFWorld 338/400 步、ScienceWorld 389/400 步出现 `[action]`、`[action>` 等畸形标记；另有 7 步 ScienceWorld 标签缺失或不完整。原 parser 的末尾 30 字符 fallback 将残片交给环境。ScienceWorld 396/400 步返回无法匹配动作，全部 16 条轨迹首轮就出现方括号动作标签。这些是模型真实输出，尚不能确定它为何偏好该格式。
2. **没有出现新的准备性行动证据。** Clean 取布从 1.5B 的 4/4 变为 7B 的 2/4，持布到水槽、真实清洁均为 0/4。首次取布后直接放脏布是 3/4 对 0/2；7B 另两条未取到布，已排除分母；0/2 的后继是查看或无效命令，不代表衔接改善。ScienceWorld 7B 实际 connect 命令、focus 成功和新增连接均为零；energy 的分数始终保持 reset 的 5 分。
3. **全部 7B 离线 AE、AS、联合优势为零。** 所有终局奖励为零，原 mean_norm/gamma=.95 精确 anchor 诊断重算与 HPC 完全一致。ScienceWorld 大量重复报错使平均组大小达 80，但正常观察下可解析动作支持的分母为 0，不能把大组解读为有价值的信用证据。历史 1.5B Pick 仍有成功/失败混合及非零优势，未混入 P1。

**下一轮优先两件事（本轮未自动运行）：** 先对原始任务起点做 2–4 条短的输出格式对照，只检验明确格式示例或输出约束，单独记录接口版本，保持任务信息、奖励和解析规则固定；格式稳定后，再用原始起点检查基本操作与准备行为，决定是否重新做完整 P0。当前不宜直接进入提示 P1 或 RL 更新，也不能用离线修复标签后的假想结果替换这轮结果。

采样代码：`exp/dep-prep-p1-sci-rep0-20261012` / `1de2441b051ae8e1eecf38f3c769abb03e029916`。新目录为 `alfworld-p0-7b-159779`、`scienceworld-p0-7b-159780`。SwanLab：[ALFWorld](https://swanlab.cn/@KANSEI/gigpo-dependency-pilot/runs/w5i38pgj)、[ScienceWorld](https://swanlab.cn/@KANSEI/gigpo-dependency-pilot/runs/2ybv3b5x)。表格来源：`task_comparison.csv`、`format_failure_counts.csv`；核验：`audit.json`、`recompute_verification.json`；详细解释：`REPORT.md`。
