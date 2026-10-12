# 本轮阶段交付：2026-10-12

## 已经执行与尚未执行

| 项目 | 状态 | 证据 |
|---|---|---|
| BEACON/RLVMR 普通 prompt、worker、parser 核查 | 已完成 | 六份固定 commit 文件与哈希；SCIENCEWORLD_INTERFACE_DIFF.md |
| 原生 ScienceWorld 端点、重复连接、导航检查 | 已运行 | 6 次真实手工 fixture 操作，独立于模型成绩 |
| ScienceWorld S1 本地模型 smoke | completed_exploratory | 2 条轨迹、13 active steps，原始 JSONL 与图已保存 |
| 四个 ALFWorld 旧前缀提取 | 已完成只读数据核对 | 原轨迹、首次取布、39/35/47/34 预算匹配；尚未远端重放 |
| ALFWorld P1 | code_ready，尚未实际运行 | 24 后缀 runner、Slurm、事实标签、分组优势、图和报告已实现 |
| ScienceWorld 匹配新 P0 | code_ready，尚未实际运行 | HPC 脚本先 2 条短测，锁定后接着原 reset、8 条最多 50 轮 |
| 完整新旧 P0 比较 | 尚无新 P0 数据 | 同版旧数据重分析已运行，保留旧源文件；比较入口已实现 |

本地输出根目录：`C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\post-p0-20261012`。完整原始结果没有放入 Git。HPC 操作仍需用户提交，助手没有直连或声称远端已运行。

## 真实短测观察

1. 接口输入已实际验证：13/13 步完整包含新对象/命令块，0 输入截断；读取电气状态未增加 native moves。原生 fixture 证明公开端点引用可执行。此证据支持“接口信息已传到模型”，不证明模型会正确使用。
2. 短测严格成功 0/2。energy/5 第一步 focus battery 后原生失败；power/7 走完 12 个 agent turns，实际开门并进入 workshop，但没有 connect 尝试。8/13 步 no-match；提前失败与短预算使此比例不能作为完整新旧 P0 的性能比较。
3. 原解析器保留。两个参考实现同样可能提取到畸形内容；严格拒绝多标签会拒绝本次有效开门动作。旁路记录格式、可提取动作、结构唯一性、fallback 与执行结果；不据此改奖励，也不把格式错误自动算原生非法。该选择存在已知解析局限，后续若比较严格解析器需独立条件。

新本地模型短测 ID：`scienceworld-interface-smoke-20261012T125520Z-964477`。模型冻结、参数更新 0；每题仅 1 条，所有 AE/AS 为 0。完整 JSONL、metrics.json、event_annotations、figure 与 run_manifest 在同名目录。本次本地 tracking 为 **disabled**，不是已上传；HPC 默认继续 SwanLab cloud，失败保留本地结果。

19 项检查通过，包含原 P0 回归、JSON RPC、四个预算、cleanable/isclean、unknown 分母、连接与 focus 分开、合成重放的相同历史以及 hint 首步一次。合成 fixture 不作为实验结果。新增 Slurm 脚本通过 Bash 语法检查。首轮本地 tracking 曾因复用旧报告格式退出，采样原始记录完好；已更换为现有 Tracking 的独立调用并实际验证 disabled 状态。

## 下一步只做两项

1. 在 HPC 运行 ALFWorld 的四前缀×三条件×两次，优先查看 p3 的第一批真实结果。观察是否实际到设施、是否 isclean、清洁后是否完成；若三条件都不会清洁，再安排独立合法起点能力检查。
2. 在 HPC 原软件环境完成 ScienceWorld 8×50 匹配新 P0，不要求成功才交付。依据真实连接、严格完成和正常观察 anchor 支持决定是否值得进入电路 P1。没有完整重跑前，不能宣称接口改善或任务已完成。

提交命令、输出位置与故障后的完整记录入口见 POST_P0_RUNBOOK.md。
