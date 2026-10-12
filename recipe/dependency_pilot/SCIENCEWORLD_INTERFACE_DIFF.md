# ScienceWorld 接口核查与解析决定（2026-10-12）

本轮在 P0 commit `a3614fab98e09af9ee2a1bb2029e65ce0a1342a6` 上继续；保留原解析器，只加入公开命令语义与对象引用。S1 是参考代码支持的接口适配，并非逐字复现实验。

## 核查来源

- RLVMR repository commit `7e22784db51070fd3ed9b03a23edfca59cb5e800`：[普通 prompt](https://github.com/Tencent/digitalhuman/blob/7e22784db51070fd3ed9b03a23edfca59cb5e800/RLVMR/code/agent_system/environments/prompts/sciworld.py)、[worker](https://github.com/Tencent/digitalhuman/blob/7e22784db51070fd3ed9b03a23edfca59cb5e800/RLVMR/code/agent_system/environments/env_package/sciworld/envs.py)、[projection](https://github.com/Tencent/digitalhuman/blob/7e22784db51070fd3ed9b03a23edfca59cb5e800/RLVMR/code/agent_system/environments/env_package/sciworld/projection.py)。只参考普通提示，未引入 meta_think。
- BEACON repository commit `c88af45736b0a178106bc14ea89603ff02d8b222`：[普通 prompt](https://github.com/ZJU-REAL/BEACON/blob/c88af45736b0a178106bc14ea89603ff02d8b222/agent_system/environments/prompts/sciworld.py)、[worker](https://github.com/ZJU-REAL/BEACON/blob/c88af45736b0a178106bc14ea89603ff02d8b222/agent_system/environments/env_package/sciworld/envs.py)、[projection](https://github.com/ZJU-REAL/BEACON/blob/c88af45736b0a178106bc14ea89603ff02d8b222/agent_system/environments/env_package/sciworld/projection.py)。未引入其 score-increase 奖励、easy 默认配置或 done+positive-score 成功判据。
- 本地安装 ScienceWorld 1.2.3，JAR SHA256 `f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7`，与旧 HPC P0 一致。六份参考文件的 commit、blob SHA、SHA256 见 `post_p0_source_refs.json`；blob 与用户计划核对值一致。

## 实际输入差异

| 项目 | 旧 P0 实际 prompt | 参考实现 | 本轮 S1 |
|---|---|---|---|
| 任务、反馈、4 轮历史 | 已有 | 普通任务/历史模板 | 保留 |
| room/inventory | 已有 | 当前观察等 | 保留 |
| 动作模板 | 已有，来自 get_possible_actions | 两个 worker 均调用该接口 | 保留原模板 |
| 命令含义 | 简短输出约束 | RLVMR 普通 prompt 提供说明 | 采用简短、重新组织的通用说明 |
| 可引用对象 | 无独立对象块 | 两个 worker 均调用 get_possible_objects | 添加排序去重对象列表，固定最多 80 个，记录裁剪计数 |
| 实例化候选 | 未给 actor | worker 另取 valid combinations | 不将巨量组合或唯一正确动作 mask 给 actor |
| focus 示例 | 无 | worker 有 focus on door 示例 | 不采用；保持任务原文的 focus 要求 |
| 端点帮助 | 无 | 公开对象列表包含端点 | S1 仅公开对象；S2 额外通用说明独立命名，本轮未启用 |
| 奖励/完成 | 首次严格完成 +10 | 两工程口径与本轮不同 | 原值保留；部分分数、focus 与接线分开 |

对象排序/裁剪不读取答案或目标进度。actor 只能看到公共对象列表；全对象树的电气边仅供观察者评价，不进入 actor prompt。读取 moves 前后核对，不自动执行 look 动作。

## 解析器为什么没有修改

BEACON 和 RLVMR 使用首次 `<action>` 与首次 `</action>` 之间的内容；缺标签时取末尾 20 字符，并另行标记 think/语言格式。旧 P0 使用大小写不敏感非贪婪提取，缺标签取末尾 30 字符。保留旧规则，避免把本轮变成提示与解析器同时改变的对照。

本轮第一次真实短测出现多标签、缺标签及嵌套标签。曾考虑拒绝非唯一动作，但核对发现：

- 13 步中，参考提取与当前提取的实际命令有 9 步相同；4 步差异都是回退截断长度。
- 将思考夹在动作中的异常，参考规则也会产生同一命令。
- `power-component/7` 的零基 step 7 输出包含两对动作标签；旧规则提取首个开门命令，环境确认门打开。拒绝所有多标签会拒绝这个真实有效动作。

因此“只允许唯一完整单行动作”并非参考实现已有的修复，也没有证据表明适合直接作为主重跑默认值。当前选择 **p0_original_unchanged**，只增加诊断：格式 flag、选出的动作是否为完整非空单行、标签结构是否唯一、fallback、实际执行及环境状态变化。`parser_usable` 和 `response_action_structure_unambiguous` 分开，完整首动作可以可用而整段响应包含多个候选。已知局限是畸形输出仍可能把无效文本送入模拟器；该失败保留，不掩盖为正确操作。若后续比较严格解析器，须另列接口组合条件。

## 已执行的原生检查与模型短测

独立原生 fixture 验证：当前对象列表中的端点字符串可以成功建立连接；重复连接被拒绝；`go workshop` 受关闭门阻挡，开门后 `go to workshop` 可进入房间。共 6 次研究者操作，没有模型采样，不计主结果。具体命令只保存在 observer fixture 记录，未写进 actor prompt。

本地真实模型短测 run：`scienceworld-interface-smoke-20261012T125520Z-964477`。同 1.5B revision、空 simplifications、4096/512 token、temperature 0.7，两个任务各 1 条、上限 12 轮。实际 13 步，0/2 严格完成、0 次 connect；energy/5 首步错误 focus battery 原生失败，power/7 到预算，期间真实开门并进入 workshop。13 步均完整看见接口块，0 输入截断，电气只读查询不消耗环境 moves。

本机 Torch 2.10.0 与原 HPC Torch 2.8.0 不同；该短测仅作接口检查。完整匹配重跑仍在用户 HPC 原环境执行。S1 不因短测失败而无限调参；HPC 脚本在同环境再执行 2 条 smoke（连同本机共 4 条），锁定后必须继续 8 条 × 最多 50 轮的新 P0。检查门槛是输入/预算/通信正确，不是成功率。
