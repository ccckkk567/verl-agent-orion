# 指标口径

| 字段 | 单位与判定 |
|---|---|
| strict_success | complete episode 中，官方/本地已核实严格终点；native score 单独记录 |
| format_compliance | active step 原 parse_valid；可能受 think 标签及大小写限制影响 |
| usable_action | active step，存在完整 action 标签，提取命令非空、单行且不含残留 action/think 标签；不保证环境接受 |
| complete_xml_action | 原始回答有一对大小写不敏感完整 action 标签；比 usable_action 更宽，不可混用 |
| malformed_square_action_marker | 没有完整 action 标签，且原始回答出现 `[action` 开头的标记；仅观察分类 |
| no_match | ScienceWorld 明确返回 No known action matches that input |
| Nothing happens | ALFWorld 通用反馈；可能来自语法或前置条件，不能单独断言原因 |
| object_pickup | 相关对象新增 holds 事实，按 episode 是否曾出现汇总 |
| any_sink_reached / sink_reached | Clean 的任意状态到水槽 / 持相关布到水槽；Pick 为 N/A |
| clean_success | 同一 cloth 新出现 isclean；cleanable 或模型自述不算 |
| target_placement | 同一相关对象新增 inreceptacle 目标事实 |
| first_pickup_next_unclean_placement | 自然首次取布、仍有下一决策且状态可判的 episode 分母；未取布等原因另列 |
| connect_attempt | 实际执行命令去空白、忽略大小写后首词严格 connect |
| connection_newly_established | observer 端点边集合的实际新增；不知道则 null |
| target_focused | 目标 focus 命令、明确反馈及 native score 证据；不是任意分数增加 |
| workshop_initial / entered_workshop_after_reset | reset 已在 workshop / 初始不在、后续确实进入 |
| positive_score_no_connection | 终点正分且无连接；可能仅为初始分数，不能等同进展 |
| normal_anchor_support | 完整组中，parser 可用且非通用报错 pre-anchor 步骤里，有其他轨迹不同可用动作；分母为 0 则 unavailable |
| nonzero episode/step advantage | 完整组诊断步骤里 abs(AE/AS)>1e-8；GiGPO 分组和奖励不作标签过滤 |

null 排除已知标签分母，unknown_count 保留；不适用的 Pick 清洁指标另有 not_applicable_count。提前基础设施失败不按任务失败填入。统计是描述性结果，没有显著性或模型总体能力判断。
