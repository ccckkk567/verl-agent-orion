# 实现、复核与边界

本次两项真实 GPU 采样均来自 `1de2441b051ae8e1eecf38f3c769abb03e029916`，远端 run manifest 记录工作区 clean。原分支 `exp/dep-prep-p1-sci-rep0-20261012`。

本次本地复核不修改 rollout、parser、prompt、奖励、GiGPO core 或已下载结果。复用 `post_audit compare` / `plot-compare`，将新旧数据以同一 v3 规则在新目录复算。两个新 analysis.json 与 HPC 版本完全相同，800 条离线诊断的 AE、AS、联合优势也完全相同。

附加工作只有分析：

- `helpers/audit_results.py`：检查模型身份、tokenizer 文件哈希、实现快照、baseline hashes、组与 step ID、seed、原始 reset 和首次 prompt；解码全部输入/回答并按原 parser 重新提取，核对记录。增加回答标签类别和案例索引，不修复、不执行候选命令。
- `helpers/build_review.py`：合并既有指标和 episode 表，核对与 HPC 的一致性，使用真实数值作图。Torch/GiGPO 与 matplotlib 分进程运行。
- `implementation_snapshot/`：采样时实现快照，补充原 ALFWorld parser 与 prompt 文件（取自同一个采样 Git commit）。不包含模型权重和凭据。

主指标定义未改变。格式合规使用原 parse_valid；parser_usable 是诊断性的完整非空单行命令检查；真实执行看实际 parsed_action、反馈和状态事实。AF cleanable 不等于 isclean，cloth 1/3/4 均按相关物体事实识别，Pick 清洁指标 N/A。SW strict connect 首词、focus、原生分数、新连接和官方完成分别统计。正常 anchor 支持以 parser 可用且非通用反馈的完整组步骤为分母；原 GiGPO 分组不据此过滤。

局限：每个固定实例只有四条；不同模型采样输出差异受格式失败强烈影响。没有反事实环境执行，无法断言把方括号修正后会成功。task_relevant_connection 缺少可判证据时仍为 null。没有远端包校验文件，仅进行本地 SHA256 留档、tar/JSON/文件一致性检查；原始模型权重不在结果包内，模型加载身份通过运行时记录、参数量、配置与校验过的下载流程核验。

完整 prompt、response、facts 和案例状态保留本地。Git 只保存复核脚本、聚合指标、图、报告、来源哈希和少量可定位短案例；没有上传原始大包或 tracker 缓存。
