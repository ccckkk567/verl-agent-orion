# P0 metric units

Every rate saves `numerator`, `denominator`, `value`, `scope`, `status`. A zero denominator has `value: null`, never zero. Outcomes are reported within one environment only.

| Prefix/key | Unit and definition |
|---|---|
| eval/success_macro | Mean of per-task-instance success fractions over evaluable episodes |
| eval/success_micro | Successful episodes / evaluable episodes |
| eval/known_outcome_rate | Evaluable episodes / attempted episodes; infra errors remain unknown |
| eval/native_score_mean | Mean native final score among episodes with that field |
| eval/budget_truncated_rate | Budget-ended episodes / evaluable episodes; bounded failure is an outcome |
| data/parse_valid_rate | Upstream parser-accepted active responses / active responses with known parser result |
| data/env_action_valid_rate | Native-valid actions / actions with explicit validity evidence; ScienceWorld currently unknown |
| data/prompt_truncation_rate | Active generated decisions whose rendered prompt lost tokens / active decisions |
| data/output_token_limit_rate | Generated responses reaching the configured token limit / active decisions; includes EOS at exactly the limit |
| data/group_* | Complete groups only: all expected trajectories present, all outcomes known. `group_known_rate` reports coverage |
| coverage/condition_known_rate; phase/* | Initially NA until a condition specification and evidence-backed annotation exist |
| cost/env_interactions_total | Recorded model-issued environment actions; native reset moves separate |
| cost/generated_tokens_total | Active response tokens, including EOS; no reference/replay tokens counted |
| anchor/group_size_mean | Occurrences per unique exact pre-action anchor group |
| anchor/unique_trajectories_mean | Distinct trajectory IDs per unique group; repeated visits are not independent samples |
| anchor/unique_actions_mean | Distinct full action strings of parseable actions per group |
| anchor/alternative_action_step_coverage | Parseable decisions with another trajectory taking a different parseable action in the same group / parseable decisions |
| anchor/alternative_action_step_coverage_after_first | Same metric after excluding decision index 0 from the evaluated steps |
| anchor/max_trajectory_share_mean | Mean maximum trajectory occurrence share per group |
| anchor/return_std_mean | Mean population standard deviation of discounted terminal returns within groups |
| adv/step_nonzero_rate | Absolute step advantage > 1e-6 / active diagnostic steps |
| adv/episode_abs_mean; adv/step_abs_mean | Scalar step-level means; not token-weighted |
| adv/reconstruction_max_abs_error | max abs(joint - episode - w*step), compared with upstream function output |

P0 diagnostics reconstruct **frozen** terminal-reward inputs with upstream functions, preserving trajectory-major/temporal order, gamma 0.95, mean normalization, w=1, no similarity or dynamic filtering. Episode rewards are repeated across flattened steps, matching the existing episode normalization default. Invalid-action penalty and KL are explicitly zero in this frozen profile. These are not captured training advantages, and `episode_only_matched` is not renamed GRPO.

The current P0 implementation does not export P1 optimizer, actual-update, prefix or condition-opportunity metrics. Their absence is recorded as unavailable rather than fabricated.
