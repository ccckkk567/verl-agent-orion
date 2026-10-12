"""Recompute using exact upstream functions; retain collector trajectory/step order."""
from collections import Counter, defaultdict
from types import SimpleNamespace


def recompute(steps, episodes, config):
    import numpy as np
    import torch
    from gigpo.core_gigpo import (build_step_group, compute_step_discounted_returns,
        compute_gigpo_outcome_advantage, episode_norm_reward, step_norm_reward)

    by_traj = defaultdict(list)
    for step in steps:
        if step["active_mask"]:
            by_traj[step["traj_uid"]].append(step)
    # Upstream gather_rollout_data flattens trajectory-major, preserving temporal order.
    rows = [row for ep in episodes for row in sorted(by_traj[ep["traj_uid"]], key=lambda r: r["step_index"])]
    if not rows:
        return [], [], {"status": "unavailable", "reason": "no active steps"}
    identities = defaultdict(set)
    for r in rows:
        identities[r["episode_group_uid"]].add((r["task_instance_id"], r["checkpoint_id"], r["rollout_batch_id"]))
    if any(len(v) != 1 for v in identities.values()):
        raise ValueError("Mixed task/checkpoint/batch in recorded episode group")
    episodic = {e["traj_uid"]: e["env_reward_sum"] for e in episodes}
    group_ids = np.array([r["episode_group_uid"] for r in rows], dtype=object)
    traj_ids = np.array([r["traj_uid"] for r in rows], dtype=object)
    anchor = np.array([r["anchor_pre_raw"] for r in rows], dtype=object)
    # One valid response token is sufficient: upstream broadcasts the scalar to all valid tokens.
    mask = torch.ones((len(rows), 1), dtype=torch.float32)
    episode_scores = torch.tensor([[episodic[r["traj_uid"]]] for r in rows], dtype=torch.float32)
    payload = SimpleNamespace(non_tensor_batch={"rewards": np.array([r["env_reward"] for r in rows]),
        "traj_uid": traj_ids, "active_masks": np.ones(len(rows))}, batch={"input_ids": torch.zeros((len(rows), 1))})
    step_returns = compute_step_discounted_returns(payload, gamma=config["gamma"])
    group = build_step_group(anchor, group_ids, enable_similarity=False)
    a_e = episode_norm_reward(episode_scores.clone(), mask, group_ids, traj_ids, remove_std=True)
    a_s = step_norm_reward(step_returns, mask, group, remove_std=True)
    joint, _ = compute_gigpo_outcome_advantage(episode_scores.clone(), step_returns, mask, anchor,
        group_ids, traj_ids, step_advantage_w=config["step_advantage_w"], mode="mean_norm", enable_similarity=False)
    matched, _ = compute_gigpo_outcome_advantage(episode_scores.clone(), step_returns, mask, anchor,
        group_ids, traj_ids, step_advantage_w=0, mode="mean_norm", enable_similarity=False)
    error = float((joint - a_e - config["step_advantage_w"] * a_s).abs().max())
    matched_error = float((matched - a_e).abs().max())
    result, groups = [], defaultdict(list)
    for i, r in enumerate(rows):
        out = {"analysis_step_id": r["analysis_step_id"], "traj_uid": r["traj_uid"],
            "episode_group_uid": r["episode_group_uid"], "step_group_uid": str(group[i]), "step_index": r["step_index"],
            "action_signature": r["action_signature"], "parse_valid": r["parse_valid"], "role_pre": r["role_pre"],
            "step_return_raw": float(step_returns[i]), "step_return_used": float(step_returns[i]),
            "episode_adv_scalar": float(a_e[i, 0]), "step_adv_scalar": float(a_s[i, 0]), "joint_adv_scalar": float(joint[i, 0]),
            "source": "diagnostic_recomputed", "norm_mode": "mean_norm", "penalty": 0.0, "kl": 0.0}
        result.append(out)
        groups[str(group[i])].append(out)
    anchors = []
    for gid, members in groups.items():
        counts = Counter(m["traj_uid"] for m in members)
        actions = {m["action_signature"] for m in members if m["parse_valid"]}
        returns = [m["step_return_used"] for m in members]
        anchors.append({"step_group_uid": gid, "episode_group_uid": members[0]["episode_group_uid"],
            "occurrences": len(members), "unique_trajectories": len(counts), "unique_actions": len(actions),
            "max_trajectory_share": max(counts.values()) / len(members), "return_std": float(np.std(returns)),
            "role_pre": "unknown", "analysis_step_ids": [m["analysis_step_id"] for m in members]})
        for member in members:
            member["alternative_action_supported"] = any(m["traj_uid"] != member["traj_uid"] and m["parse_valid"] and member["parse_valid"] and m["action_signature"] != member["action_signature"] for m in members)
    return result, anchors, {"status": "diagnostic_recomputed", "reconstruction_max_abs_error": error,
        "episode_only_matched_max_abs_error": matched_error,
        "scope": "frozen terminal reward, no invalid-action penalty or KL; not captured training advantages"}
