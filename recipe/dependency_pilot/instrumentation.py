"""Append-only raw trajectory sidecars, independent of cloud logging."""
import time
import numpy as np
from .common import append_jsonl, digest


class Observer:
    def __init__(self, run_dir, config, task, tokenizer, batch_id):
        self.run_dir, self.config, self.task = run_dir, config, task
        self.tokenizer, self.batch_id = tokenizer, batch_id
        self.episodes, self.pending, self.finished = [], [], False
        self.started = time.monotonic()

    def start(self, obs, infos, group_ids, traj_ids):
        if len(set(group_ids)) != 1:
            raise ValueError("One pilot collector batch must contain exactly one fixed task group")
        for index, (group, traj, info) in enumerate(zip(group_ids, traj_ids, infos)):
            ep = {**self.task, "run_id": self.config["run_id"], "checkpoint_id": self.config["checkpoint_id"],
                "model_revision": self.config.get("model_revision"), "algorithm": self.config.get("run_identity", "frozen_collect"),
                "condition": self.config.get("condition", "root_original"), "seed": self.config["seed"] + index,
                "rollout_batch_id": self.batch_id, "episode_group_uid": str(group), "traj_uid": str(traj),
                "expected_group_size": self.config["rollouts_per_task"], "strict_success": info.get("strict_success"),
                "definition_status": info.get("definition_status", "unknown"), "native_score": info.get("native_score"),
                "agent_turns": 0, "env_moves": 0, "reset_env_moves": info.get("reset_env_moves", 0),
                "prompt_tokens": 0, "generated_tokens": 0, "env_reward_sum": 0.0, "native_reward_sum": None,
                "train_reward_profile": self.config["reward_profile"], "any_verified_preparation": None,
                "joint_requirement_reached": None, "annotation_coverage": None, "api_calls": None,
                "termination_reason": "unknown", "status": "partial", "error_category": None}
            self.episodes.append(ep)
            append_jsonl(self.run_dir / "attempts.jsonl", ep)
            append_jsonl(self.run_dir / "resets.jsonl", {"traj_uid": str(traj), "observation": obs["anchor"][index], "info": info})

    def before_step(self, step, batch, obs, active):
        self.active = list(active)
        self.pending = []
        ids = []
        for i, ep in enumerate(self.episodes):
            step_id = f"{ep['traj_uid']}:{step}"
            ids.append(step_id)
            actual_ids = batch.batch["input_ids"][i][batch.batch["attention_mask"][i].bool()].tolist()
            chat = [{"role": "user", "content": obs["text"][i]}]
            rendered = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            original_tokens = len(self.tokenizer.encode(rendered, add_special_tokens=False))
            row = {key: ep[key] for key in ["run_id", "checkpoint_id", "env", "task_id", "variation_id", "task_instance_id", "official_split", "rollout_batch_id", "episode_group_uid", "traj_uid"]}
            row.update(analysis_step_id=step_id, step_index=step, active_mask=bool(active[i]),
                actor_prompt=obs["text"][i], rendered_prompt=rendered, rendered_prompt_hash=digest(rendered),
                actual_input_ids=actual_ids, actual_input_hash=digest(str(actual_ids)), input_tokens=len(actual_ids),
                truncated_tokens=max(0, original_tokens - len(actual_ids)), visible_history_steps=min(step + self.config.get("history_offset", 0), self.config["history_length"]),
                anchor_pre_raw=obs["anchor"][i], anchor_pre_hash=digest(obs["anchor"][i]),
                remaining_agent_turns=self.config["max_steps"] - step, native_score_pre=ep["native_score"],
                role_pre="unknown", condition_pre=None, condition_post=None,
                episode_adv_scalar=None, step_adv_scalar=None, joint_adv_scalar=None, step_group_uid=None)
            self.pending.append(row)
        return np.asarray(ids, dtype=object)

    def decisions(self, batch, responses):
        self.step_time = time.monotonic()
        for i, row in enumerate(self.pending):
            if not self.active[i]:
                continue
            mask = batch.batch["attention_mask"][i, -batch.batch["responses"].shape[-1]:]
            ids = batch.batch["responses"][i][mask.bool()].tolist()
            row.update(raw_response=responses[i], generated_tokens=len(ids), response_token_ids=ids,
                       output_at_token_limit=len(ids) >= self.config["response_tokens"])
            append_jsonl(self.run_dir / "decisions.jsonl", row)

    def after_step(self, next_obs, rewards, dones, infos):
        for i, (row, ep, info) in enumerate(zip(self.pending, self.episodes, infos)):
            if not self.active[i]:
                continue
            score = info.get("native_score")
            events = []
            if score is not None and ep["native_score"] is not None and score > ep["native_score"]:
                events.append("score_increased")
            if info.get("strict_success") is True:
                events.append("task_completed")
            row.update(parsed_action=info.get("parsed_action"), parse_valid=info.get("parse_valid"),
                env_action_valid=info.get("env_action_valid"), action_signature=info.get("parsed_action"),
                observation_post=next_obs["anchor"][i], native_info=info, native_score_post=score,
                native_reward=info.get("native_reward"), env_reward=float(rewards[i]), done_raw=bool(dones[i]),
                stop_reason=info.get("termination_reason"), event_tags=events, api_count=info.get("api_calls"),
                penalty_components={"invalid_action": 0.0, "kl": 0.0},
                env_step_seconds=time.monotonic() - self.step_time)
            ep["native_score"] = score
            ep["strict_success"] = info.get("strict_success")
            ep["definition_status"] = info.get("definition_status", ep["definition_status"])
            ep["agent_turns"] += 1
            ep["env_moves"] = info.get("env_moves")
            ep["env_reward_sum"] += float(rewards[i])
            if info.get("native_reward") is not None:
                ep["native_reward_sum"] = (ep["native_reward_sum"] or 0) + info["native_reward"]
            ep["prompt_tokens"] += row["input_tokens"]
            ep["generated_tokens"] += row["generated_tokens"]
            if dones[i]:
                ep["termination_reason"] = info.get("termination_reason") or "unknown"
                ep["status"] = "complete"
            append_jsonl(self.run_dir / "steps.jsonl", row)

    def finish(self, error=None):
        if self.finished:
            return
        for ep in self.episodes:
            if ep["status"] != "complete":
                if error:
                    ep.update(strict_success=None, status="infra_error", termination_reason="env_error", error_category=error)
                else:
                    ep.update(status="complete", termination_reason="budget_truncated")
            ep["wall_time_s"] = time.monotonic() - self.started
            append_jsonl(self.run_dir / "episodes.jsonl", ep)
        self.finished = True
