"""Synthetic fixtures ONLY for correctness; never included in pilot reports."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from recipe.dependency_pilot.analysis import metrics_for, rate
from recipe.dependency_pilot.diagnostics import recompute
from recipe.dependency_pilot.environments import terminal_reward


def episode(task, traj, success, group=None):
    return dict(env="scienceworld", task_id=task, task_instance_id=task, traj_uid=traj,
        strict_success=success, status="complete" if success is not None else "infra_error", native_score=0,
        agent_turns=1, termination_reason="task_success" if success else "budget_truncated",
        checkpoint_id="synthetic", rollout_batch_id="b0", episode_group_uid=group or traj,
        expected_group_size=1, generated_tokens=1, env_reward_sum=10 if success else 0)


def test_unknown_and_macro_micro_denominators():
    eps = [episode("a", "1", True), episode("b", "2", False), episode("b", "3", False), episode("c", "4", None)]
    m = metrics_for(eps, [])
    assert m["eval/success_macro"]["value"] == 0.5
    assert m["eval/success_micro"]["value"] == 1/3
    assert m["eval/known_outcome_rate"]["denominator"] == 4
    assert m["phase/preparation_reached_rate"]["value"] is None
    assert rate(0, 0, "empty")["value"] is None


def test_terminal_reward_once():
    assert terminal_reward(None, False) == 0
    assert terminal_reward(False, False) == 0
    assert terminal_reward(True, False) == 10
    assert terminal_reward(True, True) == 0


def test_gigpo_reconstruction_scope_and_discount_order():
    eps = [episode("a", "one", True, "g1"), episode("a", "two", False, "g1")]
    rows = []
    for ep in eps:
        ep["expected_group_size"] = 2
        for t in range(2):
            rows.append(dict(active_mask=True, traj_uid=ep["traj_uid"], step_index=t,
                analysis_step_id=f"{ep['traj_uid']}:{t}", task_instance_id="a", checkpoint_id="synthetic",
                rollout_batch_id="b0", episode_group_uid="g1", anchor_pre_raw=f"pre{t}",
                env_reward=10 if ep["strict_success"] and t == 1 else 0, parse_valid=True,
                action_signature=ep["traj_uid"], role_pre="unknown"))
    results, anchors, status = recompute(rows, eps, {"gamma": .95, "step_advantage_w": 1})
    assert results[0]["step_return_raw"] == pytest.approx(9.5)
    assert status["reconstruction_max_abs_error"] < 1e-6
    assert status["episode_only_matched_max_abs_error"] < 1e-6
    assert len(anchors) == 2
    rows[2]["rollout_batch_id"] = "other"
    with pytest.raises(ValueError, match="Mixed"):
        recompute(rows, eps, {"gamma": .95, "step_advantage_w": 1})


class TinyTokenizer:
    pad_token_id = 0
    def apply_chat_template(self, chat, **kwargs): return str(chat[0]["content"])
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    def batch_decode(self, values, **kwargs): return ["<think>x</think><action>wait</action>" for _ in values]


class TinyWorker:
    world_size = 1
    def generate_sequences(self, batch):
        from verl import DataProto
        prompts = batch.batch["input_ids"]
        responses = torch.full((len(batch), 2), 3, dtype=torch.long)
        attn = torch.cat([batch.batch["attention_mask"], torch.ones_like(responses)], -1)
        return DataProto.from_single_dict(dict(prompts=prompts, responses=responses,
            input_ids=torch.cat([prompts, responses], -1), attention_mask=attn, position_ids=attn.cumsum(-1)-1))


class TinyManager:
    def reset(self, kwargs=None):
        self.t = 0
        return {"text": ["pre0", "pre0"], "image": None, "anchor": ["pre0", "pre0"]}, [{"strict_success": False, "native_score": 0}] * 2
    def step(self, actions):
        self.t += 1
        infos = [dict(strict_success=self.t == 2, native_score=100 if self.t == 2 else 50,
            native_reward=50, parsed_action="wait", parse_valid=True, env_action_valid=True,
            termination_reason="task_success" if self.t == 2 else None)] * 2
        return {"text": [f"pre{self.t}"] * 2, "image": None, "anchor": [f"pre{self.t}"] * 2}, np.array([10. if self.t == 2 else 0.] * 2), np.array([self.t == 2] * 2), infos
    def success_evaluator(self, **kwargs): return {"success_rate": np.ones(2)}


def test_observer_does_not_change_collector_and_ids_survive_flatten(tmp_path):
    from verl import DataProto
    from omegaconf import OmegaConf
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from recipe.dependency_pilot.instrumentation import Observer
    cfg = OmegaConf.create({"data": {"max_prompt_length": 16, "truncation": "left"}, "env": {"max_steps": 3, "rollout": {"n": 2}}})
    task = dict(env="scienceworld", task_id="synthetic", task_instance_id="synthetic", variation_id=0, official_split="train")
    config = dict(run_id="synthetic_fixture", checkpoint_id="synthetic", seed=0, rollouts_per_task=2,
                  max_steps=3, history_length=4, response_tokens=2, reward_profile="research_terminal_v1")
    outputs = []
    for enabled in (False, True):
        raw = np.empty(2, dtype=object)
        for i in range(2): raw[i] = [{"role": "user", "content": ""}]
        batch = DataProto.from_single_dict(dict(input_ids=torch.zeros((2,1), dtype=torch.long), raw_prompt=raw, data_source=np.array(["test", "test"])))
        collector = TrajectoryCollector(cfg, TinyTokenizer())
        if enabled: collector.pilot_observer = Observer(tmp_path, config, task, TinyTokenizer(), "batch")
        result = collector.vanilla_multi_turn_loop(batch, TinyWorker(), TinyManager())
        outputs.append(result)
        if enabled:
            flat = collector.gather_rollout_data(*result)
            assert len(set(flat.non_tensor_batch["analysis_step_id"])) == 4
    assert np.array_equal(outputs[0][1], outputs[1][1])
    assert np.array_equal(outputs[0][2], outputs[1][2])
    for before, after in zip(outputs[0][0], outputs[1][0]):
        assert [x["rewards"] for x in before] == [x["rewards"] for x in after]
        assert all(torch.equal(a["responses"], b["responses"]) for a,b in zip(before, after))
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["anchor_pre_raw"] for r in rows] == ["pre0", "pre0", "pre1", "pre1"]
    assert [r["observation_post"] for r in rows] == ["pre1", "pre1", "pre2", "pre2"]


def test_positive_partial_score_and_done_are_not_success():
    from types import SimpleNamespace
    from recipe.dependency_pilot.environments import ScienceSession
    session = ScienceSession.__new__(ScienceSession)
    session.env = SimpleNamespace(server=SimpleNamespace(getScore=lambda: 0.5, getCompleted=lambda: False))
    session.initial_moves = 0
    session.info = dict(score=50, taskDesc="synthetic", look="", inv="", valid=[], moves=51)
    result = session.details("", 0, True, "wait")
    assert result["strict_success"] is False
    assert result["termination_reason"] == "budget_truncated"


def test_disabled_tracking_always_writes_local_state(tmp_path):
    from recipe.dependency_pilot.runtime import PilotTracking
    tracking = PilotTracking(tmp_path, {"swanlab_mode": "disabled"})
    assert tracking.tracking is None
    assert json.loads((tmp_path / "tracking.json").read_text(encoding="utf-8"))["mode"] == "disabled"
def test_probe_endpoint_unknown_is_not_false():
    from recipe.dependency_pilot.probes import wire_endpoint
    assert wire_endpoint("") is None
    assert wire_endpoint("2 false GoalObjectConnectedToWire first\n3 false GoalObjectConnectedToWire second") is False
    assert wire_endpoint("2 true GoalObjectConnectedToWire first\n3 false GoalObjectConnectedToWire second") is True
