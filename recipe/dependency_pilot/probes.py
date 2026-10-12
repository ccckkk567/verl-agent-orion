"""Small explicit ScienceWorld prefix continuation; conservative replay labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

from .common import append_jsonl, digest, git_info, read_jsonl, write_json
from .environments import PilotManager


def wire_endpoint(progress):
    states = re.findall(r"^\d+\s+(true|false)\s+GoalObjectConnectedToWire\s+", progress, re.M)
    return any(s == "true" for s in states) if len(states) == 2 else None


class PrefixManager(PilotManager):
    def __init__(self, task, config, prefix, hint):
        super().__init__(task, config)
        self.prefix, self.hint = prefix, hint
        self.replay_records = []

    def reset(self, kwargs=None):
        super().reset(kwargs)
        for source in self.prefix:
            _, _, done, _ = super().step([source["raw_response"]])
            self.replay_records.append({"source_step_id": source["analysis_step_id"],
                "observation_agrees": self.text[0] == source["observation_post"],
                "score_agrees": self.infos[0]["native_score"] == source["native_score_post"]})
            if done[0]:
                raise RuntimeError("Prefix unexpectedly terminated during replay")
        env = self.sessions[0].env
        self.start_state = {"object_tree": env.getObjectTree(), "goal_progress": env.get_goal_progress(),
            "public_observation": self.text[0], "look": self.infos[0]["look"], "inventory": self.infos[0]["inventory"],
            "history": self.history[0], "moves": env.get_num_moves()}
        self.start_hash = digest(json.dumps(self.start_state, sort_keys=True))
        write_json(Path(self.config["run_dir"]) / "replayed_start_state.json", self.start_state)
        write_json(Path(self.config["run_dir"]) / "replay_checks.json", self.replay_records)
        self.local_success = wire_endpoint(self.start_state["goal_progress"])
        if self.local_success is not False:
            raise RuntimeError("Selected local endpoint must be confirmed unmet at the prefix")
        return self.observations(), self.infos

    def observations(self):
        obs = super().observations()
        if self.hint:
            obs["text"] = [text + "\nLocal goal hint (extra information): " + self.hint for text in obs["text"]]
        return obs

    def step(self, responses):
        result = super().step(responses)
        progress = self.sessions[0].env.get_goal_progress()  # side-effect-free observer query
        endpoint = wire_endpoint(progress)
        if endpoint is True:
            self.local_success = True
        elif endpoint is None and self.local_success is not True:
            self.local_success = None
        result[3][0]["probe_local_endpoint"] = endpoint
        result[3][0]["probe_goal_progress_observer_only"] = progress
        return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, help="Source frozen run")
    parser.add_argument("--traj-uid", required=True)
    parser.add_argument("--after-step", required=True, type=int, help="Inclusive zero-based source step index")
    parser.add_argument("--hint", required=True, help="One local condition, no action sequence")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 4:
        raise ValueError("This bounded probe permits 1–4 repeats per condition")
    source = Path(args.run_dir)
    config = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    if config["env"] != "scienceworld":
        raise ValueError("This first probe supports ScienceWorld only")
    episode = next(e for e in read_jsonl(source / "episodes.jsonl") if e["traj_uid"] == args.traj_uid)
    prefix = [r for r in read_jsonl(source / "steps.jsonl") if r["traj_uid"] == args.traj_uid and r["step_index"] <= args.after_step]
    if [r["step_index"] for r in prefix] != list(range(args.after_step + 1)):
        raise ValueError("Prefix must contain every actual action from reset")
    remaining = config["max_steps"] - len(prefix)
    if remaining <= 0: raise ValueError("No remaining source budget")
    out = source.parent / ("prefix-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    out.mkdir()
    write_json(out / "run_manifest.json", {"identity": "prefix_probe", "parent_run_id": config["run_id"], "git": git_info(),
        "source_traj_uid": args.traj_uid, "through_step": args.after_step, "hint": args.hint,
        "hint_source": "analyst; one known unmet wiring condition", "extra_information": True,
        "local_endpoint": "either native GoalObjectConnectedToWire becomes true", "remaining_turns": remaining,
        "response_tokens": config["response_tokens"], "repeats": args.repeats,
        "replay_status": "approximate", "reason": "original collection did not capture full object tree/RNG; matching observed history is insufficient for exact replay",
        "source_model_revision": config["model_revision"], "status": "running"})
    print(f"PROBE_DIR={out}", flush=True)
    from .runtime import TransformersWorker
    from .instrumentation import Observer
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl import DataProto
    from omegaconf import OmegaConf
    import numpy as np
    import torch
    actor = TransformersWorker(config)
    rows = []
    for repeat in range(args.repeats):
        for condition in ["prefix_original", "prefix_subgoal_hint"]:
            run_dir = out / f"{condition}-{repeat}"
            run_dir.mkdir()
            child = {**config, "max_steps": remaining, "rollouts_per_task": 1,
                "run_id": run_dir.name, "run_dir": str(run_dir), "seed": config["seed"] + repeat + 100003 * len(prefix),
                "condition": condition, "run_identity": "prefix_probe", "history_offset": len(prefix)}
            # Native move limit retains the original total budget, including replayed actions.
            manager = PrefixManager(episode, {**child, "max_steps": config["max_steps"]}, prefix,
                args.hint if condition == "prefix_subgoal_hint" else None)
            observer = Observer(run_dir, child, episode, actor.tokenizer, run_dir.name)
            actor.config, actor.observer = child, observer
            collector = TrajectoryCollector(OmegaConf.create({"data": {"max_prompt_length": config["prompt_tokens"], "truncation": "left"},
                "env": {"rollout": {"n": 1}, "max_steps": remaining}}), actor.tokenizer)
            collector.pilot_observer = observer
            raw = np.empty(1, dtype=object); raw[0] = [{"role": "user", "content": ""}]
            batch = DataProto.from_single_dict({"input_ids": torch.zeros((1, 1), dtype=torch.long), "raw_prompt": raw,
                "data_source": np.array(["scienceworld"], dtype=object)})
            write_json(run_dir / "resolved_config.json", child)
            row = {"prefix_id": args.traj_uid + ":" + str(args.after_step), "condition": condition,
                "repeat": repeat, "seed": child["seed"], "remaining_turns": remaining,
                "response_tokens": config["response_tokens"], "replay_status": "approximate"}
            try:
                collector.vanilla_multi_turn_loop(batch, actor, manager)
                row.update(local_stage_success=manager.local_success, strict_success=manager.infos[0].get("strict_success"),
                    start_state_hash=manager.start_hash, public_replay_agrees=all(all(check[k] for k in ["observation_agrees", "score_agrees"]) for check in manager.replay_records),
                    generated_tokens=observer.episodes[0]["generated_tokens"], status="complete")
            except Exception as exc:
                observer.finish(error=f"{type(exc).__name__}: {exc}")
                row.update(status="infra_error", error=str(exc), local_stage_success=None, strict_success=None)
            finally:
                manager.close()
            rows.append(row)
            append_jsonl(out / "probes.jsonl", row)
            print(json.dumps(row), flush=True)
    for repeat in range(args.repeats):
        pair = [r for r in rows if r["repeat"] == repeat]
        equal = len(pair) == 2 and pair[0].get("start_state_hash") is not None and pair[0].get("start_state_hash") == pair[1].get("start_state_hash")
        append_jsonl(out / "pair_checks.jsonl", {"repeat": repeat, "start_snapshot_hash_agrees": equal,
            "budgets_agree": len({(r["remaining_turns"], r["response_tokens"]) for r in pair}) == 1,
            "strict_paired_analysis_eligible": False})
    report_probe(out, rows)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "partial_useful_approximate_replay" if any(r["status"] == "complete" for r in rows) else "blocked_infra"
    write_json(out / "run_manifest.json", manifest)
    print(f"PROBE_COMPLETE={out}", flush=True)


def report_probe(out, rows):
    from .analysis import csv_table
    csv_table(out / "probes.csv", rows, ["prefix_id", "condition", "repeat", "local_stage_success", "strict_success", "public_replay_agrees", "start_state_hash", "remaining_turns", "response_tokens", "generated_tokens", "status"])
    summary = {}
    for condition in ["prefix_original", "prefix_subgoal_hint"]:
        subset = [r for r in rows if r["condition"] == condition]
        summary[condition] = {key: {"numerator": sum(r[key] is True for r in subset), "denominator": sum(r[key] is not None for r in subset)} for key in ["local_stage_success", "strict_success"]}
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text("# First prefix continuation check\n\nApproximate replay: descriptive only; excluded from strict paired causal results. Original full simulator state/RNG was not recorded. Both conditions use the same real action prefix, visible history, remaining turns, response token budget and local endpoint. The hint is extra information.\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\nSee pair_checks.jsonl for B/C snapshot agreement and child steps.jsonl for actual actions. No training update was performed.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
