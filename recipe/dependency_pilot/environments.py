"""Small adapters for a fixed task group. No process-wide Ray services are started."""
import json
import os
import random
import re
from pathlib import Path

from .common import ROOT, digest, file_hash, versions


SCIENCE_FAMILIES = ["power-component", "power-component-renewable-vs-nonrenewable-energy"]
ALF_FAMILIES = ["pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep", "pick_two_obj_and_place", "pick_and_place_simple"]


def science_provenance():
    from scienceworld.constants import JAR_PATH
    import scienceworld.scienceworld as source
    return {"package_version": versions()["scienceworld"], "jar_path": str(JAR_PATH),
            "jar_sha256": file_hash(JAR_PATH), "wrapper_sha256": file_hash(source.__file__),
            "source_path": source.__file__, "success_definition": "simulator score >= 1 AND native getCompleted(); see SOURCES.md",
            "success_definition_status": "verified_score_contract"}


def enumerate_tasks(env_name, config):
    """Enumerate installed assets before selection. Never invent instance IDs."""
    external = os.environ.get(env_name.upper() + "_PYTHON")
    if external and env_name == "appworld" and not config.get("inside_env_worker"):
        from .env_rpc import EnvProcess
        remote = EnvProcess(external, config)
        try:
            return remote.call("enumerate", env=env_name, config=config)
        finally:
            remote.close()
    rng = random.Random(config["seed"])
    tasks, metadata = [], {}
    if env_name == "scienceworld":
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv(envStepLimit=config["max_steps"])
        try:
            metadata = science_provenance()
            for family in SCIENCE_FAMILIES:
                if family not in env.get_task_names():
                    raise RuntimeError(f"Installed ScienceWorld has no task {family}")
                env.load(family, 0, config["simplifications"])
                for split, variations in [("train", env.get_variations_train()), ("dev", env.get_variations_dev())]:
                    variations = list(variations)
                    rng.shuffle(variations)
                    for rank, variation in enumerate(variations):
                        tasks.append({"env": env_name, "task_id": family, "task_family": family,
                            "variation_id": int(variation), "task_instance_id": f"{family}/{variation}",
                            "official_split": split, "pilot_split": ("confirmation_dev" if split == "dev" and rank % 2 else f"discovery_{split}"),
                            "selection_reason": "predeclared power family; seeded instance order, not outcome-selected",
                            "source_path": metadata["jar_path"], "source_hash": metadata["jar_sha256"],
                            "simplifications": config["simplifications"], "analysis_annotations": [], "structure_notes": None})
        finally:
            env.close()
    elif env_name == "alfworld":
        data = Path(os.environ["ALFWORLD_DATA"]) / "json_2.1.1"
        for split, directory in [("train", "train"), ("dev", "valid_seen")]:
            for path in sorted((data / directory).rglob("game.tw-pddl")):
                trajectory = path.with_name("traj_data.json")
                if not trajectory.exists():
                    continue
                td = json.loads(trajectory.read_text(encoding="utf-8"))
                game = json.loads(path.read_text(encoding="utf-8"))
                if td["task_type"] not in ALF_FAMILIES or not game.get("solvable") or "Sliced" in str(path) or "movable" in str(path):
                    continue
                tasks.append({"env": env_name, "task_id": path.parent.name, "task_family": td["task_type"],
                    "task_instance_id": path.parent.relative_to(data).as_posix(), "variation_id": None,
                    "official_split": split, "source_split": directory, "pilot_split": f"discovery_{split}",
                    "selection_reason": "predeclared Heat/Cool/Clean/Pick2/Pick families", "source_path": str(path),
                    "source_hash": file_hash(path), "analysis_annotations": [], "structure_notes": None})
        rng.shuffle(tasks)
    elif env_name == "appworld":
        from appworld import load_task_ids
        from appworld.task import Task
        for split in ["train", "dev"]:
            ids = list(load_task_ids(split))
            rng.shuffle(ids)
            for task_id in ids[:max(8, config["max_tasks"] * 2)]:
                task = Task.load(task_id=task_id)
                instruction = task.instruction
                tasks.append({"env": env_name, "task_id": task_id, "task_instance_id": task_id,
                    "task_family": "candidate_dependency_unverified", "variation_id": None,
                    "official_split": split, "pilot_split": f"discovery_{split}", "instruction": instruction,
                    "selection_reason": "seeded train/dev interaction candidate; semantic dependency requires review",
                    "source_path": str(Path(os.environ.get("APPWORLD_ROOT", ".")) / "data/tasks" / task_id),
                    "source_hash": digest(instruction), "analysis_annotations": [], "structure_notes": None})
                if hasattr(task, "close"):
                    task.close()
    else:
        raise ValueError(env_name)
    if not tasks:
        raise RuntimeError(f"No installed legal train/dev tasks for {env_name}")
    # Round-robin families and splits, holding confirmation tasks out of collection.
    buckets = {}
    for task in tasks:
        if task["pilot_split"] != "confirmation_dev":
            buckets.setdefault((task["task_family"], task["official_split"]), []).append(task)
    selected = []
    while len(selected) < config["max_tasks"] and any(buckets.values()):
        for key in sorted(buckets, key=lambda key: (key[1] != "train", key[0])):
            bucket = buckets[key]
            if bucket and len(selected) < config["max_tasks"]:
                selected.append(bucket.pop(0))
    return selected, tasks, metadata


def terminal_reward(success, already_paid):
    return 10.0 if success is True and not already_paid else 0.0


class ScienceSession:
    def __init__(self, task, config):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv(envStepLimit=config["max_steps"] + 1)
        self.task, self.config = task, config
        self.paid = False

    def reset(self):
        self.env.load(self.task["task_id"], self.task["variation_id"], self.config["simplifications"])
        obs, self.info = self.env.reset()
        if self.info["taskName"] != self.task["task_id"] or self.info["variationIdx"] != self.task["variation_id"]:
            raise RuntimeError("ScienceWorld reset identity mismatch")
        self.initial_moves = self.info["moves"]
        return obs, self.details(obs, 0, False, None)

    def details(self, obs, reward, done, action):
        score = self.info["score"]
        full = float(self.env.server.getScore()) >= 1.0 and bool(self.env.server.getCompleted())
        extra = {}
        if getattr(self, "config", {}).get("phase") == "post_p0":
            from .post_interface import electrical_snapshot
            extra = {"electrical_observer": electrical_snapshot(self.env), "native_score_scale": "0-100 (failure -100)",
                     "outer_truncated": False, "native_budget_truncated": bool(done and score >= 0 and not full)}
        return {**extra, "native_score": score, "native_reward": reward, "strict_success": full,
                "definition_status": "verified_score_contract", "won": full,
                "simulator_score": float(self.env.server.getScore()), "native_completed": bool(self.env.server.getCompleted()),
                "task_description": self.info["taskDesc"], "look": self.info["look"], "inventory": self.info["inv"],
                "valid_actions": self.info["valid"], "done_raw": bool(done),
                "env_moves": self.info["moves"], "reset_env_moves": self.initial_moves,
                "termination_reason": "task_success" if full else ("task_failure" if score < 0 else ("budget_truncated" if done else None))}

    def step(self, action):
        valid_before = self.info["valid"]
        obs, native, done, self.info = self.env.step(action)
        info = self.details(obs, native, done, action)
        info["action_in_native_valid_list"] = action in valid_before
        # The native list includes broad connect candidates; membership is not execution success.
        info["env_action_valid"] = None
        reward = terminal_reward(info["strict_success"], self.paid)
        self.paid |= info["strict_success"]
        return obs, reward, done, info

    def close(self):
        self.env.close()


class AlfSession:
    def __init__(self, task, config):
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        cfg = yaml.safe_load((ROOT / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml").read_text(encoding="utf-8"))
        cfg["dataset"]["num_train_games"] = -1
        cfg["rl"]["training"]["max_nb_steps_per_episode"] = config["max_steps"]
        cfg["dagger"]["training"]["max_nb_steps_per_episode"] = config["max_steps"]
        base = AlfredTWEnv(cfg, train_eval="train" if task["official_split"] == "train" else "eval_in_distribution")
        base.use_expert = False
        if task["source_path"] not in base.game_files:
            raise RuntimeError("Selected game is not in installed ALFWorld sampler")
        base.game_files, base.num_games = [task["source_path"]], 1
        self.env = base.init_env(batch_size=1)
        self.env.seed(config["seed"])
        self.task, self.paid, self.turn = task, False, 0

    def unpack(self, infos):
        return {key: value[0] for key, value in infos.items()}

    def reset(self):
        obs, infos = self.env.reset()
        self.info = self.unpack(infos)
        actual = Path(self.info["extra.gamefile"]).resolve()
        if actual != Path(self.task["source_path"]).resolve():
            raise RuntimeError("ALFWorld reset identity mismatch")
        self.description = obs[0].split("Your task is to: ")[-1].strip()
        return obs[0], {**self.info, "task_description": self.description, "native_score": 0,
                       "strict_success": False, "definition_status": "native_won", "env_moves": 0}

    def step(self, action):
        valid = action in self.info["admissible_commands"]
        obs, scores, dones, infos = self.env.step([action])
        self.turn += 1
        self.info = self.unpack(infos)
        won = bool(self.info["won"])
        reward = terminal_reward(won, self.paid)
        self.paid |= won
        return obs[0], reward, bool(dones[0]), {**self.info, "native_score": float(scores[0]),
            "native_reward": None, "native_reward_status": "TextWorld returns score; not treated as immediate reward",
            "strict_success": won, "definition_status": "native_won", "env_action_valid": valid,
            "task_description": self.description, "env_moves": self.turn,
            "termination_reason": "task_success" if won else ("budget_truncated" if dones[0] else None)}

    def close(self):
        self.env.close()


class AppSession:
    def __init__(self, task, config):
        from appworld import AppWorld, load_task_ids
        if task["official_split"] not in ("train", "dev") or task["task_id"] not in load_task_ids(task["official_split"]):
            raise ValueError("AppWorld pilot permits only verified train/dev task IDs")
        self.env = AppWorld(task_id=task["task_id"], experiment_name=config["run_id"] + "-" + config["session_id"])
        self.task, self.config, self.turn, self.paid = task, config, 0, False

    def reset(self):
        return self.env.task.instruction, {"task_description": self.env.task.instruction,
            "supervisor": dict(self.env.task.supervisor), "strict_success": None,
            "definition_status": "native_evaluate_at_endpoint", "env_moves": 0}

    def step(self, action):
        obs = self.env.execute(action)
        self.turn += 1
        completed = bool(self.env.task_completed())
        done = completed or self.turn >= self.config["max_steps"]
        success = bool(self.env.evaluate().success) if done else None
        reward = terminal_reward(success, self.paid)
        self.paid |= success is True
        return obs, reward, done, {"won": success, "strict_success": success,
            "definition_status": "native_evaluate_at_endpoint", "native_score": None, "native_reward": None,
            "env_moves": self.turn, "api_calls": None, "env_action_valid": None,
            "termination_reason": "task_success" if success else ("task_failure" if completed else ("budget_truncated" if done else None))}

    def close(self):
        self.env.close()


def parse_action(raw, env_name):
    if env_name in ("alfworld", "appworld"):
        # Load the unchanged upstream parser without importing unrelated environment workers.
        import importlib.util
        path = ROOT / f"agent_system/environments/env_package/{env_name}/projection.py"
        spec = importlib.util.spec_from_file_location(f"pilot_{env_name}_projection", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        projection = getattr(module, env_name + "_projection")
        actions, valids = projection([raw], [[]]) if env_name == "alfworld" else projection([raw])
        return actions[0], bool(valids[0])
    match = re.search(r"<action>(.*?)</action>", raw, re.S | re.I)
    action = match.group(1).strip() if match else raw[-30:]
    valid = bool(match and "<think>" in raw and "</think>" in raw and "\n" not in action)
    return action, valid


class PilotManager:
    """The upstream collector's manager interface with exact, fixed-instance reset."""
    def __init__(self, task, config):
        self.task, self.config = task, config
        self.sessions = []
        if task["env"] == "appworld" and config["rollouts_per_task"] > 1 and not os.environ.get("APPWORLD_PYTHON"):
            raise ValueError("Set APPWORLD_PYTHON for isolated per-trajectory AppWorld processes")
        self.session_cls = {"scienceworld": ScienceSession, "alfworld": AlfSession, "appworld": AppSession}[task["env"]]
        if os.environ.get(task["env"].upper() + "_PYTHON") and task["env"] in ("alfworld", "appworld"):
            from .env_rpc import RpcSession
            self.session_cls = RpcSession

    def reset(self, kwargs=None):
        self.history, self.infos, self.text, self.done = [], [], [], []
        for i in range(self.config["rollouts_per_task"]):
            session = self.session_cls(self.task, {**self.config, "session_id": str(i)})
            self.sessions.append(session)
            obs, info = session.reset()
            self.text.append(obs)
            self.infos.append(info)
            self.history.append([])
            self.done.append(False)
        self.descriptions = [i["task_description"] for i in self.infos]
        if self.task["env"] == "appworld":
            self.supervisors = [i["supervisor"] for i in self.infos]
        return self.observations(), self.infos

    def observations(self):
        from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
        from agent_system.environments.prompts.appworld import APPWORLD_TEMPLATE, APPWORLD_TEMPLATE_NO_HIS
        prompts = []
        for i, obs in enumerate(self.text):
            recent = self.history[i][-self.config["history_length"]:]
            history = "\n".join(f"[Observation {j+1}: '{o}', Action {j+1}: '{a}']" for j, (o, a) in enumerate(recent))
            common = dict(task_description=self.descriptions[i], step_count=len(self.history[i]),
                current_step=len(self.history[i]) + 1, history_length=len(recent), current_observation=obs, action_history=history)
            if self.task["env"] == "alfworld":
                actions = "\n ".join(repr(a) for a in self.infos[i]["admissible_commands"] if a != "help")
                template = ALFWORLD_TEMPLATE if recent else ALFWORLD_TEMPLATE_NO_HIS
                prompt = template.format(**common, admissible_actions=actions)
            elif self.task["env"] == "appworld":
                sup = self.supervisors[i]
                code_history = "\n".join(f"Code {j+1}:\n{a}\nResult {j+1}:\n{o}" for j, (o,a) in enumerate(recent))
                common["action_history"] = code_history if len(code_history) <= 10000 else "... " + code_history[-10000:]
                template = APPWORLD_TEMPLATE if recent else APPWORLD_TEMPLATE_NO_HIS
                prompt = template.format(**common, **{"supervisor_" + k: v for k,v in sup.items()})
            else:
                # Only public current observation, task, inventory and action templates are actor-visible.
                # Native goal progress / object trees / gold paths never enter this prompt.
                prompt = ("You are an agent in ScienceWorld. Complete this task: " + self.descriptions[i]
                    + "\nRecent observation/action history:\n" + history + "\nCurrent observation:\n" + obs
                    + "\nCurrent room:\n" + self.infos[i].get("look", "") + "\nInventory:\n" + self.infos[i].get("inventory", "")
                    + "\nAvailable action templates:\n" + "\n".join(self.sessions[i].env.get_possible_actions())
                    + "\nReason briefly inside <think>...</think>, then give ONE environment command inside <action>...</action>.")
                profile = self.config.get("prompt_profile", "p0_original")
                if profile != "p0_original":
                    from .post_interface import public_context
                    block, metadata = public_context(self.sessions[i].env, profile)
                    self.infos[i]["interface_metadata"] = metadata
                    prompt += "\n" + block
            prompts.append(prompt)
        return {"text": prompts, "image": None, "anchor": list(self.text)}

    def step(self, responses):
        import numpy as np
        rewards, dones, infos = [], [], []
        for i, raw in enumerate(responses):
            if self.done[i]:
                rewards.append(0.0); dones.append(True); infos.append(self.infos[i]); continue
            action, valid = parse_action(raw, self.task["env"])
            before = self.text[i]
            obs, reward, done, info = self.sessions[i].step(action)
            self.history[i].append((obs if self.task["env"] == "appworld" else before, action))
            info.update(parsed_action=action, parse_valid=valid, is_action_valid=valid)
            self.text[i], self.infos[i], self.done[i] = obs, info, bool(done)
            rewards.append(reward); dones.append(done); infos.append(info)
        return self.observations(), np.asarray(rewards), np.asarray(dones), infos

    def success_evaluator(self, **kwargs):
        import numpy as np
        # Collector return only; research metrics retain unknown rather than this convenience value.
        return {"success_rate": np.asarray([i.get("strict_success") is True for i in self.infos], dtype=float)}

    def close(self):
        for session in self.sessions:
            session.close()
