"""Minimal P0 CLI. All results are local first; cloud is optional."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

from .common import ROOT, append_jsonl, digest, git_info, read_jsonl, versions, write_json


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=["inspect", "prepare", "collect", "analyze", "plot", "report", "track", "reference-check", "interface-check"])
    p.add_argument("--env", choices=["scienceworld", "alfworld", "appworld"], default="scienceworld")
    p.add_argument("--profile", choices=["smoke", "discovery"], default="smoke")
    p.add_argument("--output-root", default=os.environ.get("PILOT_OUTPUT_ROOT", str(ROOT / "outputs/dependency_pilot")))
    p.add_argument("--run-dir")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--model-revision")
    p.add_argument("--seed", type=int)
    for key in ["max-tasks", "rollouts-per-task", "max-total-episodes", "max-steps", "response-tokens", "prompt-tokens", "history-length"]:
        p.add_argument("--" + key, type=int)
    p.add_argument("--device", choices=["auto", "cpu"])
    p.add_argument("--gpu-memory-gib", type=int)
    p.add_argument("--swanlab-mode", choices=["auto", "cloud", "offline", "disabled"])
    return p


def resolve(args):
    import yaml
    config = yaml.safe_load((Path(__file__).parent / "configs/base.yaml").read_text(encoding="utf-8"))
    config["env"], config["profile"] = args.env, args.profile
    if args.profile == "smoke":
        config.update(max_tasks=2, max_total_episodes=8)
    if args.env == "appworld":
        config.update(max_tasks=4 if args.profile == "discovery" else 2, rollouts_per_task=2, max_steps=30, response_tokens=1024, prompt_tokens=8192)
    if args.env == "alfworld" and args.profile == "discovery":
        config.update(max_tasks=12, max_total_episodes=48)
    for key, value in vars(args).items():
        if key in config and value is not None:
            config[key] = value
    for key in ["max_tasks", "max_total_episodes", "rollouts_per_task", "max_steps", "prompt_tokens", "response_tokens", "history_length"]:
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["max_total_episodes"] < config["rollouts_per_task"]:
        raise ValueError("Episode budget must permit a complete group")
    model_manifest = Path(config["model_path"]) / "download_manifest.json"
    if model_manifest.exists():
        config["model_revision"] = json.loads(model_manifest.read_text(encoding="utf-8"))["revision"]
    config["checkpoint_id"] = config["model_path"] + "@" + str(config["model_revision"])
    if any(getattr(args, key) is not None for key in ["max_steps", "response_tokens", "prompt_tokens", "rollouts_per_task"]):
        config["protocol"] = "frozen_p0_v1_explicit_budget_override"
    config["seed_rule"] = "torch seed = base_seed + rollout_index + 100003 * decision_turn; task start fixed"
    return config


def save_config(run_dir, config):
    import yaml
    write_json(run_dir / "resolved_config.json", config)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def inspect():
    import importlib.util
    result = {"repository": str(ROOT), "git": git_info(), "python": sys.version, "platform": platform.platform(),
              "versions": versions(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "hpc_connected": False, "credential_presence": {name: bool(os.environ.get(name)) for name in ["HF_TOKEN", "SWANLAB_API_KEY"]}}
    if importlib.util.find_spec("torch"):
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["gpus"] = [{"name": torch.cuda.get_device_name(i), "memory_gib": torch.cuda.get_device_properties(i).total_memory / 2**30} for i in range(torch.cuda.device_count())]
    print(json.dumps(result, indent=2))


def prepare(run_dir, config):
    from .environments import enumerate_tasks
    selected, all_tasks, metadata = enumerate_tasks(config["env"], config)
    for filename, rows in [("tasks_manifest.jsonl", selected), ("task_inventory.jsonl", all_tasks)]:
        for row in rows:
            append_jsonl(run_dir / filename, row)
    write_json(run_dir / "environment_provenance.json", metadata)
    config["manifest_hash"] = digest(json.dumps(selected, sort_keys=True))
    save_config(run_dir, config)
    return selected


def collect(run_dir, config, tasks):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from .environments import PilotManager
    from .instrumentation import Observer
    from .runtime import TransformersWorker
    actor = TransformersWorker(config)
    save_config(run_dir, config)
    n = config["rollouts_per_task"]
    for batch_idx, task in enumerate(tasks[:config["max_total_episodes"] // n]):
        manager = PilotManager(task, config)
        observer = Observer(run_dir, config, task, actor.tokenizer, f"{config['run_id']}:batch{batch_idx}")
        actor.observer = observer
        upstream_config = OmegaConf.create({"data": {"max_prompt_length": config["prompt_tokens"], "truncation": "left", "return_raw_chat": True},
            "env": {"rollout": {"n": n}, "max_steps": config["max_steps"]}})
        collector = TrajectoryCollector(upstream_config, actor.tokenizer)
        collector.pilot_observer = observer
        raw_prompts = np.empty(n, dtype=object)
        for i in range(n): raw_prompts[i] = [{"role": "user", "content": ""}]
        batch = DataProto.from_single_dict({"input_ids": torch.zeros((n, 1), dtype=torch.long),
            "raw_prompt": raw_prompts, "data_source": np.array([config["env"]] * n, dtype=object)})
        try:
            collector.vanilla_multi_turn_loop(batch, actor, manager)
        except Exception as exc:
            observer.finish(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            manager.close()
        print(f"COMPLETED_GROUP {batch_idx + 1}/{len(tasks)} {task['task_instance_id']}", flush=True)
        # A report survives later group failures.
        summarize(run_dir)


def reference_check(run_dir, config, tasks):
    """Official reference actions are interface checks, never model-sampled evidence."""
    if config["env"] != "scienceworld":
        raise ValueError("reference-check currently supports ScienceWorld; use collect for ALFWorld/AppWorld")
    from scienceworld import ScienceWorldEnv
    for task in tasks:
        env = ScienceWorldEnv(envStepLimit=100)
        try:
            env.load(task["task_id"], task["variation_id"], config["simplifications"], generateGoldPath=True)
            actions = env.get_gold_action_sequence()
            if not actions or actions[0].startswith("ERROR"):
                raise RuntimeError("Official gold path unavailable for this installed task variation")
            obs, info = env.reset()
            append_jsonl(run_dir / "reference_resets.jsonl", {"task": task, "observation": obs, "info": info, "data_kind": "official_reference_execution"})
            for idx, action in enumerate(actions[:100]):
                obs, reward, done, info = env.step(action)
                row = {"task_instance_id": task["task_instance_id"], "step_index": idx, "action": action,
                    "observation": obs, "native_reward": reward, "native_score": info["score"], "done_raw": done,
                    "simulator_score": float(env.server.getScore()), "native_completed": bool(env.server.getCompleted()),
                    "data_kind": "official_reference_execution", "model_generated": False}
                append_jsonl(run_dir / "reference_steps.jsonl", row)
                if done: break
            print(f"REFERENCE {task['task_instance_id']} score={info['score']} actions={idx + 1}", flush=True)
        finally:
            env.close()


def interface_check(run_dir, config, tasks):
    if config["env"] != "appworld":
        raise ValueError("interface-check is the early AppWorld interaction check")
    from .environments import AppSession
    from .env_rpc import RpcSession
    cls = RpcSession if os.environ.get("APPWORLD_PYTHON") else AppSession
    actions = ["print(apis.api_docs.show_app_descriptions())", "print(apis.api_docs.show_api_descriptions(app_name='supervisor'))"]
    for i, task in enumerate(tasks[:2]):
        session = cls(task, {**config, "session_id": f"interface{i}"})
        try:
            obs, info = session.reset()
            append_jsonl(run_dir / "interface_steps.jsonl", {"task": task, "step_index": -1, "observation": obs, "info": info, "model_generated": False})
            for step, action in enumerate(actions):
                obs, reward, done, info = session.step(action)
                append_jsonl(run_dir / "interface_steps.jsonl", {"task_instance_id": task["task_instance_id"], "step_index": step,
                    "action": action, "observation": obs, "info": info, "model_generated": False})
        finally:
            session.close()


def summarize(run_dir, tracking=None):
    from .analysis import analyze, plots, report
    episodes, steps, metrics, anchors = analyze(run_dir)
    figures = plots(run_dir, episodes, steps, anchors)
    if tracking:
        tracking.log_summary(metrics, figures)
    report(run_dir, episodes, steps, metrics, figures)


def main():
    args = parser().parse_args()
    if args.command == "inspect":
        inspect(); return 0
    if args.command in ["analyze", "plot", "report", "track"]:
        if not args.run_dir: raise ValueError("--run-dir is required")
        run_dir = Path(args.run_dir)
        tracker = None
        if args.command == "track":
            # Match the working collection import order on Windows CUDA wheels.
            import torch
            from .runtime import PilotTracking
            from .common import file_hash
            tracking_config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
            tracking_config["parent_run_id"] = tracking_config["run_id"]
            tracking_config["run_id"] += "-analysis-" + uuid.uuid4().hex[:6]
            tracking_config["swanlab_mode"] = args.swanlab_mode or "auto"
            tracking_config["source_steps_sha256"] = file_hash(run_dir / "steps.jsonl")
            previous = run_dir / "tracking.json"
            if previous.exists():
                previous.rename(run_dir / ("tracking-previous-" + uuid.uuid4().hex[:6] + ".json"))
            tracker = PilotTracking(run_dir, tracking_config)
        try:
            if tracker:
                # Upload durable derived files; do not recompute plots after a cloud SDK
                # has loaded its own native libraries into this Windows process.
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                tracker.log_summary(metrics, sorted((run_dir / "figures").glob("*.png")))
            else:
                summarize(run_dir)
        finally:
            if tracker: tracker.close()
        if tracker:
            from .analysis import report
            report(run_dir, read_jsonl(run_dir / "episodes.jsonl"), read_jsonl(run_dir / "steps.jsonl"),
                metrics, sorted((run_dir / "figures").glob("*.png")))
        return 0
    config = resolve(args)
    run_id = f"{args.env}-{args.command}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.output_root) / run_id
    # Every invocation gets an immutable raw-output directory, including preparation.
    run_dir.mkdir(parents=True, exist_ok=False)
    config["run_id"] = run_id
    config["run_dir"] = str(run_dir.resolve())
    save_config(run_dir, config)
    manifest = {"schema_version": 1, "run_id": run_id, "git": git_info(), "base_commit": "24305ac85b5c7456ed2c0f851b12d70bda45823d",
        "command": " ".join(sys.argv), "versions": versions(), "status": "running", "platform": platform.platform(),
        "execution_kind": "official_reference_execution" if args.command == "reference-check" else args.command,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"RUN_DIR={run_dir}", flush=True)
    tracking = None
    try:
        tasks = prepare(run_dir, config)
        if args.command == "collect":
            collect(run_dir, config, tasks)
            manifest["status"] = "partial_useful"
        elif args.command == "reference-check":
            reference_check(run_dir, config, tasks)
            manifest["status"] = "interface_checked_not_model_sampling"
        elif args.command == "interface-check":
            interface_check(run_dir, config, tasks)
            manifest["status"] = "interface_checked_not_model_sampling"
        else:
            manifest["status"] = "prepared_not_sampled"
    except Exception as exc:
        manifest.update(status="blocked_infra", error=f"{type(exc).__name__}: {exc}")
        append_jsonl(run_dir / "errors.jsonl", {"error": manifest["error"], "traceback": traceback.format_exc()})
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        (run_dir / "execution.log").write_text("Command: " + manifest["command"] + "\nStatus: " + manifest["status"] + "\n" + str(manifest.get("error", "")), encoding="utf-8")
        write_json(run_dir / "run_manifest.json", manifest)
        if args.command == "collect":
            from .runtime import PilotTracking
            # Only summary scalars/images after durable local raw recording. Never credentials or raw traces.
            summarize(run_dir)
            tracking = PilotTracking(run_dir, config)
            try:
                tracking.log_summary(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
                    sorted((run_dir / "figures").glob("*.png")))
            finally:
                tracking.close()
            from .analysis import report
            report(run_dir, read_jsonl(run_dir / "episodes.jsonl"), read_jsonl(run_dir / "steps.jsonl"),
                json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")), sorted((run_dir / "figures").glob("*.png")))
        else:
            (run_dir / "report.md").write_text(f"# {args.command}\n\nStatus: {manifest['status']}\n\nThis run contains no model-sampled trajectories.\n\n{manifest.get('error', '')}\n", encoding="utf-8")
    print(f"RESULT={manifest['status']} RUN_DIR={run_dir}", flush=True)
    return 1 if manifest["status"] == "blocked_infra" else 0


if __name__ == "__main__":
    raise SystemExit(main())
