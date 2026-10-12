import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .common import read_jsonl, write_json


def rate(numerator, denominator, scope):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None,
            "scope": scope, "status": "known" if denominator else "unavailable"}


def average(values, scope):
    values = [x for x in values if x is not None]
    return {"numerator": sum(values), "denominator": len(values),
            "value": mean(values) if values else None, "scope": scope,
            "status": "known" if values else "unavailable"}


def metrics_for(episodes, steps, diagnostics=(), anchors=()):
    if len({e["env"] for e in episodes}) > 1:
        raise ValueError("Report environments separately")
    known = [e for e in episodes if e["strict_success"] is not None and e["status"] != "infra_error"]
    task_results = defaultdict(list)
    for ep in known:
        task_results[ep["task_instance_id"]].append(int(ep["strict_success"]))
    active = [s for s in steps if s["active_mask"]]
    parse = [s for s in active if s.get("parse_valid") is not None]
    valid = [s for s in active if s.get("env_action_valid") is not None]
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["checkpoint_id"], ep["rollout_batch_id"], ep["episode_group_uid"])].append(ep)
    complete_groups = [es for es in groups.values() if len(es) == es[0]["expected_group_size"]
        and all(e["strict_success"] is not None and e["status"] == "complete" for e in es)]
    metrics = {
        "eval/success_macro": average([mean(values) for values in task_results.values()], "task-instance macro"),
        "eval/success_micro": rate(sum(e["strict_success"] for e in known), len(known), "evaluable episodes"),
        "eval/known_outcome_rate": rate(len(known), len(episodes), "attempted episodes"),
        "eval/native_score_mean": average([e["native_score"] for e in episodes], "episodes with native score; within environment"),
        "eval/budget_truncated_rate": rate(sum(e["termination_reason"] == "budget_truncated" for e in known), len(known), "evaluable episodes"),
        "data/parse_valid_rate": rate(sum(s["parse_valid"] for s in parse), len(parse), "active steps with known parse validity"),
        "data/env_action_valid_rate": rate(sum(s["env_action_valid"] for s in valid), len(valid), "active steps with native validity evidence"),
        "data/turns_mean": average([e["agent_turns"] for e in known], "evaluable episodes"),
        "data/prompt_truncation_rate": rate(sum(s["truncated_tokens"] > 0 for s in active), len(active), "active generated steps"),
        "data/output_token_limit_rate": rate(sum(s["output_at_token_limit"] for s in active), len(active), "active generated steps"),
        "data/env_exception_rate": rate(sum(e["status"] == "infra_error" for e in episodes), len(episodes), "attempted episodes"),
        "data/group_all_fail_rate": rate(sum(not any(e["strict_success"] for e in es) for es in complete_groups), len(complete_groups), "complete known-outcome groups"),
        "data/group_all_success_rate": rate(sum(all(e["strict_success"] for e in es) for es in complete_groups), len(complete_groups), "complete known-outcome groups"),
        "data/group_mixed_rate": rate(sum(0 < sum(e["strict_success"] for e in es) < len(es) for es in complete_groups), len(complete_groups), "complete known-outcome groups"),
        "data/group_known_rate": rate(len(complete_groups), len(groups), "all attempted groups"),
        "coverage/condition_known_rate": rate(0, 0, "condition specifications not yet annotated"),
        "phase/preparation_reached_rate": rate(0, 0, "no verified condition labels yet"),
        "phase/prepared_but_incomplete_rate": rate(0, 0, "no verified condition labels yet"),
        "cost/env_interactions_total": {"value": len(active), "scope": "model-issued environment actions", "status": "known"},
        "cost/generated_tokens_total": {"value": sum(e["generated_tokens"] for e in episodes), "scope": "active model output tokens", "status": "known"},
        "anchor/group_size_mean": average([a["occurrences"] for a in anchors], "unique anchor groups"),
        "anchor/unique_trajectories_mean": average([a["unique_trajectories"] for a in anchors], "unique anchor groups"),
        "anchor/unique_actions_mean": average([a["unique_actions"] for a in anchors], "unique anchor groups; parseable actions only"),
        "anchor/singleton_group_rate": rate(sum(a["occurrences"] == 1 for a in anchors), len(anchors), "unique anchor groups"),
        "anchor/single_trajectory_group_rate": rate(sum(a["unique_trajectories"] == 1 for a in anchors), len(anchors), "unique anchor groups"),
        "anchor/max_trajectory_share_mean": average([a["max_trajectory_share"] for a in anchors], "unique anchor groups"),
        "anchor/return_std_mean": average([a["return_std"] for a in anchors], "unique anchor groups; population std"),
        "adv/step_nonzero_rate": rate(sum(abs(d["step_adv_scalar"]) > 1e-6 for d in diagnostics), len(diagnostics), "active diagnostic steps"),
        "adv/episode_abs_mean": average([abs(d["episode_adv_scalar"]) for d in diagnostics], "active diagnostic steps"),
        "adv/step_abs_mean": average([abs(d["step_adv_scalar"]) for d in diagnostics], "active diagnostic steps"),
    }
    for suffix, subset in [("", diagnostics), ("_after_first", [d for d in diagnostics if d["step_index"] > 0])]:
        parsed = [d for d in subset if d["parse_valid"]]
        metrics["anchor/alternative_action_step_coverage" + suffix] = rate(sum(d["alternative_action_supported"] for d in parsed), len(parsed), "parseable active steps" + suffix)
    return metrics


def csv_table(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    episodes, steps = read_jsonl(run_dir / "episodes.jsonl"), read_jsonl(run_dir / "steps.jsonl")
    # Attempts lacking a final record are explicit interrupted/unknown executions.
    final_ids = {e["traj_uid"] for e in episodes}
    for ep in read_jsonl(run_dir / "attempts.jsonl"):
        if ep["traj_uid"] not in final_ids:
            ep.update(strict_success=None, status="infra_error", termination_reason="unknown", error_category="interrupted_before_final_record")
            episodes.append(ep)
    diagnostics, anchors, status = [], [], {"status": "unavailable", "reason": "no complete groups"}
    group_members = defaultdict(list)
    for ep in episodes:
        group_members[ep["episode_group_uid"]].append(ep)
    eligible = {gid for gid, es in group_members.items() if len(es) == es[0]["expected_group_size"] and all(e["status"] == "complete" and e["strict_success"] is not None for e in es)}
    if steps and eligible:
        try:
            from .diagnostics import recompute
            diagnostics, anchors, status = recompute([s for s in steps if s["episode_group_uid"] in eligible],
                [e for e in episodes if e["episode_group_uid"] in eligible], config)
        except Exception as exc:
            status = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    metrics = metrics_for(episodes, steps, diagnostics, anchors)
    if status.get("reconstruction_max_abs_error") is not None:
        metrics["adv/reconstruction_max_abs_error"] = {"value": status["reconstruction_max_abs_error"], "scope": "active diagnostic steps", "status": "known"}
    write_json(run_dir / "diagnostics_status.json", status)
    for filename, rows in [("diagnostics.jsonl", diagnostics), ("anchors.jsonl", anchors)]:
        # Derived files can be recomputed; raw episode/step records are never overwritten.
        (run_dir / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "metrics.jsonl").write_text(json.dumps({"analysis_batch_idx": 0, "metrics": metrics}) + "\n", encoding="utf-8")
    table_dir = run_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    csv_table(table_dir / "episodes.csv", episodes, ["task_instance_id", "traj_uid", "official_split", "strict_success", "native_score", "agent_turns", "termination_reason", "status"])
    csv_table(table_dir / "metrics.csv", [{"metric": k, **v} for k,v in metrics.items()], ["metric", "value", "numerator", "denominator", "scope", "status"])
    return episodes, steps, metrics, anchors


def plots(run_dir, episodes, steps, anchors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = Path(run_dir) / "figures"
    figures.mkdir(exist_ok=True)
    if not episodes:
        return []
    output = []
    labels = ["success", "budget", "failure", "unknown/infra"]
    families = sorted({e["task_family"] for e in episodes})
    counts = {family: [0] * len(labels) for family in families}
    for ep in episodes:
        idx = 0 if ep["strict_success"] is True else (3 if ep["strict_success"] is None or ep["status"] == "infra_error" else (1 if ep["termination_reason"] == "budget_truncated" else 2))
        counts[ep["task_family"]][idx] += 1
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = [0] * len(families)
    for index, label in enumerate(labels):
        values = [counts[f][index] for f in families]
        ax.bar(range(len(families)), values, bottom=bottom, label=label)
        bottom = [a+b for a,b in zip(bottom, values)]
    ax.set_xticks(range(len(families)), [f.replace("power-component", "power").replace("-", "\n") for f in families], fontsize=8)
    ax.set_ylabel("Episodes (counts)")
    ax.set_title("Frozen model outcomes: full completion and truncation kept separate")
    ax.legend(fontsize=8)
    fig.text(.02, .015, f"{len(episodes)} attempts; {len({e['task_instance_id'] for e in episodes})} task-instances; train/dev; condition labels unknown", fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ["png", "svg"]:
        path = figures / f"outcome_overview.{ext}"
        fig.savefig(path, dpi=170)
        if ext == "png": output.append(path)
    plt.close(fig)
    if steps:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        by_traj = defaultdict(list)
        for row in steps:
            if row.get("native_score_post") is not None:
                by_traj[row["traj_uid"]].append(row)
        for traj, rows in by_traj.items():
            ax.plot([r["step_index"] + 1 for r in rows], [r["native_score_post"] for r in rows], marker=".", linewidth=1, label=traj[:8])
        ax.set(xlabel="Model decision turn", ylabel="Native score", title="Raw score trajectories (score growth is not a verified preparation label)")
        if len(by_traj) <= 12: ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        for ext in ["png", "svg"]:
            path = figures / f"native_score_trajectories.{ext}"
            fig.savefig(path, dpi=170)
            if ext == "png": output.append(path)
        plt.close(fig)
    if anchors:
        fig, ax = plt.subplots(figsize=(7, 4))
        c = Counter(a["unique_trajectories"] for a in anchors)
        ax.bar(list(c), list(c.values()))
        ax.set(xlabel="Distinct trajectories per exact pre-action anchor group", ylabel="Groups", title="Anchor support; phase labels currently unknown")
        fig.tight_layout()
        for ext in ["png", "svg"]:
            path = figures / f"anchor_support.{ext}"
            fig.savefig(path, dpi=170)
            if ext == "png": output.append(path)
        plt.close(fig)
    return output


def report(run_dir, episodes, steps, metrics, figures):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    config = json.loads((run_dir / "resolved_config.json").read_text())
    tracking = json.loads((run_dir / "tracking.json").read_text()) if (run_dir / "tracking.json").exists() else {"mode": "disabled", "url": None}
    def show(key):
        m = metrics[key]
        return f"{m['value']:.3f} ({m.get('numerator', 'NA')}/{m.get('denominator','NA')})" if m["value"] is not None else "NA (denominator=0)"
    lines = ["# Dependency / preparation pilot — first bounded run", "",
        f"Status: **{manifest['status']}**. Identity: **frozen_collect**, no policy update.", "",
        f"Environment: {config['env']}; model: {config['model_path']} @ {config.get('model_revision')}; seed: {config['seed']}.",
        f"Actual: {len(episodes)} attempted episodes, {len(steps)} recorded model/environment steps. Budget: {config['max_steps']} turns, {config['response_tokens']} response tokens, group size {config['rollouts_per_task']}.", "",
        "## Observations and limits", "",
        f"1. Full completion (episode micro): {show('eval/success_micro')}; task macro: {show('eval/success_macro')}; known outcome coverage: {show('eval/known_outcome_rate')}.",
        f"2. Parse validity: {show('data/parse_valid_rate')}; budget truncation: {show('eval/budget_truncated_rate')}. Format, budget, and base capability remain competing explanations.",
        f"3. Nonzero local advantage: {show('adv/step_nonzero_rate')}; cross-trajectory alternative-action coverage after the first turn: {show('anchor/alternative_action_step_coverage_after_first')}.", "",
        "Preparation/join/use labels are unknown unless separately verified. Raw score increases are descriptive events only. These frozen trajectories do not compare GiGPO and GRPO behavior or establish H1/H2/H3. No prefix or training result is claimed.", "",
        "## Cases linked to raw records", ""]
    for ep in episodes[:6]:
        rows = [s for s in steps if s["traj_uid"] == ep["traj_uid"]]
        lines += [f"- `{ep['task_instance_id']}` / `{ep['traj_uid']}`: {ep['termination_reason']}, native score={ep['native_score']}, strict_success={ep['strict_success']}; {ep['agent_turns']} turns."]
        chosen = [r for r in rows if r["event_tags"]][:2] or rows[-2:]
        for row in chosen:
            lines += [f"  - `steps.jsonl` → `{row['analysis_step_id']}`: action `{row['parsed_action']}`; observation: {str(row['observation_post'])[:280].replace(chr(10), ' ')}"]
    if not episodes:
        lines += ["No model trajectory completed. See execution.log and errors.jsonl; no synthetic results are substituted."]
    lines += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figures]
    lines += ["", "## Next two actions", "",
        "1. Run the same bounded profile on Orion, or expand a truncated local connectivity run to the recorded 50-turn baseline; review 5–10 raw trajectories for task-specific preparation evidence.",
        "2. Once genuine preparation prefixes exist, compare original-goal continuation and one local-goal hint at identical replay state, remaining turns, token budget and evaluation endpoint. If none exist, diagnose basic operation/format first.", "",
        "## Reproduction", "", f"Branch/commit: `{manifest.get('git')}`", f"Command: `{manifest.get('command')}`",
        f"Output: `{run_dir}`; resolved config: `resolved_config.json` / `.yaml`; installed versions: `run_manifest.json`.",
        f"SwanLab: {tracking.get('mode')}; URL: {tracking.get('url') or 'not available'}. Raw prompts/responses remain local."]
    if manifest.get("error"):
        lines += ["", f"Infrastructure error: `{manifest['error']}`."]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
