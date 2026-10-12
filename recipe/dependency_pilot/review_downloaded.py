"""Audit downloaded frozen runs without modifying their original records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def facts_key(info):
    if "facts" not in info:
        return None
    return fingerprint(sorted(json.dumps(f, sort_keys=True) for f in info["facts"]))


def held(info):
    if "facts" not in info:
        return None
    return sorted(a["name"].strip() for f in info["facts"] if f["name"] == "holds"
                  for a in f["arguments"] if a["type"] == "object")


def target_clean(info, target):
    if target != "cloth" or "facts" not in info:
        return None
    # Explicit native predicate membership. Source semantic cross-check:
    # alfworld/agents/expert/handcoded_expert_tw.py, CleanObjectTask.get_predicates.
    return any(f["name"] == "isclean" and any(a["name"].strip().startswith(target + " ")
               for a in f["arguments"]) for f in info["facts"])


def has_target(info, target):
    values = held(info)
    return None if values is None else any(x.startswith(target + " ") for x in values)


def audit_run(directory, output):
    from recipe.dependency_pilot.analysis import metrics_for
    from recipe.dependency_pilot.diagnostics import recompute
    eps, rows, resets, decisions, diagnostics, anchors = [read_jsonl(directory / f) for f in
        ["episodes.jsonl", "steps.jsonl", "resets.jsonl", "decisions.jsonl", "diagnostics.jsonl", "anchors.jsonl"]]
    config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    tracking = json.loads((directory / "tracking.json").read_text(encoding="utf-8"))
    env = config["env"]
    index = {r["analysis_step_id"]: r for r in rows}
    assert len(index) == len(rows)
    assert set(index) == {r["analysis_step_id"] for r in decisions}
    assert len(eps) == len(resets) == config["max_total_episodes"]
    assert all(r["active_mask"] for r in rows)
    by_ep = {e["traj_uid"]: sorted([r for r in rows if r["traj_uid"] == e["traj_uid"]],
                                  key=lambda r: r["step_index"]) for e in eps}
    before, summaries, annotations, groups = {}, [], [], defaultdict(list)
    for ep_number, ep in enumerate(eps):
        seq = by_ep[ep["traj_uid"]]
        info = next(r["info"] for r in resets if r["traj_uid"] == ep["traj_uid"])
        assert [r["step_index"] for r in seq] == list(range(ep["agent_turns"]))
        assert len(seq) <= config["max_steps"]
        assert sum(r["generated_tokens"] for r in seq) == ep["generated_tokens"]
        assert sum(r["env_reward"] for r in seq) == ep["env_reward_sum"]
        assert sum(r["env_reward"] > 0 for r in seq) == int(ep["strict_success"])
        assert ep["env_reward_sum"] == (10.0 if ep["strict_success"] else 0.0)
        assert seq[-1]["native_info"]["strict_success"] == ep["strict_success"]
        target = "cloth" if ep["task_family"] == "pick_clean_then_place_in_recep" else "creditcard"
        summary = {k: ep[k] for k in ["traj_uid", "task_instance_id", "task_family", "strict_success",
                    "native_score", "agent_turns", "termination_reason", "episode_group_uid"]}
        summary.update(episode_number=ep_number + 1, task_description=info["task_description"],
            pickup_turns=[], place_turns=[], native_clean_turns=[], holding_sink_visit_turns=[],
            clean_menu_offered_turns=[], clean_command_turns=[], clean_fact_coverage=0,
            nothing_happens=sum(r["observation_post"] == "Nothing happens." for r in seq),
            unrecognized=sum(r["observation_post"] == "No known action matches that input." for r in seq),
            format_valid=sum(r["parse_valid"] for r in seq),
            native_action_valid=sum(r["env_action_valid"] is True for r in seq),
            action_block_present=sum(bool(re.search(r"<action>.*?</action>", r["raw_response"], re.I | re.S)) for r in seq))
        transcript = [f"# {env}: {ep['traj_uid']}", "", info["task_description"], "",
                      f"Strict success: {ep['strict_success']}; {ep['termination_reason']}", ""]
        for r in seq:
            step_id, turn = r["analysis_step_id"], r["step_index"] + 1
            before[step_id] = info
            assert len(r["response_token_ids"]) == r["generated_tokens"]
            assert len(r["actual_input_ids"]) == r["input_tokens"]
            assert hashlib.sha256(str(r["actual_input_ids"]).encode()).hexdigest() == r["actual_input_hash"]
            post = r["native_info"]
            annotation = dict(analysis_step_id=step_id, turn=turn, env=env,
                source_run=directory.name, source_file="steps.jsonl", rule_version="hpc_review_v1",
                annotation_kind="posthoc_observer_only", target_held_pre=None, target_held_post=None,
                target_isclean_pre=None, target_isclean_post=None, clean_menu_available_pre=None,
                wiring_condition_verified=None, source_reward_unchanged=True)
            if env == "alfworld":
                annotation.update(target_type=target, target_held_pre=has_target(info, target),
                    target_held_post=has_target(post, target), target_isclean_pre=target_clean(info, target),
                    target_isclean_post=target_clean(post, target),
                    clean_menu_available_pre=any(a.startswith("clean " + target + " ") for a in info["admissible_commands"]),
                    evidence="native facts membership; absent/nonapplicable facts remain null")
                if r["observation_post"].startswith("You pick up the " + target + " ") and annotation["target_held_post"]:
                    summary["pickup_turns"].append(turn)
                if r["observation_post"].startswith("You move the " + target + " "):
                    summary["place_turns"].append(turn)
                if annotation["target_isclean_post"] is True:
                    summary["native_clean_turns"].append(turn)
                summary["clean_fact_coverage"] += annotation["target_isclean_post"] is not None
                if annotation["clean_menu_available_pre"]:
                    summary["clean_menu_offered_turns"].append(turn)
                if r["parsed_action"].startswith("clean "):
                    summary["clean_command_turns"].append(dict(turn=turn, action=r["parsed_action"], feedback=r["observation_post"]))
                if (r["parsed_action"].startswith("go to sinkbasin ") and r["env_action_valid"]
                        and annotation["target_held_post"]):
                    summary["holding_sink_visit_turns"].append(turn)
                assert r["env_action_valid"] == (r["parsed_action"] in info["admissible_commands"])
            else:
                assert post["strict_success"] == (post["simulator_score"] >= 1 and post["native_completed"])
            annotations.append(annotation)
            transcript += [f"## Turn {turn} — `{step_id}`", "", f"Action: `{r['parsed_action']}`", "",
                           r["observation_post"], "", f"Score: {r['native_score_post']}; reward: {r['env_reward']}", ""]
            info = post
        summaries.append(summary)
        groups[ep["episode_group_uid"]].append(ep)
        (output / "cases" / f"{env}-{ep['traj_uid'][:8]}.md").write_text("\n".join(transcript), encoding="utf-8")
    assert all(len(group) == config["rollouts_per_task"] for group in groups.values())
    assert len(groups) == config["max_tasks"]
    recomputed, _, diagnostic_status = recompute(rows, eps, config)
    saved_d = {d["analysis_step_id"]: d for d in diagnostics}
    diag_max = max(abs(d[k] - saved_d[d["analysis_step_id"]][k]) for d in recomputed
                   for k in ["episode_adv_scalar", "step_adv_scalar", "joint_adv_scalar", "step_return_raw"])
    assert diag_max < 2e-6, diag_max
    rebuilt = metrics_for(eps, rows, diagnostics, anchors)
    for key, value in rebuilt.items():
        assert value == metrics[key], (key, value, metrics[key])
    audit, support = [], {}
    for anchor in anchors:
        seq = [index[i] for i in anchor["analysis_step_ids"]]
        compatible = defaultdict(list)
        for r in seq:
            state = facts_key(before[r["analysis_step_id"]])
            if state is not None:
                compatible[state].append(r)
        for r in seq:
            state = facts_key(before[r["analysis_step_id"]])
            support[r["analysis_step_id"]] = None if state is None else any(
                other["traj_uid"] != r["traj_uid"] and other["parse_valid"] and r["parse_valid"]
                and other["parsed_action"] != r["parsed_action"] for other in compatible[state])
        audit.append(dict(step_group_uid=anchor["step_group_uid"], episode_group_uid=anchor["episode_group_uid"],
            task_family=next(e["task_family"] for e in eps if e["episode_group_uid"] == anchor["episode_group_uid"]),
            anchor=seq[0]["anchor_pre_raw"], occurrences=len(seq),
            distinct_native_fact_sets=len(compatible) if compatible else None,
            held_objects_counts=dict(Counter(" + ".join(held(before[r["analysis_step_id"]]) or []) or "empty" for r in seq)) if env == "alfworld" else None,
            distinct_admissible_action_sets=len({tuple(sorted(before[r["analysis_step_id"]]["admissible_commands"])) for r in seq}) if env == "alfworld" else None,
            return_std=anchor["return_std"], analysis_step_ids=anchor["analysis_step_ids"]))
    parsed_after_first = [r for r in rows if r["parse_valid"] and r["step_index"] > 0]
    source_support = metrics["anchor/alternative_action_step_coverage_after_first"]
    fact_support = None if env != "alfworld" else dict(
        numerator=sum(support[r["analysis_step_id"]] is True for r in parsed_after_first),
        denominator=len(parsed_after_first), interpretation="same recorded native fact set, same anchor/group, another trajectory and action; not full simulator/RNG equivalence")
    families = []
    for family in dict.fromkeys(e["task_family"] for e in eps):
        family_eps = [e for e in eps if e["task_family"] == family]
        ids = {e["traj_uid"] for e in family_eps}
        family_rows = [r for r in rows if r["traj_uid"] in ids]
        family_diags = [d for d in diagnostics if d["traj_uid"] in ids]
        families.append(dict(family=family, episodes=len(family_eps), successes=sum(e["strict_success"] for e in family_eps),
            budget_terminated=sum(e["termination_reason"] == "budget_truncated" for e in family_eps),
            native_failure=sum(e["termination_reason"] == "task_failure" for e in family_eps),
            interactions=len(family_rows), nonzero_local_adv=sum(abs(d["step_adv_scalar"]) > 1e-6 for d in family_diags),
            env_action_valid=sum(r["env_action_valid"] is True for r in family_rows) if env == "alfworld" else None,
            unrecognized=sum(r["observation_post"] == "No known action matches that input." for r in family_rows)))
    return dict(env=env, run_id=directory.name, config={k:v for k,v in config.items() if k != "chat_template"},
        manifest=manifest, tracking=tracking, metrics=metrics, episodes=summaries, families=families,
        integrity=dict(episodes=len(eps), steps=len(rows), metrics_match=True, raw_id_token_reward_checks=True,
            diagnostic_max_absolute_difference=diag_max, diagnostic_reconstruction=diagnostic_status),
        anchor_audit=sorted(audit, key=lambda a:-a["occurrences"]),
        original_alternative_support_after_first=source_support, native_fact_matched_support_after_first=fact_support,
        format_detail=dict(chinese_response_steps=sum(bool(re.search(r"[\u4e00-\u9fff]",r["raw_response"])) for r in rows),
            action_block_present=sum(bool(re.search(r"<action>.*?</action>",r["raw_response"],re.I|re.S)) for r in rows),
            flagged_but_native_valid=sum(not r["parse_valid"] and r["env_action_valid"] is True for r in rows)),
        positive_local_adv_steps=sum(d["step_adv_scalar"] > 1e-6 for d in diagnostics),
        positive_local_adv_invalid_actions=sum(d["step_adv_scalar"] > 1e-6 and index[d["analysis_step_id"]]["env_action_valid"] is False for d in diagnostics)), annotations, rows


def figures(output, analyses, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False,
                        "font.size":11,"figure.facecolor":"white","axes.facecolor":"white","svg.fonttype":"path"})
    colors = ["#2a8c77", "#d6ad62", "#be6259"]
    fs = analyses["scienceworld"]["families"] + analyses["alfworld"]["families"]
    labels = ["ScienceWorld\n基础供电", "ScienceWorld\n非可再生供电", "ALFWorld\n简单放置", "ALFWorld\n清洗后放置"]
    fig, axs = plt.subplots(1,2,figsize=(12.4,5.0),gridspec_kw={"width_ratios":[1.35,1]})
    bottom = [0]*4
    for key,name,color in zip(["successes","budget_terminated","native_failure"],["完整成功","预算耗尽","环境判定失败"],colors):
        values=[f[key] for f in fs]
        axs[0].bar(range(4),values,bottom=bottom,color=color,label=name,width=.6)
        for i,value in enumerate(values):
            if value: axs[0].text(i,bottom[i]+value/2,str(value),ha="center",va="center",color="#202733",fontweight="bold")
        bottom=[a+b for a,b in zip(bottom,values)]
    axs[0].set(xticks=range(4),xticklabels=labels,yticks=range(5),ylim=(0,4.8),ylabel="轨迹数（每个实例 4 条）",title="A  四个固定实例的真实结果")
    axs[0].legend(loc="upper center",ncol=3,fontsize=9,frameon=False)
    clean=[e for e in analyses["alfworld"]["episodes"] if e["task_family"]=="pick_clean_then_place_in_recep"]
    stages=[sum(bool(e[k]) for e in clean) for k in ["pickup_turns","holding_sink_visit_turns","native_clean_turns"]]+[sum(e["strict_success"] for e in clean)]
    stage_labels=["成功拿起目标布","持布前往水槽","出现 isclean 事实","完整任务成功"]
    axs[1].barh(range(4),stages,color=["#527f9c"]+ ["#d8e2e8"]*3,height=.55)
    for i,count in enumerate(stages):axs[1].text(count+.08,i,f"{count}/4",va="center")
    axs[1].set(yticks=range(4),yticklabels=stage_labels,xlim=(0,4.8),xticks=range(5),title="B  清洗任务：进展停在取物之后")
    axs[1].invert_yaxis()
    fig.suptitle("P0 首批结果｜Qwen2.5-1.5B-Instruct，冻结采样",fontsize=16,fontweight="bold")
    fig.text(.03,.025,"16 条轨迹，601 次交互；每条最多 50 步。仅四个实例，不代表完整 benchmark 成绩。",fontsize=10,color="#55606d")
    fig.tight_layout(rect=(0,.07,1,.92))
    export(fig,output/"figures"/"p0_overview")

    af=analyses["alfworld"]
    fig,ax=plt.subplots(figsize=(13,5.5))
    palette={"invalid":"#e7b8b2","ordinary":"#d4dde4","pickup":"#4c81a5","place":"#936eae","success":"#28866c"}
    ylabels=[]
    for y,ep in enumerate(af["episodes"]):
        seq=[r for r in rows["alfworld"] if r["traj_uid"]==ep["traj_uid"]]
        for r in seq:
            t=r["step_index"]+1
            key="ordinary" if r["env_action_valid"] else "invalid"
            if t in ep["pickup_turns"]:key="pickup"
            if t in ep["place_turns"]:key="place"
            if r["native_info"]["strict_success"]:key="success"
            ax.broken_barh([(t-.46,.92)],(y-.34,.68),facecolors=palette[key])
        ylabels.append(("放置" if y<4 else "清洗")+f" {y%4+1} · {ep['traj_uid'][:8]}"+(" 成功" if ep["strict_success"] else ""))
    ax.axhline(3.5,color="#98a5b2",linewidth=.7)
    ax.set(yticks=range(8),yticklabels=ylabels,xlim=(.4,50.7),xticks=[1,5,10,15,20,25,30,35,40,45,50],xlabel="模型动作轮次（从 1 开始）",title="ALFWorld 动作轨迹：拿起与放回已经出现，清洗尚未出现")
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=palette[k],label=n) for k,n in [("ordinary","其他合法动作"),("invalid","无效动作 / Nothing happens"),("pickup","拿起目标物"),("place","放置目标物"),("success","任务成功")]],ncol=3,loc="upper center",bbox_to_anchor=(.5,-.15),frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1))
    export(fig,output/"figures"/"alfworld_action_timeline")

    groups=[next(a for a in af["anchor_audit"] if a["anchor"]=="Nothing happens." and a["task_family"]==family)
            for family in ["pick_and_place_simple","pick_clean_then_place_in_recep"]]
    fig,axs=plt.subplots(1,2,figsize=(11.6,4.7))
    inventory_labels={"empty":"空手","creditcard 2":"信用卡 2","creditcard 4":"信用卡 4","laptop 1":"笔记本 1","laptop 2":"笔记本 2","cloth 4":"布 4"}
    invcolors=["#aebbc6","#527f9c","#79a8be","#b693ba","#7f6894"]
    for ax,group,name in zip(axs,groups,["简单放置","清洗后放置"]):
        counts=group["held_objects_counts"]
        ax.bar(range(len(counts)),list(counts.values()),color=invcolors[:len(counts)],width=.58)
        for i,v in enumerate(counts.values()):ax.text(i,v+.8,str(v),ha="center")
        ax.set(xticks=range(len(counts)),xticklabels=[inventory_labels[k] for k in counts],ylabel="同一 anchor 组中的动作记录数",ylim=(0,max(counts.values())*1.2),title=f"{name}：{group['occurrences']} 条相同反馈，持有物不同")
        ax.text(.03,.93,f"{group['distinct_admissible_action_sets']} 种合法动作列表",transform=ax.transAxes,color="#52606b")
    fig.suptitle('相同 “Nothing happens.” 反馈不等于相同决策状态',fontsize=15,fontweight="bold")
    fig.text(.04,.025,"按动作前的原生 facts 核对；组内还存在位置、可执行动作和历史差异。此图不推断动作因果价值。",fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.92))
    export(fig,output/"figures"/"anchor_state_mixing")


def export(fig, path):
    import matplotlib.pyplot as plt
    for extension in ["png","svg"]:fig.savefig(str(path)+"."+extension,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--repo",required=True,type=Path)
    parser.add_argument("--plots-only", action="store_true")
    args=parser.parse_args()
    if args.plots_only:
        analyses=json.loads((args.output/"analysis.json").read_text(encoding="utf-8"))
        rows={env:read_jsonl(args.input/value["run_id"]/"steps.jsonl") for env,value in analyses.items()}
        figures(args.output,analyses,rows)
        return
    # Load Torch's runtime first in the local Windows/Anaconda environment.
    # NumPy-first import can initialize a second OpenMP runtime at plot time.
    import torch
    sys.path.insert(0,str(args.repo))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"figures").mkdir()
    (args.output/"cases").mkdir()
    receipt=json.loads((args.input/"import_manifest.json").read_text(encoding="utf-8"))
    for file,expected in receipt["file_sha256"].items():
        assert hashlib.sha256((args.input/file).read_bytes()).hexdigest()==expected,file
    analyses,annotations,rows={},{},{}
    for directory in sorted(args.input.glob("*-collect-*")):
        value,labels,trace=audit_run(directory,args.output)
        env=value["env"]
        analyses[env],annotations[env],rows[env]=value,labels,trace
    save(args.output/"analysis.json",analyses)
    (args.output/"condition_annotations.jsonl").write_text("".join(json.dumps(a,ensure_ascii=False)+"\n" for labels in annotations.values() for a in labels),encoding="utf-8")
    # Keep Anaconda's plotting libraries and Torch's OpenMP runtime in separate
    # processes instead of suppressing a duplicate-runtime error.
    subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--input",str(args.input),
                    "--output",str(args.output),"--repo",str(args.repo),"--plots-only"],check=True)
    clean=[e for e in analyses["alfworld"]["episodes"] if e["task_family"]=="pick_clean_then_place_in_recep"]
    prefixes=[]
    for ep in clean[:2]:
        turn=ep["pickup_turns"][0]
        prefixes.append(dict(source_run=analyses["alfworld"]["run_id"],source_traj_uid=ep["traj_uid"],
            after_step=turn-1,prefix_turns=turn,remaining_turns=50-turn,
            hint="Clean the cloth you are holding.",source_step_id=f"{ep['traj_uid']}:{turn-1}",
            selection="first two clean-task trajectories in manifest order; earliest verified pickup; not ranked by outcome"))
    save(args.output/"next_probe_manifest.json",dict(status="selected_not_run",prefixes=prefixes,
        conditions=["original_goal_continuation","original_goal_plus_local_hint"],repeats_per_condition=2,
        rollout_count=8,max_new_actions=sum(p["remaining_turns"] for p in prefixes)*4,
        response_tokens=512,prompt_tokens=4096,history_length=4,
        endpoints=["first native isclean for the held target cloth","strict task success at original 50-turn endpoint"],
        rules=["Same prefix, paired seeds, remaining turns and endpoints within each pair.",
               "Local hint is extra information, not an action sequence.",
               "Compare replayed facts, observations, admissible actions and history with source and between conditions.",
               "Original full simulator/RNG snapshots are absent: label reconstructed replay, not exact causal replay.",
               "Current probes.py supports only ScienceWorld; an ALFWorld replay adapter is still required."]))
    for file,expected in receipt["file_sha256"].items():
        assert hashlib.sha256((args.input/file).read_bytes()).hexdigest()==expected,file
    save(args.output/"verification.json",dict(raw_files_verified_unchanged=len(receipt["file_sha256"]),
        archive_sha256=receipt["archive_sha256"],runs={k:v["integrity"] for k,v in analyses.items()},
        model_updates=0,hpc_access="user-operated; local analysis of downloaded files"))
    print(json.dumps({k:{"integrity":v["integrity"],"families":v["families"],"fact_matched_support":v["native_fact_matched_support_after_first"],"format":v["format_detail"]} for k,v in analyses.items()},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
