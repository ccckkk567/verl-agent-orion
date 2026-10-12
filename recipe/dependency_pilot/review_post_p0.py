"""Reproduce the downloaded 159709/159710 audit without modifying raw results.

Analysis and plotting are separate commands to avoid Torch/Matplotlib OpenMP
runtime collisions on Windows. No environment/model rollout is performed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import digest, file_hash, read_jsonl, write_json


def grouped(rows, key='traj_uid'):
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def audit(args):
    from transformers import AutoTokenizer
    from .environments import parse_action
    from .post_p0 import canonical_state, HINTS
    from .post_analysis import analyze_run, cloth_state, write_rows, table
    from .post_audit import compare

    raw, old, out = args.raw, args.old, args.output
    if out.resolve() == raw.resolve() or raw.resolve() in out.resolve().parents:
        raise ValueError('Analysis must be outside the imported raw directory')
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((raw/'import_manifest.json').read_text(encoding='utf8'))
    checks = {}

    def check(name, values):
        values = list(values)
        checks[name] = {'passed': sum(bool(v) for v in values), 'total': len(values)}
        write_json(out/'audit_checks.json', checks)
        if not values or not all(values):
            raise AssertionError(f'{name}: {checks[name]}')

    check('imported_file_sha256', (file_hash(raw/k) == v for k, v in manifest['files'].items()))
    af = raw/'alfworld-p1-159709'
    sw = raw/'scienceworld-rep0-159710'
    oldaf = old/'alfworld-collect-20261012T095303Z-d56e91'
    oldsw = old/'scienceworld-collect-20261012T094001Z-8ecf63'
    smoke = raw/'scienceworld-interface-smoke-159710'
    prefixes = read_jsonl(af/'prefix_manifest.jsonl')
    original = {s['analysis_step_id']: s for s in read_jsonl(oldaf/'steps.jsonl')}
    steps = read_jsonl(af/'steps.jsonl')
    eps = read_jsonl(af/'episodes.jsonl')
    byuid = grouped(steps)
    epmap = {e['traj_uid']: e for e in eps}
    byprefix = {p['prefix_id']: p for p in prefixes}
    check('source_alf_steps_and_config_hash', (p['source_steps_sha256'] == file_hash(oldaf/'steps.jsonl')
          and p['source_config_sha256'] == file_hash(oldaf/'resolved_config.json') for p in prefixes))
    check('alf_24_episodes_930_new_steps', [len(eps) == 24 and len(steps) == 930])
    replay = read_jsonl(af/'replay_checks.jsonl')
    check('replay_270_rows_excluded_from_rollouts', [len(replay) == 270 and all(s['record_kind'] == 'continuation' for s in steps)])
    check('replay_actual_native_state_vs_source', (
        canonical_state(r['observation'], r['info']) == canonical_state(original[r['source_step_id']]['observation_post'], original[r['source_step_id']]['native_info'])
        and r['parsed_action'] == original[r['source_step_id']]['parsed_action'] and not r['done'] for r in replay))
    check('groups_never_cross_prefix_or_condition', (len(g) == 2 and len({(e['prefix_id'], e['condition']) for e in g}) == 1 for g in grouped(eps, 'episode_group_uid').values()))
    check('prefix_condition_pair_seeds', (sorted(e['continuation_seed'] for e in eps if e['prefix_id'] == pid and e['condition'] == arm) == [10000,10001] for pid in byprefix for arm in HINTS))
    check('prefix_and_suffix_budget_50', (e['prefix_steps_consumed'] + e['remaining_turns'] == 50 and e['agent_turns'] == e['remaining_turns'] == len(byuid[e['traj_uid']]) for e in eps))
    check('suffix_contiguous_steps_and_absolute_seed', (s['absolute_step_index'] == epmap[s['traj_uid']]['prefix_steps_consumed'] + s['step_index'] and s['generation_seed'] == s['continuation_seed'] + 100003*s['absolute_step_index'] for s in steps))
    first = [ss[0] for ss in byuid.values()]
    check('first_prompt_before_hint_identical_to_old_next_decision', (
        s['actor_prompt_before_hint'] == original[byprefix[s['prefix_id']]['source_traj_uid']+':'+str(byprefix[s['prefix_id']]['prefix_steps_consumed'])]['actor_prompt'] for s in first))
    check('first_native_state_matches_recorded_pickup', (
        canonical_state(s['anchor_pre_raw'],s['native_info_pre']) == canonical_state(original[byprefix[s['prefix_id']]['source_traj_uid']+':'+str(byprefix[s['prefix_id']]['prefix_end_step_index'])]['observation_post'],original[byprefix[s['prefix_id']]['source_traj_uid']+':'+str(byprefix[s['prefix_id']]['prefix_end_step_index'])]['native_info']) for s in first))

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    swsteps = read_jsonl(sw/'steps.jsonl')
    swold = read_jsonl(oldsw/'steps.jsonl')
    smokesteps = read_jsonl(smoke/'steps.jsonl')
    check('new_sw_8_episodes_385_steps', [len(read_jsonl(sw/'episodes.jsonl')) == 8 and len(swsteps) == 385])
    check('token_ids_reencode_rendered_prompt', (tokenizer.encode(s['rendered_prompt'],add_special_tokens=False) == s['actual_input_ids'] for s in steps+swsteps+smokesteps))
    check('hint_visible_only_first_decision', (
        (HINTS[s['condition']] in tokenizer.decode(s['actual_input_ids'],skip_special_tokens=False)) == (s['step_index'] == 0)
        for s in steps if HINTS[s['condition']]))
    check('no_hint_in_original_condition', (all(h not in s['actor_prompt'] for h in HINTS.values() if h) for s in steps if s['condition']=='original'))
    check('unchanged_execution_parser_reproduces_all_actions', (parse_action(s['raw_response'],s['env'])[0] == s['parsed_action'] for s in steps+swsteps+swold))
    check('science_interface_fully_visible_no_omitted_objects', (s['interface_metadata']['actor_interface_block'] in tokenizer.decode(s['actual_input_ids'],skip_special_tokens=False) and s['interface_metadata']['object_refs_omitted_count']==0 for s in swsteps+smokesteps))
    check('science_observer_has_edges_and_costs_no_moves', (s['native_info']['electrical_observer']['edges'] is not None and s['native_info']['electrical_observer']['moves_after_read'] == s['native_info']['env_moves'] for s in swsteps+smokesteps))
    oldresets = read_jsonl(oldsw/'resets.jsonl')
    oldeps = {e['traj_uid']:e for e in read_jsonl(oldsw/'episodes.jsonl')}
    neweps = {e['traj_uid']:e for e in read_jsonl(sw/'episodes.jsonl')}
    # Resets identify trajectories; task identity lives in episodes, not reset rows.
    oldstarts = {oldeps[r['traj_uid']]['task_instance_id']:r for r in oldresets}
    newresets = read_jsonl(sw/'resets.jsonl')
    check('science_same_original_reset_observation', (r['observation'] == oldstarts[neweps[r['traj_uid']]['task_instance_id']]['observation'] for r in newresets))
    check('science_same_original_reset_native_core', (all(r['info'].get(k) == oldstarts[neweps[r['traj_uid']]['task_instance_id']]['info'].get(k) for k in ['native_score','task_description','look','inventory','env_moves']) for r in newresets))
    check('science_reset_first_rollout_no_prefix', (ss[0]['step_index']==0 and ss[0]['absolute_step_index']==0 and ss[0]['visible_history_steps']==0 and ss[0]['record_kind']=='rollout' for ss in grouped(swsteps).values()))

    afanalysis = analyze_run(af,out/'alfworld')
    analyze_run(smoke,out/'smoke')
    compare(oldsw,sw,out/'science-comparison')
    check('science_protocol_only_interface_changed', [not json.loads((out/'science-comparison/comparison_config_diff.json').read_text())['other_changes']])
    changes = []
    events = read_jsonl(out/'alfworld/event_annotations.jsonl')
    eventmap = {r['analysis_step_id']:r for r in events}
    for row in steps:
        event = eventmap[row['analysis_step_id']]
        if row['parser_usable'] != event['parser_usable']:
            changes.append({'analysis_step_id':row['analysis_step_id'],'old_parser_usable':row['parser_usable'],'corrected_parser_usable':event['parser_usable'],'old_parser_fallback_used':row['parser_fallback_used'],'corrected_parser_fallback_used':event['parser_fallback_used'],'parsed_action_unchanged':row['parsed_action']})
    write_rows(out/'diagnostic_corrections.jsonl',changes)
    diagmatches = {}
    for name,source,derived in [('alfworld',af,out/'alfworld'),('scienceworld',sw,out/'science-comparison/new_reanalysis')]:
        before = {r['analysis_step_id']:r for r in read_jsonl(source/'diagnostics.jsonl')}
        after = read_jsonl(derived/'diagnostics.jsonl')
        check(name+'_diagnostic_step_ids', [set(before)=={r['analysis_step_id'] for r in after}])
        err = max(abs(r[k]-before[r['analysis_step_id']][k]) for r in after for k in ['episode_adv_scalar','step_adv_scalar','joint_adv_scalar','step_return_raw'])
        diagmatches[name]={'steps':len(after),'max_abs_error_vs_hpc':err,'all_advantages_zero':all(r[k]==0 for r in after for k in ['episode_adv_scalar','step_adv_scalar','joint_adv_scalar'])}
        check(name+'_advantages_match_hpc', [err==0])

    arms = {}
    outcomes = {e['traj_uid']:e for e in afanalysis['episodes']}
    for arm in HINTS:
        armep = [e for e in eps if e['condition']==arm]
        armfirst = [byuid[e['traj_uid']][0] for e in armep]
        armsteps = [s for s in steps if s['condition']==arm]
        arms[arm] = {'episodes':len(armep),'steps':len(armsteps),
            'first_direct_target_placement':sum(eventmap[s['analysis_step_id']]['placement_before_clean'] is True for s in armfirst),
            'first_actions':dict(Counter(s['parsed_action'] for s in armfirst)),
            'sink_arrival_any_inventory':sum(any(cloth_state(s['native_info'])['at_sink'] for s in byuid[e['traj_uid']]) for e in armep),
            'clean_attempt':sum(outcomes[e['traj_uid']]['clean_attempt'] is True for e in armep),
            **{k:sum(outcomes[e['traj_uid']][k] is True for e in armep) for k in ['sink_reached','clean_success','placement_before_clean','strict_success']}}
    pairs={}
    for arm in ['goal_reminder','clean_subgoal']:
        values=[]
        for pid in byprefix:
            for seed in [10000,10001]:
                selected={e['condition']:e for e in eps if e['prefix_id']==pid and e['continuation_seed']==seed}
                original_rows=byuid[selected['original']['traj_uid']];armrows=byuid[selected[arm]['traj_uid']]
                values.append({'prefix_id':pid,'seed':seed,'identical_action_sequence':[s['parsed_action'] for s in original_rows]==[s['parsed_action'] for s in armrows]})
        pairs[arm]=values

    science_events=read_jsonl(out/'science-comparison/new_reanalysis/event_annotations.jsonl')
    taskdetail={}
    for task,ss in grouped(swsteps,'task_instance_id').items():
        ids={s['analysis_step_id'] for s in ss}
        ev=[e for e in science_events if e['analysis_step_id'] in ids]
        attempts=[s for s in ss if s['parsed_action'].lower().startswith('connect')]
        taskdetail[task]={'steps':len(ss),'connect_attempts':len(attempts),'connect_feedback':dict(Counter(s['observation_post'] for s in attempts)),
            'connect_outcomes':dict(Counter(e['execution_outcome'] for e in ev if e['connect_attempt'])),
            'score_changes':[{'analysis_step_id':s['analysis_step_id'],'turn':s['step_index']+1,'action':s['parsed_action'],'before':s['native_score_pre'],'after':s['native_score_post'],'feedback':s['observation_post']} for s in ss if s['native_score_pre']!=s['native_score_post']]}
    cases=[]
    caseuids=['40a66edd-188a-404c-b1b1-554fa56dd84d','2d86b411-8165-4999-83ba-c76837d55b14','312d8432-1227-4d06-8406-1cd7b890af1e']
    for uid in caseuids:
        chosen=[byuid[uid][0]] if uid!=caseuids[-1] else [s for s in byuid[uid] if s['step_index']<=21 and (eventmap[s['analysis_step_id']]['placement_before_clean'] or cloth_state(s['native_info'])['at_sink'])]
        for s in chosen:
            cases.append({'case':'alf_clean_or_empty_hand_sink','analysis_step_id':s['analysis_step_id'],'condition':s['condition'],'prefix_id':s['prefix_id'],'turn':s['step_index']+1,'action':s['parsed_action'],'response':s['raw_response'],'feedback':s['observation_post'],'pre_cloth_state':{k:sorted(v) if isinstance(v,set) else v for k,v in cloth_state(s['native_info_pre']).items()},'post_cloth_state':{k:sorted(v) if isinstance(v,set) else v for k,v in cloth_state(s['native_info']).items()},'command_admissible':s['parsed_action'] in s['native_info_pre']['admissible_commands']})
    endpoint=next(s for s in swsteps if 'Connections must specify' in s['observation_post'])
    late=next(s for s in swsteps if s['traj_uid']=='e27917ba-bead-465f-bf69-029e7052e646' and s['step_index']==49)
    for s in [endpoint,late]:
        cases.append({'case':'science_endpoint_or_late_workshop','analysis_step_id':s['analysis_step_id'],'turn':s['step_index']+1,'action':s['parsed_action'],'response':s['raw_response'],'feedback':s['observation_post'],'native_score_pre':s['native_score_pre'],'native_score_post':s['native_score_post'],'edges_pre':s['native_info_pre']['electrical_observer']['edges'],'edges_post':s['native_info']['electrical_observer']['edges']})
    write_json(out/'cases.json',cases)
    swanalysis=json.loads((out/'science-comparison/new_reanalysis/analysis.json').read_text())
    oldanalysis=json.loads((out/'science-comparison/old_reanalysis/analysis.json').read_text())
    summary={'source_archive_sha256':manifest['archive_sha256'],'analysis_only_no_new_rollouts':True,
        'audit_checks':checks,'alf_arms':arms,'paired_trajectories':pairs,'science_task_detail':taskdetail,
        'alf_overall':afanalysis['overall'],'science_old_overall':oldanalysis['overall'],'science_new_overall':swanalysis['overall'],
        'science_old_tasks':oldanalysis['subsets'],'science_new_tasks':swanalysis['subsets'],
        'diagnostic_correction':{'affected_alf_steps':len(changes),'old_usable_count':sum(s['parser_usable'] for s in steps),'corrected_usable_count':sum(e['parser_usable'] for e in events),'total_steps':len(steps),'execution_parser_changed':False,'hpc_swanlab_metrics_overwritten':False},
        'giGPO_recompute':diagmatches,'new_science_all_observed_edges_empty':all(s['native_info_pre']['electrical_observer']['edges']==[] and s['native_info']['electrical_observer']['edges']==[] for s in swsteps),
        'scope':'one ALF task, four prefixes, two paired seeds; two ScienceWorld instances, four seeds; descriptive frozen-policy results'}
    check('raw_unchanged_after_analysis', (file_hash(raw/k)==v for k,v in manifest['files'].items()))
    summary['audit_checks']=checks
    write_json(out/'review_summary.json',summary)
    table(out/'alf_condition_summary.csv',[{'condition':k,**{a:b for a,b in v.items() if a!='first_actions'}} for k,v in arms.items()])
    print(json.dumps({'output':str(out),'checks':len(checks),'correction':summary['diagnostic_correction'],'advantages':diagmatches},ensure_ascii=False))


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    s=json.loads((out/'review_summary.json').read_text(encoding='utf8'))
    dest=out/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none'})
    colors=['#52677d','#c59a42','#217d80']
    arms=['original','goal_reminder','clean_subgoal'];labels=['H0 Original','H1 Goal reminder','H2 Clean subgoal']
    fig,ax=plt.subplots(figsize=(11,5.3))
    metrics=['first_direct_target_placement','clean_attempt','sink_reached','clean_success','strict_success']
    titles=['Dirty target placement\nat first decision','Any clean command\nattempt','Cloth held\nat sink','Native clean\nfact acquired','Strict task\ncompletion']
    for j,arm in enumerate(arms):
        vals=[s['alf_arms'][arm][k] for k in metrics];xs=np.arange(len(metrics))+(j-1)*.25
        ax.bar(xs,vals,width=.23,color=colors[j],label=labels[j])
        for x,v in zip(xs,vals):ax.text(x,v+.12,f'{v}/8',ha='center',fontsize=9)
    ax.set(xticks=np.arange(len(metrics)),xticklabels=titles,ylim=(0,9.2),yticks=[0,2,4,6,8],ylabel='Continuation episodes (8 per condition)',title='ALFWorld: a first-decision shift without successful cleaning')
    ax.legend(frameon=False,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.08,.025,'4 pickup prefixes of ONE Clean task × 2 paired seeds; 930 new turns. Hint appears once. No training.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(dest/'01_alf_preparation.png',dpi=180);fig.savefig(dest/'01_alf_preparation.svg');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,5.4),gridspec_kw={'width_ratios':[1,1.1]})
    tasks=sorted(s['science_old_tasks'],key=lambda t:0 if t.startswith('power-component/7') else 1)
    names=['power/7','energy/5']
    for j,version in enumerate(['old','new']):
        data=s['science_'+version+'_tasks'];rs=[data[t]['no_match'] for t in tasks];xs=np.arange(2)+(j-.5)*.32
        axs[0].bar(xs,[r['value']*100 for r in rs],width=.30,color=[colors[0],colors[2]][j],label=['Old P0','New interface P0'][j])
        for x,r in zip(xs,rs):axs[0].text(x,r['value']*100+1.5,f"{r['value']:.1%}\n{r['numerator']}/{r['denominator']}",ha='center',fontsize=9)
    axs[0].set(xticks=[0,1],xticklabels=names,ylim=(0,103),ylabel='Unmatched commands / all active turns (%)',title='No-match changes differ by task')
    axs[0].legend(frameon=False,fontsize=9,loc='lower left')
    counts=Counter()
    for task in s['science_task_detail'].values():counts.update(task['connect_outcomes'])
    total=sum(task['connect_attempts'] for task in s['science_task_detail'].values())
    vals=[counts['unmatched_command'],counts['object_or_endpoint_resolution_failed'],sum(t['connect_success_count'] for t in s['science_new_tasks'].values())]
    axs[1].barh([2,1,0],vals,color=[colors[0],colors[1],colors[2]])
    for y,v in zip([2,1,0],vals):axs[1].text(v+1,y,f'{v}/{total}',va='center')
    axs[1].set(yticks=[2,1,0],yticklabels=['Unmatched command','Endpoint resolution failed','New connection'],xlim=(0,83),xlabel='New P0 connection attempts',title='118 attempts, no connection established')
    fig.suptitle('ScienceWorld: full reset runs, 4 episodes per task in each version',fontsize=14,fontweight='bold')
    fig.text(.06,.018,'Strict completion: old 0/8; new 0/8. Denominators differ with early termination (219 vs 385 turns).',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(dest/'02_science_old_new.png',dpi=180);fig.savefig(dest/'02_science_old_new.svg');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,5.1))
    old=s['science_old_overall'];new=s['science_new_overall']
    keys=['format_compliance','usable_action','generic_feedback_pre_anchor','normal_anchor_support']
    labels2=['Format\nflag','Extractable\naction','Generic feedback\nanchor share','Normal-anchor\nalternative\nsupport']
    for j,d in enumerate([old,new]):
        xs=np.arange(4)+(j-.5)*.34;rs=[d[k] for k in keys]
        axs[0].bar(xs,[r['value']*100 for r in rs],width=.32,color=[colors[0],colors[2]][j],label=['Old','New'][j])
        for x,r in zip(xs,rs):axs[0].text(x,r['value']*100+1,f"{r['value']:.0%}",ha='center',fontsize=8)
    axs[0].set(xticks=np.arange(4),xticklabels=labels2,ylim=(0,112),ylabel='%',title='Input expansion did not fix response format');axs[0].legend(frameon=False)
    axs[0].tick_params(axis='x',labelsize=9)
    scores=[];labels3=[]
    for task in tasks:
        values=s['science_new_tasks'][task]
        for i,v in enumerate(values['final_native_scores']):scores.append(v);labels3.append(('P' if task.startswith('power-component/7') else 'E')+str(i))
    axs[1].bar(np.arange(8),scores,color=[colors[0]]*4+[colors[2]]*4)
    for i,v in enumerate(scores):axs[1].text(i,v+(3 if v>=0 else -8),str(v),ha='center',fontsize=9)
    axs[1].axhline(0,color='#888',lw=.7);axs[1].set(xticks=np.arange(8),xticklabels=labels3,ylim=(-120,65),ylabel='Final native score (not training reward)',title='Partial native scores do not imply wiring')
    fig.text(.035,.025,'Alternative support retains the original format-validity gate. All offline AE, AS and joint advantages remain zero.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(dest/'03_science_format_and_scores.png',dpi=180);fig.savefig(dest/'03_science_format_and_scores.svg');plt.close(fig)
    for path in dest.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['audit','plot'])
    p.add_argument('--raw',type=Path);p.add_argument('--old',type=Path);p.add_argument('--tokenizer',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='audit':audit(args)
    else:plot(args.output)


if __name__=='__main__':main()
