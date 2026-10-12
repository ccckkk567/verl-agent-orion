"""Evidence-only post-P0 metrics. Plotting runs without importing Torch."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from .common import read_jsonl, write_json
from .post_interface import execution_outcome, parser_metadata

VERSION='post_p0_evidence_v2_case_insensitive_tags'
GENERIC={'No known action matches that input.','Nothing happens.'}

def rate(values):
    values=list(values);known=[v for v in values if v is not None]
    n=sum(v is True for v in known)
    return {'value':n/len(known) if known else None,'numerator':n,'denominator':len(known),'unknown_count':len(values)-len(known)}

def ever(values):
    values=list(values)
    return True if True in values else (None if not values or None in values else False)

def facts(info):
    if not isinstance(info.get('facts'),list): return None
    return {(f['name'].lower(),tuple(a['name'].strip() for a in f['arguments'])) for f in info['facts']}

def cloth_state(info):
    f=facts(info)
    if f is None:return None
    held={a[1] for n,a in f if n=='holds' and a[1].startswith('cloth ')}
    clean={a[0] for n,a in f if n=='isclean' and a[0].startswith('cloth ')}
    placed={a[0] for n,a in f if n=='inreceptacle' and a[0].startswith('cloth ') and a[1].startswith('toilet ')}
    locations={a[1] for n,a in f if n=='atlocation' and a[0].startswith('agent')}
    sinks={a[1] for n,a in f if n=='receptacleatlocation' and a[0].startswith('sinkbasin ')}
    return {'held':held,'clean':clean,'placed':placed,'at_sink':bool(locations&sinks) if locations and sinks else None}

def annotate(row,pre):
    info=row['native_info'];action=row.get('parsed_action','');feedback=row['observation_post']
    event={'run_id':row['run_id'],'analysis_step_id':row['analysis_step_id'],'traj_uid':row['traj_uid'],
        'step_index':row['step_index'],'absolute_step_index':row.get('absolute_step_index',row['step_index']),
        'source':'native facts/read-only state/explicit feedback','generic_pre_anchor':row['anchor_pre_raw'].strip() in GENERIC,
        **parser_metadata(row['raw_response'],action,row['env'])}
    if row['env']=='alfworld':
        before,after=cloth_state(pre),cloth_state(info)
        newly_clean=after['clean']-before['clean'] if before is not None and after is not None else None
        new_placed=after['placed']-before['placed'] if before is not None and after is not None else None
        event.update(sink_reached=(bool(after['held']) and after['at_sink']) if after is not None else None,
            clean_attempt=bool(re.match(r'^clean\s+cloth\b',action,re.I)),
            clean_success=bool(newly_clean) if newly_clean is not None else None,
            cleaned_ids=sorted(newly_clean) if newly_clean is not None else None,
            original_cloth_cleaned=('cloth 4' in newly_clean) if newly_clean is not None else None,
            placement_before_clean=bool(new_placed-after['clean']) if new_placed is not None else None,
            clean_then_target=bool(new_placed&before['clean']) if new_placed is not None else None,
            target_placed_ids=sorted(new_placed) if new_placed is not None else None,
            take_success=bool(after['held']-before['held']) if before is not None and after is not None else None,
            put_success=bool(before['held']-after['held']) and action.startswith('move ') if before is not None and after is not None else None,
            execution_outcome='accepted_state_change' if before is not None and after!=before else ('precondition_blocked' if feedback.strip()=='Nothing happens.' else 'unknown'))
    else:
        before=pre.get('electrical_observer',{});after=info.get('electrical_observer',{})
        attempt=action.lower().startswith('connect')
        ack=' is now connected to ' in feedback.lower() if attempt else False
        added=None
        if before.get('edges') is not None and after.get('edges') is not None:
            added=sorted(set(map(tuple,after['edges']))-set(map(tuple,before['edges'])))
        outcome=execution_outcome(action,feedback,before,after)
        failed=outcome in ['unmatched_command','object_or_endpoint_resolution_failed','precondition_blocked']
        connected=bool(added) if added is not None else (True if ack else (False if not attempt or failed else None))
        event.update(connect_attempt=attempt,connection_acknowledged=ack,
            connection_newly_established=bool(added) if added is not None else None,
            connect_success=connected,connection_edges_added=added,
            connection_present=bool(after['edges']) if after.get('edges') is not None else None,
            target_focused=action.lower().startswith('focus on ') and info.get('native_score',-100)>0 and feedback.strip().lower()=='you focus on the green light bulb.',
            workshop_reached='This room is called the workshop.' in info.get('look',''),
            execution_outcome=outcome, native_score=info.get('native_score'))
    return event

def write_rows(path,rows):
    with path.open('w',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def table(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize_subset(episodes,steps,events,diagnostics):
    ids={e['traj_uid'] for e in episodes};steps=[s for s in steps if s['traj_uid'] in ids]
    events=[e for e in events if e['traj_uid'] in ids];diag=[d for d in diagnostics if d['traj_uid'] in ids]
    scores=[s['native_score_post'] for s in steps if s['native_score_post'] is not None]
    result={'episodes':len(episodes),'active_steps':len(steps),
        'strict_success':rate(e['strict_success'] if e['status']=='complete' else None for e in episodes),
        'budget_exhaustion':rate(e['termination_reason']=='budget_truncated' for e in episodes),
        'format_compliance':rate(s.get('format_compliance',s['parse_valid']) for s in steps),
        'usable_action':rate(e['parser_usable'] for e in events),
        'no_match':rate(e['execution_outcome']=='unmatched_command' for e in events),
        'execution_unknown':rate(e['execution_outcome']=='unknown' for e in events),
        'input_truncated':rate(s['truncated_tokens']>0 for s in steps),
        'output_limited':rate(s['output_at_token_limit'] for s in steps),
        'generated_tokens':sum(s['generated_tokens'] for s in steps),
        'execution_counts':dict(Counter(e['execution_outcome'] for e in events)),
        'generic_feedback_pre_anchor':rate(e['generic_pre_anchor'] for e in events),
        'final_native_scores':[e['native_score'] for e in episodes],
        'initial_native_scores':[e.get('initial_native_score') for e in episodes],
        'max_native_score':max(scores) if scores else None}
    event_by_id={e['analysis_step_id']:e for e in events}
    for key in ['episode','step','joint']:
        result['nonzero_'+key+'_advantage']=rate(abs(d[key+'_adv_scalar'])>1e-8 for d in diag)
    result['alternative_action_coverage']=rate(d['alternative_action_supported'] for d in diag)
    normal=[d for d in diag if not event_by_id[d['analysis_step_id']]['generic_pre_anchor']]
    result['normal_anchor_support']=rate(d['alternative_action_supported'] for d in normal)
    result['supported_with_nonzero_AS']=rate(d['alternative_action_supported'] and abs(d['step_adv_scalar'])>1e-8 for d in diag)
    groups=defaultdict(list)
    for e in episodes:groups[e['episode_group_uid']].append(e)
    result['mixed_terminal_group_rate']=rate(any(e['strict_success'] is True for e in g) and any(e['strict_success'] is False for e in g)
        for g in groups.values() if len(g)>=2 and all(e['status']=='complete' and e['strict_success'] is not None for e in g))
    return result

def analyze_run(run_dir, output=None):
    out=output or run_dir;out.mkdir(parents=True,exist_ok=True)
    episodes=read_jsonl(run_dir/'episodes.jsonl');steps=[s for s in read_jsonl(run_dir/'steps.jsonl') if s['active_mask']]
    config=json.loads((run_dir/'resolved_config.json').read_text(encoding='utf8'))
    resets={r['traj_uid']:r['info'] for r in read_jsonl(run_dir/'resets.jsonl')}
    for ep in episodes:
        ep.setdefault('initial_native_score',resets.get(ep['traj_uid'],{}).get('native_score'))
    previous=dict(resets);events=[]
    for row in steps:
        pre=row.get('native_info_pre',previous.get(row['traj_uid'],{}))
        events.append(annotate(row,pre));previous[row['traj_uid']]=row['native_info']
    groups=defaultdict(list)
    for ep in episodes:groups[ep['episode_group_uid']].append(ep)
    if config.get('phase')=='post_p0' and config['env']=='alfworld':
        if any(len({(e.get('prefix_id'),e.get('condition')) for e in g})!=1 for g in groups.values()):
            raise ValueError('P1 credit group crosses prefix or prompt arm')
    eligible=[e for g in groups.values() if len(g)==g[0]['expected_group_size'] and all(e['status']=='complete' for e in g) for e in g]
    eligible_ids={e['traj_uid'] for e in eligible}
    from .diagnostics import recompute
    diag,anchors,checks=recompute([s for s in steps if s['traj_uid'] in eligible_ids],eligible,config)
    for d in diag:d['run_id']=config['run_id']
    write_rows(out/'diagnostics.jsonl',diag);write_rows(out/'anchors.jsonl',anchors);write_rows(out/'event_annotations.jsonl',events)
    write_json(out/'diagnostic_checks.json',checks)
    by_ep=defaultdict(list)
    for e in events:by_ep[e['traj_uid']].append(e)
    outcome=[]
    for ep in episodes:
        ev=by_ep[ep['traj_uid']];r={k:ep.get(k) for k in ['run_id','traj_uid','task_instance_id','prefix_id','condition','continuation_seed','remaining_turns','agent_turns','status','strict_success','termination_reason','native_score','initial_native_score']}
        if config['env']=='alfworld':
            for k in ['sink_reached','clean_attempt','clean_success','original_cloth_cleaned','placement_before_clean','clean_then_target']:
                r[k]=ever(e[k] for e in ev)
            cleaned=[e for e in ev if e['clean_success'] is True]
            r['clean_first_turn']=cleaned[0]['step_index']+1 if cleaned else None
            r['clean_first_turn_censored']=not bool(cleaned)
            r['take_success_count']=sum(e['take_success'] is True for e in ev)
            r['put_success_count']=sum(e['put_success'] is True for e in ev)
            r['loop_take_put_events']=max(0,min(r['take_success_count'],r['put_success_count'])-1)
        else:
            r['connect_attempt_count']=sum(e['connect_attempt'] for e in ev)
            r['connect_success_count']=sum(e['connect_success'] is True for e in ev)
            r['has_connection']=ever(e['connect_success'] for e in ev)
            r['target_focused']=ever(e['target_focused'] for e in ev)
            r['workshop_initial']='This room is called the workshop.' in resets.get(ep['traj_uid'],{}).get('look','')
            r['workshop_reached']=r['workshop_initial'] or ever(e['workshop_reached'] for e in ev)
            r['positive_score_no_connection']=(ep['native_score']>0 and not r['has_connection']) if ep['native_score'] is not None and r['has_connection'] is not None else None
        if ep['status']!='complete':
            # Positive observed events survive, but incomplete negative outcomes are censored.
            for k,v in list(r.items()):
                if v is False and k not in ['clean_first_turn_censored','workshop_initial']:r[k]=None
        outcome.append(r)
    table(out/'episode_outcomes.csv',outcome);write_json(out/'episode_outcomes.json',outcome)
    subsets={}
    keys=sorted({(e.get('prefix_id') or e['task_instance_id'],e.get('condition','root_original')) for e in episodes})
    for identity,arm in keys:
        eps=[e for e in episodes if (e.get('prefix_id') or e['task_instance_id'])==identity and e.get('condition','root_original')==arm]
        summary=summarize_subset(eps,steps,events,diag)
        rows=[r for r in outcome if r['traj_uid'] in {e['traj_uid'] for e in eps}]
        for k in (['sink_reached','clean_attempt','clean_success','original_cloth_cleaned','placement_before_clean','clean_then_target'] if config['env']=='alfworld' else ['has_connection','target_focused','workshop_reached','positive_score_no_connection']):
            summary[k]=rate(r[k] for r in rows)
        if config['env']=='scienceworld':
            summary['connect_attempt_count']=sum(r['connect_attempt_count'] for r in rows)
            summary['connect_success_count']=sum(r['connect_success_count'] for r in rows)
            n=summary['connect_attempt_count'];summary['connect_success_per_attempt']=summary['connect_success_count']/n if n else None
        subsets[identity+'|'+arm]=summary
    result={'analysis_version':VERSION,'run_id':config['run_id'],'env':config['env'],'execution_kind':config.get('execution_kind','old_p0'),'subsets':subsets,
        'overall':summarize_subset(episodes,steps,events,diag),'episodes':outcome,
        'policy_updated':False,'advantages':'unchanged upstream, step-weighted episode mean, exact pre-observation groups; incomplete groups excluded',
        'unknown_policy':'null is excluded from known-label denominator; incomplete negative events censored',
        'loop_rule':'max(0,min(successful suffix takes,successful suffix puts)-1); replay excluded'}
    if config['env']=='alfworld':
        arms={}
        for arm in ['original','goal_reminder','clean_subgoal']:
            rows=[r for r in outcome if r['condition']==arm]
            armres={k:rate(r[k] for r in rows) for k in ['clean_success','strict_success','sink_reached','placement_before_clean']}
            for k in ['clean_success','strict_success']:
                vals=[s[k]['value'] for key,s in subsets.items() if key.endswith('|'+arm) and s[k]['value'] is not None]
                armres[k+'_prefix_equal_mean']=sum(vals)/len(vals) if vals else None
            arms[arm]=armres
        result['arms']=arms
        result['arm_differences']={}
        for a,b in [('clean_subgoal','original'),('clean_subgoal','goal_reminder'),('goal_reminder','original')]:
            result['arm_differences'][a+'-'+b]={k:(arms[a][k+'_prefix_equal_mean']-arms[b][k+'_prefix_equal_mean'])
                if arms[a][k+'_prefix_equal_mean'] is not None and arms[b][k+'_prefix_equal_mean'] is not None else None for k in ['clean_success','strict_success']}
    write_json(out/'analysis.json',result)
    flat={}
    for group,values in {'overall':result['overall'],**subsets}.items():
        for key,v in values.items():
            if isinstance(v,dict) and 'value' in v:flat[group+'/'+key]=v
    flat['progress/completed_samples']={'value':sum(e['status']=='complete' for e in episodes)}
    flat['cost/env_steps_total']={'value':len(steps)}
    flat['training_updates']={'value':0}
    write_json(out/'metrics.json',flat)
    table(out/'metrics.csv',[{'metric':k,**v} for k,v in flat.items()])
    (out/'report.md').write_text('# '+config['run_id']+'\n\nFrozen policy; training updates = 0.\n\n'+
        '\n'.join(f"- {k}: strict completion {s['strict_success']['numerator']}/{s['strict_success']['denominator']}; active steps {s['active_steps']}." for k,s in subsets.items())+
        '\n\nRead analysis.json and episode_outcomes.csv for per-prefix/arm/task results; event_annotations.jsonl joins raw steps by run_id + analysis_step_id. Unknowns remain null. Replay is excluded; partial/infrastructure outcomes are not counted as strict failures.\n',encoding='utf8')
    return result

def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads((out/'analysis.json').read_text(encoding='utf8'))
    dest=out/'figures';dest.mkdir(exist_ok=True)
    if a['env']=='alfworld':
        arms=list(a.get('arms',{}));fig,ax=plt.subplots(figsize=(9,4.5))
        for j,k in enumerate(['clean_success','strict_success']):
            xs=[i+(j-.5)*.35 for i in range(len(arms))];rs=[a['arms'][arm][k] for arm in arms]
            ax.bar(xs,[r['value'] or 0 for r in rs],width=.32,label=k)
            for x,r in zip(xs,rs):ax.text(x,(r['value'] or 0)+.025,f"{r['numerator']}/{r['denominator']}" if r['denominator'] else 'unknown',ha='center',fontsize=9)
        ax.set_xticks(range(len(arms)),arms);ax.set_ylim(0,1.15);ax.legend();ax.set_title('ALFWorld frozen 1.5B: prefix continuations (one task)')
        name='AF01_clean_and_completion_by_arm'
    else:
        labels=[k.split('|')[0].replace('power-component-renewable-vs-nonrenewable-energy/','energy/') for k in a['subsets']]
        cats=sorted({c for s in a['subsets'].values() for c in s['execution_counts']})
        fig,ax=plt.subplots(figsize=(10,4.5));bottom=[0.]*len(labels)
        for cat in cats:
            vals=[s['execution_counts'].get(cat,0)/s['active_steps'] if s['active_steps'] else 0 for s in a['subsets'].values()]
            ax.bar(labels,vals,bottom=bottom,label=cat);bottom=[x+y for x,y in zip(bottom,vals)]
        ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1));ax.set_ylim(0,1)
        ax.set_title(f"ScienceWorld {a.get('execution_kind','')} | frozen 1.5B | n={a['overall']['episodes']}, steps={a['overall']['active_steps']}",fontsize=12)
        name='SW01_execution_outcomes'
    fig.tight_layout();fig.savefig(dest/(name+'.png'),dpi=180);fig.savefig(dest/(name+'.svg'));plt.close(fig)
    if a['env']=='alfworld' and a['subsets']:
        prefixes=sorted({k.split('|')[0] for k in a['subsets']});arms=['original','goal_reminder','clean_subgoal']
        cells=[]
        for prefix in prefixes:
            line=[]
            for arm in arms:
                s=a['subsets'].get(prefix+'|'+arm)
                line.append('not run' if not s else '\n'.join(f"{k}: {s[k]['numerator']}/{s[k]['denominator']}" for k in ['clean_success','strict_success']))
            cells.append(line)
        fig,ax=plt.subplots(figsize=(11,1.4+len(prefixes)*.7));ax.axis('off')
        t=ax.table(cellText=cells,rowLabels=prefixes,colLabels=arms,cellLoc='center',loc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,2.7)
        ax.set_title('ALFWorld frozen 1.5B: four prefixes of one task',pad=20);fig.tight_layout()
        fig.savefig(dest/'AF02_prefix_outcome_matrix.png',dpi=180);fig.savefig(dest/'AF02_prefix_outcome_matrix.svg');plt.close(fig)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['analyze','plot','track']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.command=='plot':plot(args.run_dir)
    elif args.command=='track':
        import torch
        from .runtime import PilotTracking
        cfg=json.loads((args.run_dir/'resolved_config.json').read_text(encoding='utf8'))
        tracker=PilotTracking(args.run_dir,cfg)
        try:tracker.log_summary(json.loads((args.run_dir/'metrics.json').read_text(encoding='utf8')),sorted((args.run_dir/'figures').glob('*.png')))
        finally:tracker.close()
    else:analyze_run(args.run_dir,args.output)

if __name__=='__main__':main()
