"""Audit actual prompts, freeze an interface after smoke, compare real old/new runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .common import file_hash, read_jsonl, write_json
from .post_interface import parser_metadata

MATCHED_PROTOCOL={'model_revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','max_steps':50,
    'rollouts_per_task':4,'max_total_episodes':8,'prompt_tokens':4096,'response_tokens':512,
    'history_length':4,'temperature':0.7,'top_p':1.0,'simplifications':'',
    'gamma':0.95,'step_advantage_w':1.0,'reward_profile':'research_terminal_v1',
    'normalization':'mean_norm','enable_similarity':False,'invalid_action_penalty':0.0,'kl_penalty':0.0}

def audit(old,out):
    from .environments import parse_action
    from .post_analysis import table,write_rows
    out.mkdir(parents=True,exist_ok=False)
    steps=read_jsonl(old/'steps.jsonl')
    diffs=[]
    for row in steps:
        action,valid=parse_action(row['raw_response'],'scienceworld')
        raw=row['raw_response'];start=raw.find('<action>');end=raw.find('</action>')
        reference=raw[-20:] if start<0 or end<0 else raw[start+len('<action>'):end].strip()
        diffs.append({'run_id':row['run_id'],'analysis_step_id':row['analysis_step_id'],
            'old_action':row['parsed_action'],'candidate_action':action,'action_changed':action!=row['parsed_action'],
            'beacon_rlvmr_extracted_command':reference,'reference_action_differs':reference!=row['parsed_action'],
            'reference_scope':'first-tag extraction and 20-character fallback; no reward/format gating imported',
            'old_format_flag':row['parse_valid'],'candidate_format_flag':valid,
            **parser_metadata(row['raw_response'],action,'scienceworld')})
    write_rows(out/'parser_comparison.jsonl',diffs)
    seen=set()
    for row in steps:
        if row['task_instance_id'] in seen:continue
        seen.add(row['task_instance_id']);name=row['task_instance_id'].replace('/','-')
        (out/(name+'-old-actual-prompt.txt')).write_text(row['actor_prompt'],encoding='utf8')
    write_json(out/'audit.json',{'source_steps_sha256':file_hash(old/'steps.jsonl'),
        'steps':len(steps),'changed_actions':sum(d['action_changed'] for d in diffs),
        'unusable_action_count':sum(not d['parser_usable'] for d in diffs),
        'fallback_count':sum(d['parser_fallback_used'] for d in diffs),
        'reference_extraction_differs':sum(d['reference_action_differs'] for d in diffs),
        'old_prompt_has_templates':all('Available action templates:' in s['actor_prompt'] for s in steps),
        'old_prompt_has_public_object_block':all('Object references (lexical order, at most 80):' in s['actor_prompt'] for s in steps),
        'parser_decision':'unchanged; no fallback rewrite in matched interface comparison'})

def lock_interface(smoke,out):
    steps=read_jsonl(smoke/'steps.jsonl');eps=read_jsonl(smoke/'episodes.jsonl')
    cfg=json.loads((smoke/'resolved_config.json').read_text(encoding='utf8'))
    prov=json.loads((smoke/'environment_provenance.json').read_text(encoding='utf8'))
    if len(eps)!=2 or any(e['status']!='complete' for e in eps):raise ValueError('Need both real smoke episodes completed')
    if cfg['execution_kind']!='science_smoke' or cfg['max_steps']!=12:raise ValueError('Not a 12-turn smoke run')
    if not steps or any(s['interface_block_token_visible'] is not True for s in steps):raise ValueError('Public interface block not fully token-visible')
    if any(s['truncated_tokens'] for s in steps):raise ValueError('Smoke input truncated; inspect before locking')
    if any(s['native_info']['electrical_observer']['moves_after_read'] != s['native_info']['env_moves'] for s in steps
           if s['native_info']['electrical_observer'].get('error') is None):raise ValueError('Observer read changed move budget')
    write_json(out,{'status':'locked_after_real_smoke','selected_profile':cfg['prompt_profile'],
        'smoke_run_id':cfg['run_id'],'smoke_steps_sha256':file_hash(smoke/'steps.jsonl'),
        'jar_sha256':prov['jar_sha256'],'model_revision':cfg['model_revision'],
        'reason':'Templates, public object references, token visibility and native communication checked. No performance threshold or success selection.',
        'parser_profile':'p0_original_unchanged','matched_protocol':MATCHED_PROTOCOL})

def compare(old,new,out):
    from .post_analysis import analyze_run,table
    out.mkdir(parents=True,exist_ok=False)
    # Old data remains immutable: all reanalysis goes to this separate directory.
    a=analyze_run(old,out/'old_reanalysis');b=analyze_run(new,out/'new_reanalysis')
    ca=json.loads((old/'resolved_config.json').read_text(encoding='utf8'))
    cb=json.loads((new/'resolved_config.json').read_text(encoding='utf8'))
    differences={k:{'old':ca.get(k),'new':cb.get(k)} for k in MATCHED_PROTOCOL if ca.get(k)!=cb.get(k)}
    pa=json.loads((old/'environment_provenance.json').read_text(encoding='utf8'))
    pb=json.loads((new/'environment_provenance.json').read_text(encoding='utf8'))
    for k in ['jar_sha256','wrapper_sha256','package_version']:
        if pa.get(k)!=pb.get(k):differences[k]={'old':pa.get(k),'new':pb.get(k)}
    ma=json.loads((old/'run_manifest.json').read_text(encoding='utf8'))
    mb=json.loads((new/'run_manifest.json').read_text(encoding='utf8'))
    for k in ['torch','transformers']:
        if ma.get('versions',{}).get(k)!=mb.get('versions',{}).get(k):
            differences[k]={'old':ma.get('versions',{}).get(k),'new':mb.get('versions',{}).get(k)}
    write_json(out/'comparison_config_diff.json',{'other_changes':differences,
        'comparability':'matched_except_interface' if not differences else 'additional_differences_listed',
        'old_profile':ca.get('prompt_profile','p0_original'),'new_profile':cb.get('prompt_profile'),
        'new_run_id':cb['run_id'],'old_run_id':ca['run_id'],'new_is_full_rep0':cb.get('execution_kind')=='science_rep0' and len(b['episodes'])==8})
    rows=[]
    for version,data in [('old',a),('new',b)]:
        for task,s in data['subsets'].items():
            r={'version':version,'task':task,'episodes':s['episodes'],'active_steps':s['active_steps'],
               'connect_attempt_count':s['connect_attempt_count'],'connect_success_count':s['connect_success_count']}
            for k in ['strict_success','no_match','has_connection','target_focused','positive_score_no_connection',
                      'generic_feedback_pre_anchor','normal_anchor_support','nonzero_episode_advantage','nonzero_step_advantage']:
                for field in ['numerator','denominator','unknown_count','value']:r[k+'_'+field]=s[k][field]
            rows.append(r)
    table(out/'task_metrics.csv',rows);write_json(out/'plot_data.json',{'old':a,'new':b})
    (out/'report.md').write_text('# ScienceWorld old / new\n\nNew execution kind: '+str(cb.get('execution_kind'))+'; full re-P0: '+str(cb.get('execution_kind')=='science_rep0' and len(b['episodes'])==8)+'.\n\n'+
        ('Matched except interface.' if not differences else 'Additional differences: '+json.dumps(differences))+
        '\n\n'+ '\n'.join(f"- {r['version']} {r['task']}: success {r['strict_success_numerator']}/{r['strict_success_denominator']}; no-match {r['no_match_numerator']}/{r['no_match_denominator']}; successful connection events {r['connect_success_count']}." for r in rows)+
        '\n\nAll figures and counts use the same evidence annotation version. Zero rewards and advantages are valid results.\n',encoding='utf8')

def plot_compare(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((out/'plot_data.json').read_text(encoding='utf8'))
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    pairs=[(v,k,s) for v,a in data.items() for k,s in a['subsets'].items()]
    labels=[v+'\n'+('power/7' if k.startswith('power-component/7') else 'energy/5') for v,k,s in pairs]
    for name,metrics in [('SW02_connection_and_completion_old_new',['has_connection','strict_success']),
                         ('SW04_anchor_support_old_new',['generic_feedback_pre_anchor','normal_anchor_support'])]:
        fig,ax=plt.subplots(figsize=(10,4.8))
        for j,m in enumerate(metrics):
            rs=[s[m] for v,k,s in pairs];xs=[i+(j-.5)*.35 for i in range(len(pairs))]
            ax.bar(xs,[r['value'] or 0 for r in rs],width=.32,label=m)
            for x,r in zip(xs,rs):ax.text(x,(r['value'] or 0)+.02,f"{r['numerator']}/{r['denominator']}" if r['denominator'] else 'unknown',ha='center',fontsize=8)
        ax.set_xticks(range(len(pairs)),labels);ax.set_ylim(0,1.2);ax.legend(fontsize=8);ax.set_title('Frozen Qwen2.5-1.5B: '+name.split('_',1)[1]);fig.tight_layout()
        fig.savefig(figdir/(name+'.png'),dpi=180);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));bottom=[0.]*len(pairs)
    cats=sorted({c for v,k,s in pairs for c in s['execution_counts']})
    for c in cats:
        vals=[s['execution_counts'].get(c,0)/s['active_steps'] if s['active_steps'] else 0 for v,k,s in pairs]
        ax.bar(labels,vals,bottom=bottom,label=c);bottom=[x+y for x,y in zip(bottom,vals)]
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1));ax.set_ylim(0,1);ax.set_title('Frozen Qwen2.5-1.5B: execution evidence, unknown retained');fig.tight_layout()
    fig.savefig(figdir/'SW01_execution_outcomes_old_new.png',dpi=180);fig.savefig(figdir/'SW01_execution_outcomes_old_new.svg');plt.close(fig)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['audit-interface','lock-interface','compare','plot-compare'])
    p.add_argument('--old',type=Path);p.add_argument('--new',type=Path);p.add_argument('--smoke',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='audit-interface':audit(a.old,a.output)
    elif a.command=='lock-interface':lock_interface(a.smoke,a.output)
    elif a.command=='compare':compare(a.old,a.new,a.output)
    else:plot_compare(a.output)

if __name__=='__main__':main()
