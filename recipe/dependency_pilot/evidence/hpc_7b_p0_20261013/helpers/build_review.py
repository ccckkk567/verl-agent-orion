"""Assemble recorded metrics and publication figures; no inference or environment calls."""
import csv,json,sys,shutil
from pathlib import Path
from collections import Counter
root=Path(r'C:\Users\19096\Desktop\verl-agent\outputs\dependency_pilot\7b-p0-20261012')
out=root/'review-159779-159780'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def table(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
rows=[];verify={};all_episodes=[]
for env in ['alfworld','scienceworld']:
    plot=load(out/env/'plot_data.json')
    job='159779' if env=='alfworld' else '159780'
    new=root/'hpc-159779-159780'/f'{env}-p0-7b-{job}'
    remote=load(new/'analysis.json');local=load(out/env/'new_reanalysis/analysis.json')
    assert remote==local,'Reanalysis differs from HPC analysis'
    remote_diag={s['analysis_step_id']:s for s in lines(new/'diagnostics.jsonl')}
    local_diag=lines(out/env/'new_reanalysis/diagnostics.jsonl')
    assert len(remote_diag)==len(local_diag)==400
    max_error=max(abs(d[k]-remote_diag[d['analysis_step_id']][k]) for d in local_diag for k in ['episode_adv_scalar','step_adv_scalar','joint_adv_scalar'])
    assert max_error==0
    first=Counter(r['tag_category'] for r in lines(out/'format_annotations.jsonl') if r['run_id']==remote['run_id'] and r['analysis_step_id'].endswith(':0'))
    verify[env]={'hpc_and_local_analysis_equal':True,'diagnostic_rows':400,'maximum_advantage_recompute_difference':max_error,
        'AE_AS_joint_all_zero':all(d[k]==0 for d in local_diag for k in ['episode_adv_scalar','step_adv_scalar','joint_adv_scalar']),
        'first_step_tag_categories':dict(first),'tracker':load(new/'tracking.json')}
    for version,a in plot.items():
        model='1.5B' if version=='old' else '7B'
        for e in a['episodes']:all_episodes.append({'model':model,'env':env,**e})
        for task,s in a['subsets'].items():
            label=('Pick' if 'pick_and_place' in task else 'Clean') if env=='alfworld' else ('Power/7' if task.startswith('power-component/7') else 'Energy/5')
            row={'environment':env,'task_label':label,'task_instance_id':task.split('|')[0],'model':model,'run_id':a['run_id'],'episodes':s['episodes'],'active_steps':s['active_steps']}
            for k,v in s.items():
                if isinstance(v,dict) and 'value' in v:
                    for field,value in v.items():row[k+'_'+field]=value
            if env=='scienceworld':row['connect_attempts']=s['connect_attempt_count'];row['connect_successes']=s['connect_success_count']
            rows.append(row)
table(out/'task_comparison.csv',rows);save(out/'task_comparison.json',rows)
table(out/'all_episode_outcomes.csv',all_episodes);save(out/'recompute_verification.json',verify)
save(out/'source_index.json',{'raw_root':str(root/'hpc-159779-159780'),'import_manifest':load(root/'hpc-159779-159780/import_manifest.json'),
    'reanalysis_sources':{env:load(out/env/'comparison_config_diff.json') for env in ['alfworld','scienceworld']},
    'sampling_commit':'1de2441b051ae8e1eecf38f3c769abb03e029916','branch':'exp/dep-prep-p1-sci-rep0-20261012',
    'local_review_date':'2026-10-13','new_sampling_in_local_review':False,'remote_sha256_file_provided':False})

# Matplotlib is deliberately in a separate process from Torch/GiGPO.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
figdir=out/'figures';figdir.mkdir(exist_ok=True)
colors=['#5b7c99','#b6533b']
tasks=['Pick','Clean','Power/7','Energy/5']
fig,axes=plt.subplots(1,2,figsize=(12.6,4.4),layout='constrained')
for ax,metric,title in zip(axes,['strict_success','usable_action'],['Strict task completion','Parser-usable command (step denominator)']):
    for j,model in enumerate(['1.5B','7B']):
        rr=[next(r for r in rows if r['task_label']==t and r['model']==model) for t in tasks]
        xx=[i+(j-.5)*.34 for i in range(4)]
        ax.bar(xx,[r[metric+'_value'] or 0 for r in rr],width=.31,color=colors[j],label=model)
        for x,r in zip(xx,rr):ax.text(x,(r[metric+'_value'] or 0)+.026,f"{r[metric+'_numerator']}/{r[metric+'_denominator']}",ha='center',fontsize=9)
    ax.set_xticks(range(4),tasks);ax.set_ylim(0,1.18);ax.set_ylabel('Fraction');ax.set_title(title);ax.legend(frameon=False,loc='upper right')
fig.suptitle('Frozen Qwen2.5: matched root P0 | 4 episodes per task/model',fontsize=14)
fig.savefig(figdir/'completion_and_action_format.png',dpi=200);fig.savefig(figdir/'completion_and_action_format.svg');plt.close(fig)

fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
metrics=['object_pickup','any_sink_reached','sink_reached','clean_success','clean_then_target']
labels=['Picked up cloth','Visited sink\n(any state)','At sink\nholding cloth','Actually cleaned','Cleaned then\ntarget placement']
for j,model in enumerate(['1.5B','7B']):
    r=next(r for r in rows if r['task_label']=='Clean' and r['model']==model)
    xx=[i+(j-.5)*.34 for i in range(len(metrics))]
    ax.bar(xx,[r[m+'_value'] or 0 for m in metrics],width=.31,color=colors[j],label=model)
    for x,m in zip(xx,metrics):ax.text(x,(r[m+'_value'] or 0)+.025,f"{r[m+'_numerator']}/{r[m+'_denominator']}",ha='center',fontsize=10)
ax.set_xticks(range(len(metrics)),labels);ax.set_ylim(0,1.2);ax.set_ylabel('Episode fraction');ax.legend(frameon=False)
ax.set_title('ALFWorld Clean: observed native facts | events may overlap')
fig.savefig(figdir/'alfworld_clean_behavior.png',dpi=200);fig.savefig(figdir/'alfworld_clean_behavior.svg');plt.close(fig)

fmt=load(out/'format_failure_counts.json');fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
for ax,env in zip(axes,['alfworld','scienceworld']):
    rr=[next(r for r in fmt if r['env']==env and r['model']==model and (('pick_and_place' in r['task']) if t=='Pick' else ('pick_clean' in r['task']) if t=='Clean' else r['task']=='power-component/7' if t=='Power/7' else 'renewable' in r['task'])) for t in (tasks[:2] if env=='alfworld' else tasks[2:]) for model in ['1.5B','7B']]
    bottom=[0.]*4
    for metric,color,label in [('complete_xml_action','#5b7c99','Complete XML action tags'),('malformed_square_action_marker','#b6533b','Malformed square action marker'),('other_missing_or_incomplete_action_tags','#c7c4ba','Other missing/incomplete tags')]:
        vals=[r[metric]/r['active_steps'] for r in rr]
        ax.bar(range(4),vals,bottom=bottom,color=color,label=label)
        bottom=[x+y for x,y in zip(bottom,vals)]
    ax.set_xticks(range(4),[model+'\n'+t for t in (tasks[:2] if env=='alfworld' else tasks[2:]) for model in ['1.5B','7B']]);ax.set_ylim(0,1.13);ax.set_ylabel('Active response fraction');ax.set_title(env)
    for i,r in enumerate(rr):ax.text(i,1.025,'n='+str(r['active_steps']),ha='center',fontsize=9)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=3,frameon=False,fontsize=10)
fig.suptitle('Observed response tags — no repaired commands executed',fontsize=14)
fig.savefig(figdir/'response_tag_categories.png',dpi=200,bbox_inches='tight');fig.savefig(figdir/'response_tag_categories.svg',bbox_inches='tight');plt.close(fig)
print(json.dumps(verify,indent=2))
