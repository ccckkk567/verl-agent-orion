"""Read-only diagnostics and stop conditions around the unmodified GiGPO result."""
import os,json,math,time,random
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(os.environ['PILOT_ROOT']);ARM=os.environ['PILOT_ARM'];LAST={};UP=0;TRACE=[]
def set_batch(batch):
 global TRACE
 TRACE=[json.loads(x) for x in batch.non_tensor_batch["pilot_trace"]]

def stats(x,prefix):
 x=np.asarray(x,dtype=float)
 assert np.isfinite(x).all(),'NONFINITE_ADVANTAGE'
 return {prefix+'_'+k:float(v) for k,v in dict(mean=x.mean(),std=x.std(),abs_mean=np.abs(x).mean(),nonzero_fraction=(np.abs(x)>1e-8).mean(),positive_fraction=(x>1e-8).mean(),negative_fraction=(x< -1e-8).mean()).items()}
def record_advantages(episode,step,total,mask,groups,index,traj,returns,anchors):
 global LAST,UP
 UP+=1
 # A is constant over response tokens; average over each valid response once.
 scalar=lambda t:(t.sum(-1)/mask.sum(-1).clamp(min=1)).detach().cpu().numpy()
 ae,as_,at=map(scalar,[episode,step,total]);ret=returns.detach().cpu().numpy()
 assert np.allclose(at,ae+as_,rtol=1e-5,atol=1e-5),'JOINT_ADV_MISMATCH'
 pool=defaultdict(list)
 for i,g in enumerate(groups):pool[g].append(i)
 assert len(pool)<len(groups),'ALL_STEP_GROUPS_SINGLETON'
 task_returns=defaultdict(set)
 for i,v in enumerate(ret):task_returns[str(index[i])].add(round(float(v),8))
 if not np.any(np.abs(as_)>1e-8) and any(len(v)>1 for v in task_returns.values()):raise RuntimeError('ZERO_STEP_SIGNAL_WITH_MIXED_RETURNS')
 group_rows=[]
 for gid,ix in pool.items():
  nt=len(set(str(traj[i]) for i in ix));nr=len(set(round(float(ret[i]),8) for i in ix))
  group_rows.append(dict(group_id=str(gid),task_instance_id=str(index[ix[0]]).split(':')[-1],P_M_F=json.loads(str(anchors[ix[0]])),trajectory_ids=[str(traj[i]) for i in ix],batch_step_indices=ix,returns=[float(ret[i]) for i in ix],A_episode=ae[ix].tolist(),A_step=as_[ix].tolist(),A_total=at[ix].tolist(),group_size=len(ix),cross=nt>=2,effective=nt>=2 and nr>=2))
 for g in group_rows:
  ix=g['batch_step_indices'];g['steps']=[TRACE[i] for i in ix]
  g['step_indices']=[TRACE[i]['step_idx'] for i in ix]
  g['normalized_actions']=[TRACE[i]['normalized_action'] for i in ix]
  g['decision_informative']=g['effective'] and len({json.dumps(x,sort_keys=True) for x in g['normalized_actions']})>=2
 sizes=[g['group_size'] for g in group_rows];n=len(groups)
 LAST=dict(update=UP,arm=ARM,**stats(ae,'episode_adv'),**stats(as_,'step_adv'),num_step_groups=len(sizes),singleton_group_rate=float(np.mean(np.array(sizes)==1)),non_singleton_step_coverage=sum(s for s in sizes if s>1)/n,cross_trajectory_peer_coverage=sum(g['group_size'] for g in group_rows if g['cross'])/n,effective_peer_coverage=sum(g['group_size'] for g in group_rows if g['effective'])/n,mean_group_size=float(np.mean(sizes)),median_group_size=float(np.median(sizes)),max_group_size=max(sizes),oom=False,nan_or_inf=False)
 LAST['decision_informative_peer_coverage']=sum(g['group_size'] for g in group_rows if g['decision_informative'])/n
 d=ROOT/'runs'/ARM/'debug_groups';d.mkdir(parents=True,exist_ok=True)
 rng=random.Random(2026092700+UP)
 ranked=list(group_rows);rng.shuffle(ranked)
 # Prioritize informative categories, randomize within each category.
 chosen=sorted(ranked,key=lambda g:(not g['effective'],not g['decision_informative'],not any(any(t.get('carried_mask',[])) for t in g['steps']),-max(abs(x) for x in g['A_step'])))[:20]
 (d/f'update_{UP:04d}.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in chosen))
 (ROOT/'runs'/ARM/f'advantage_{UP:04d}.json').write_text(json.dumps(LAST,indent=2))
def record_update(metrics,update):
 for k,v in metrics.items():
  if isinstance(v,(int,float)) and not math.isfinite(v):raise RuntimeError('NONFINITE_METRIC '+k)
 outcomes=[json.loads(p.read_text()) for p in (ROOT/'runs'/ARM/f'update_{update:04d}').glob('*/outcome.json')]
 assert len(outcomes)==16,'INCOMPLETE_UPDATE'
 assert np.mean([o['valid_action_rate'] for o in outcomes])>=0.8,'VALID_ACTION_RATE_COLLAPSE_BELOW_0_8'
 memories=[json.loads(p.read_text()) for p in (ROOT/'runs'/ARM/f'update_{update:04d}').glob('*/steps/*/memory.json')]
 ages=[a for m in memories for a,c in zip(m['P_age'],m['carried_mask']) if c]
 unknown=sum(json.loads(m['anchor_obs_baseline'])[0].count('U') for m in memories)
 diag=dict(P_obs_known_ratio=float(np.mean([sum(v!='U' for v in json.loads(m['anchor_obs_baseline'])[0])/len(m['P_mem']) for m in memories])) if memories else None,P_mem_known_ratio=float(np.mean([sum(v!='U' for v in m['P_mem'])/len(m['P_mem']) for m in memories])) if memories else None,carry_recovery_rate=len(ages)/unknown if unknown else 0,carried_predicate_step_rate=len(ages)/sum(len(m['P_mem']) for m in memories) if memories else 0,mean_memory_age=float(np.mean(ages)) if ages else 0,p95_memory_age=float(np.percentile(ages,95)) if ages else 0,max_memory_age=max(ages,default=0),num_F_to_T_overwrites=sum(m['transition_types'].count('OVERWRITE_F_TO_T') for m in memories),num_T_to_F_overwrites=sum(m['transition_types'].count('OVERWRITE_T_TO_F') for m in memories))
 row=dict(LAST,update=update,official_metrics=metrics,memory_diagnostics=diag,recorded_at=time.time())
 for field,key in {'policy_loss':'actor/pg_loss','kl':'actor/ppo_kl','entropy':'actor/entropy_loss','clip_fraction':'actor/pg_clipfrac','grad_norm':'actor/grad_norm','learning_rate':'actor/lr','gpu_peak_allocated':'perf/max_memory_allocated_gb','gpu_peak_reserved':'perf/max_memory_reserved_gb','rollout_time':'timing_s/gen','train_time':'timing_s/update_actor','tokens':'perf/total_num_tokens'}.items():row[field]=metrics.get(key)
 row['task_outcomes']=[]
 for tid in ['3256fd1e1c9855d3','dd820aeaef5c4c22']:
  oo=[json.loads(p.read_text()) for p in (ROOT/'runs'/ARM/f'update_{update:04d}').glob(tid+'_r*/outcome.json')]
  if oo:row['task_outcomes'].append(dict(task_instance_id=tid,num_rollouts=len(oo),num_success=sum(o['success'] for o in oo),success_rate=np.mean([o['success'] for o in oo]),mixed_group=0<sum(o['success'] for o in oo)<len(oo),mean_steps=np.mean([o['num_steps'] for o in oo]),termination_reason_counts=dict(Counter(o['termination_reason'] for o in oo)),parse_rate=np.mean([o['parse_rate'] for o in oo]),valid_action_rate=np.mean([o['valid_action_rate'] for o in oo])))
 with (ROOT/'runs'/ARM/'update_metrics.jsonl').open('a') as f:f.write(json.dumps(row,default=float)+'\n')
