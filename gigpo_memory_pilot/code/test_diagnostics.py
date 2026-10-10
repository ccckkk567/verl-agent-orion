"""Synthetic schema checks; these are not training results."""
import os,json,tempfile,importlib
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='pilot_diag_test_') as tmp:
 os.environ['PILOT_ROOT']=tmp;os.environ['PILOT_ARM']='obs'
 d=importlib.import_module('diagnostics');root=Path(tmp)/'runs/obs/update_0001';root.mkdir(parents=True)
 try:d.record_update({},1)
 except AssertionError as e:assert str(e)=='INCOMPLETE_UPDATE'
 else:raise AssertionError('missing outcomes accepted')
 for tid in ['3256fd1e1c9855d3','dd820aeaef5c4c22']:
  for i in range(8):
   p=root/f'{tid}_r{i}';p.mkdir()
   (p/'outcome.json').write_text(json.dumps(dict(success=i==0,num_steps=2,termination_reason='budget',parse_rate=1,valid_action_rate=1)))
   q=p/'steps/001';q.mkdir(parents=True)
   (q/'memory.json').write_text(json.dumps(dict(anchor_obs_baseline=json.dumps([['U','F'],'mode',[]]),P_mem=['T','F'],P_age=[1,0],carried_mask=[True,False],transition_types=['CARRY_T','OVERWRITE_T_TO_F'])))
 d.record_update({'actor/pg_loss':.1,'perf/total_num_tokens':123},1)
 row=json.loads((root.parent/'update_metrics.jsonl').read_text())
 assert len(row['task_outcomes'])==2
 assert all(t['num_rollouts']==8 and t['num_success']==1 for t in row['task_outcomes'])
 m=row['memory_diagnostics'];assert m['P_obs_known_ratio']==.5 and m['P_mem_known_ratio']==1
 assert m['num_T_to_F_overwrites']==16 and m['carry_recovery_rate']==1 and m['mean_memory_age']==1
 assert row['tokens']==123
 try:d.record_update({'loss':float('nan')},1)
 except RuntimeError as e:assert 'NONFINITE' in str(e)
 else:raise AssertionError('NaN accepted')
 print('PASS: incomplete update rejected; per-task outcomes; carry/age/overwrite metrics; nonfinite rejected. Synthetic only.')
