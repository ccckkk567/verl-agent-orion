"""Serial AndroidWorld environment process using the frozen qualification runtime."""
import os,sys,json,random,traceback
from pathlib import Path
import numpy as np
Q=Path('/mnt/data/cjk/qualification_v1_20260926')
sys.path.insert(0,str(Q/'src'))
import runtime_env
from android_world import registry,suite_utils
from android_world.env import env_launcher,adb_utils
from reset_audit import snapshot,task_audit
from params_codec import decode
from observation import capture,dump
from policy import parse_syntax,validate_text,execute
from mobile_actor_reference import SYSTEM
config=json.loads((Q/'configs/qualification_tasks_v1.json').read_text())
reg=registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
env=None;task=None

def observation():
 global state,anchor
 d=dest/'steps'/f'{step:03d}'
 public={k:row[k] for k in ('task_family','task_instance_id','public_params','instruction')}
 state,anchor=capture(env,d,public)
 messages=[dict(role='system',content=SYSTEM),dict(role='user',content='<image>\nTask: '+row['instruction']+'\nRemaining decisions: '+str(row['max_steps']-step)+'\nPast attempted actions (not confirmed facts): '+json.dumps(history,ensure_ascii=False))]
 dump(d/'policy_messages.json',messages)
 return dict(anchor=anchor,messages=messages,image=str(d/'screenshot.png'),step_idx=step,task_family=row['task_family'],task_instance_id=row['task_instance_id'])

def reset(x):
 global env,task,row,dest,step,history,records,seed
 row=next(r for r in config['tasks'] if r['task_instance_id']==x['task_instance_id']);dest=Path(x['destination']);assert not dest.exists();dest.mkdir(parents=True)
 if task is not None:task.tear_down(env)
 snapshot('load','cjk_guiowl_clean_v2')
 if env is None:env=env_launcher.load_and_setup_env(console_port=5564,grpc_port=8564,adb_path='/mnt/data/cjk/runtime_androidworld_v1/android/platform-tools/adb',emulator_setup=False)
 else:env.controller.refresh_env()
 random.seed(row['environment_seed']);np.random.seed(row['environment_seed'])
 task=suite_utils._instantiate_task(reg[row['task_family']],params=decode(json.loads((Q/row['private_params_path']).read_text())),env=env)
 task.initialize_task(env)
 baseline=json.loads((Q/'manifests/baselines'/f"{row['task_instance_id']}.json").read_text())
 snapshot('load',baseline['snapshot']);env.controller.refresh_env();env.interaction_cache='';env.reset(go_home=task.start_on_home_screen)
 audit=task_audit(task,env,dest/'audit_only');assert audit['digest']==baseline['digest'],'RESET_BASELINE_MISMATCH'
 dump(dest/'reset.json',dict(expected=baseline['digest'],actual=audit['digest'],matched=True))
 snapshot('load',baseline['snapshot']);env.controller.refresh_env();env.interaction_cache='';env.reset(go_home=task.start_on_home_screen)
 step=0;history=[];records=[];seed=x['seed'];return observation()

def act(x):
 global step
 text=x['text'];record=dict(step_idx=step,**anchor,raw_action=text,parse_success=False,action_valid=False,normalized_action=None)
 action=None
 try:
  record['action_parsed']=parse_syntax(text);record['parse_success']=True
  action=validate_text(text);record['action_valid']=True;record['normalized_action']=action
 except (ValueError,AssertionError,TypeError,KeyError) as e:record['action_error']=repr(e)
 if action is not None:
  record['execution']=execute(env,action,state.pixels.shape);history.append(dict(attempted_action=action))
 else:history.append(dict(invalid_attempt=text[-1500:]))
 records.append(record);dump(dest/'steps'/f'{step:03d}'/'record.json',record);step+=1
 done=step>=row['max_steps'] or (action is not None and action['action']=='terminate')
 reward=float(task.is_successful(env)) if done else 0.0
 if done:
  assert reward in (0,1)
  outcome=dict(success=bool(reward),episode_reward=reward,num_steps=step,seed=seed,termination_reason='MODEL_TERMINATE' if action and action['action']=='terminate' else 'STEP_LIMIT',parse_rate=sum(r['parse_success'] for r in records)/step,valid_action_rate=sum(r['action_valid'] for r in records)/step)
  dump(dest/'outcome.json',outcome)
  return dict(done=True,reward=reward,is_action_valid=record['action_valid'],outcome=outcome)
 return dict(done=False,reward=0.0,is_action_valid=record['action_valid'],observation=observation())
for line in sys.stdin:
 try:
  x=json.loads(line);result=reset(x) if x['op']=='reset' else act(x)
  print('PILOT_RPC '+json.dumps(dict(ok=True,result=result)),flush=True)
 except Exception:
  print('PILOT_RPC '+json.dumps(dict(ok=False,error=traceback.format_exc())),flush=True);break
