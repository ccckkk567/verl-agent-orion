"""Serial environment transport around official TrajectoryCollector; no trainer/objective replacement."""
import os,json,subprocess,uuid,copy
from pathlib import Path
import numpy as np
from PIL import Image
from anchor_bridge import AnchorBridge
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
ROOT=Path(os.environ['PILOT_ROOT']);ARM=os.environ['PILOT_ARM']
TASKS=['3256fd1e1c9855d3','dd820aeaef5c4c22']
class SerialAndroid:
 def __init__(self,config):
  self.config=config;self.bridge=AnchorBridge(ARM);self.proc=None;self.update=0
 def rpc(self,request):
  if self.proc is None:
   log=(ROOT/'logs'/f'android_{ARM}.log').open('a')
   self.proc=subprocess.Popen(['bash','/mnt/data/cjk/qualification_v1_20260926/scripts/run_env.sh',str(ROOT/'code/android_rpc.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
  self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
  while True:
   line=self.proc.stdout.readline()
   if not line:raise RuntimeError('Android RPC exited')
   if line.startswith('PILOT_RPC '):
    msg=json.loads(line[len('PILOT_RPC '):]);assert msg['ok'],msg.get('error');return msg['result']
 def pack(self,o):
  if o['step_idx']==0:self.bridge.reset(self.tid,o['task_family'],o['anchor']['P_names'])
  mem=self.bridge.observe(self.tid,o['step_idx'],o['anchor'])
  d=self.dest/'steps'/f"{o['step_idx']:03d}"
  (d/'memory.json').write_text(json.dumps(mem,indent=2))
  self.current=o
  return dict(text=[o['messages'][1]['content']],chat=[o['messages']],image=[np.asarray(Image.open(o['image']).convert('RGB'))],anchor=[mem['anchor_obs']])
 def reset(self,kwargs=None):
  self.dest=ROOT/'runs'/ARM/f'update_{self.update:04d}'/self.tid
  return self.pack(self.rpc(dict(op='reset',task_instance_id=self.task_id,destination=str(self.dest),seed=self.seed))),[{}]
 def step(self,text_actions):
  assert len(text_actions)==1
  out=self.rpc(dict(op='step',text=text_actions[0]));info=dict(is_action_valid=out['is_action_valid'])
  if out['done']:
   self.bridge.finish(self.tid);self.outcome=out['outcome'];obs=dict(text=[''],chat=[[dict(role='user',content='')]],image=None,anchor=['TERMINAL_UNUSED'])
  else:obs=self.pack(out['observation'])
  return obs,np.array([out['reward']]),np.array([out['done']]),[info]
 def success_evaluator(self,**kwargs):return {'success_rate':np.array([float(self.outcome['success'])])}
 def close(self):
  if self.proc and self.proc.poll() is None:self.proc.terminate()

class AndroidCollector(TrajectoryCollector):
 def vanilla_multi_turn_loop(self,gen_batch,actor_rollout_wg,envs):
  # Official multi_turn_loop repeats 2 tasks eight times. Run serially to avoid 16 emulator copies.
  assert len(gen_batch.batch)==16
  envs.update+=1
  lists=[];rewards=[];lengths=[];ids=[];calls=[];success=[]
  for i in range(16):
   envs.task_id=TASKS[i//8];envs.tid=f'{envs.task_id}_r{i%8}';envs.seed=2026092700+(envs.update-1)*100+i
   one=gen_batch[i:i+1]
   result=super().vanilla_multi_turn_loop(one,actor_rollout_wg,envs)
   steps,rew,lens,succ,uid,tool=result
   # Re-establish exact task/update scope across serial invocations; trajectory UID stays unique.
   for j,record in enumerate(steps[0]):
    record['uid']=f'{ARM}:{envs.update}:{envs.task_id}'
    raw=json.loads((envs.dest/'steps'/f'{j:03d}'/'record.json').read_text())
    mem=json.loads((envs.dest/'steps'/f'{j:03d}'/'memory.json').read_text())
    record['pilot_trace']=json.dumps(dict(task_instance_id=envs.task_id,trajectory_id=envs.tid,step_idx=j,normalized_action=raw['normalized_action'],P_obs=raw['P'],P_names=raw['P_names'],**mem))
   lists.extend(steps);rewards.extend(rew);lengths.extend(lens);ids.extend(uid);calls.extend(tool);success.extend(succ['success_rate'])
  return lists,np.asarray(rewards),np.asarray(lengths),{'success_rate':np.asarray(success)},np.asarray(ids,dtype=object),np.asarray(calls)

def make_envs(config):
 env=SerialAndroid(config)
 return env,env
