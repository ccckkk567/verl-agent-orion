"""Apply small, fail-closed integration patches only in the dedicated frozen repo copy."""
from pathlib import Path
import os,json,hashlib
r=Path(os.environ['PILOT_ROOT']);repo=r/'repo';changes=[]
def patch(rel,old,new):
 p=repo/rel;s=p.read_text();assert s.count(old)==1,(rel,s.count(old));before=hashlib.sha256(p.read_bytes()).hexdigest();p.write_text(s.replace(old,new));changes.append(dict(file=rel,before=before,after=hashlib.sha256(p.read_bytes()).hexdigest()))
patch('verl/trainer/main_ppo.py','from agent_system.environments import make_envs','from collector_bridge import make_envs')
patch('verl/trainer/main_ppo.py','from agent_system.multi_turn_rollout import TrajectoryCollector','from collector_bridge import AndroidCollector as TrajectoryCollector')
patch('agent_system/multi_turn_rollout/rollout_loop.py','''chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])''','''chat = np.array(obs['chat'][item]) if obs.get('chat') is not None else np.array([{
            "content": obs_content,
            "role": "user",
        }])''')
(r/'reports/integration_patches.json').write_text(json.dumps(changes,indent=2))
patch('gigpo/core_gigpo.py','''scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores''','''scores = episode_advantages + step_advantage_w * step_advantages
    from diagnostics import record_advantages
    record_advantages(episode_advantages, step_advantages, scores, response_mask, step_group_uids, index, traj_index, step_rewards, anchor_obs)
    return scores, scores''')
patch('verl/trainer/ppo/ray_trainer.py','''                logger.log(data=metrics, step=self.global_steps)''','''                from diagnostics import record_update
                record_update(metrics, self.global_steps)
                logger.log(data=metrics, step=self.global_steps)''')
(r/'reports/integration_patches.json').write_text(json.dumps(changes,indent=2))

patch('verl/trainer/ppo/ray_trainer.py', '    elif adv_estimator == AdvantageEstimator.GiGPO:\n', '    elif adv_estimator == AdvantageEstimator.GiGPO:\n        from diagnostics import set_batch\n        set_batch(data)\n')
(r/'reports/integration_patches.json').write_text(json.dumps(changes,indent=2))

# Fix adapter initialization identically across both arms before any update.
patch('verl/workers/fsdp_workers.py', '                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))', '                from transformers import set_seed\n                set_seed(2026092700)\n                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))')
(r/'reports/integration_patches.json').write_text(json.dumps(changes,indent=2))
