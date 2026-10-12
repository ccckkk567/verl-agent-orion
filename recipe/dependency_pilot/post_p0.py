"""Bounded post-P0 runner: real ALF continuations and matched ScienceWorld re-P0."""
import argparse
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import uuid

from .common import ROOT, append_jsonl, digest, file_hash, git_info, read_jsonl, versions, write_json
from .environments import PilotManager, science_provenance
from .instrumentation import Observer
from .post_interface import PROFILES, parser_metadata, execution_outcome

PREFIXES = [
    ('clean_pickup_p1', '636bd564-6758-4fdf-bce0-633499b75de6', 10, 39),
    ('clean_pickup_p2', '06a43d5a-ed78-428a-942e-0e89cdd5a77f', 14, 35),
    ('clean_pickup_p3', 'd67c93ce-fa86-43ee-9c65-e1ecf55c8846', 2, 47),
    ('clean_pickup_p4', '61b55007-57dd-46ae-b192-41e3659d3545', 15, 34),
]
HINTS = {'original': '', 'goal_reminder': 'Reminder: your task is to put a clean cloth in toilet.',
         'clean_subgoal': 'Before putting the cloth in the toilet, make sure the cloth has been cleaned.'}

def fact_tuples(info):
    if not isinstance(info.get('facts'), list): return None
    return sorted((f['name'], tuple((a['name'].strip(), a.get('type')) for a in f['arguments']))
                  for f in info['facts'])

def holding_cloth(info):
    facts = fact_tuples(info)
    if facts is None: return None
    return sorted(args[1][0] for name, args in facts if name == 'holds' and args[1][0].startswith('cloth '))

def canonical_state(obs, info):
    return {'facts': fact_tuples(info), 'observation': obs,
            'admissible_commands': sorted(info.get('admissible_commands', [])),
            'strict_success': info.get('strict_success'), 'native_score': info.get('native_score'),
            'env_moves': info.get('env_moves'), 'task_description': info.get('task_description')}

def prepare_prefixes(source, destination):
    config = json.loads((source/'resolved_config.json').read_text(encoding='utf8'))
    assert config['max_steps'] == 50 and config['env'] == 'alfworld'
    steps = read_jsonl(source/'steps.jsonl')
    episodes = {e['traj_uid']: e for e in read_jsonl(source/'episodes.jsonl')}
    result = []
    for pid, uid, last, remaining in PREFIXES:
        prefix = sorted([s for s in steps if s['traj_uid'] == uid and s['step_index'] <= last], key=lambda s:s['step_index'])
        assert [s['step_index'] for s in prefix] == list(range(last+1))
        assert 50-len(prefix) == remaining and holding_cloth(prefix[-1]['native_info']) == ['cloth 4']
        assert all(not holding_cloth(s['native_info']) for s in prefix[:-1])
        row = {'prefix_id': pid, 'source_run_id': config['run_id'], 'source_traj_uid': uid,
            'prefix_end_step_index': last, 'prefix_steps_consumed': last+1, 'remaining_turns': remaining,
            'task_instance_id': episodes[uid]['task_instance_id'], 'original_cloth_id': 'cloth 4',
            'source_steps_sha256': file_hash(source/'steps.jsonl'), 'source_config_sha256': file_hash(source/'resolved_config.json'),
            'state_fingerprint': digest(json.dumps(canonical_state(prefix[-1]['observation_post'], prefix[-1]['native_info']), sort_keys=True)),
            'history_fingerprint': digest(json.dumps([(s['anchor_pre_raw'],s['parsed_action']) for s in prefix])),
            'replay_status': 'not_run', 'prefix': prefix, 'task': episodes[uid]}
        result.append(row)
        append_jsonl(destination/'prefix_manifest.jsonl', {k:v for k,v in row.items() if k not in ['prefix','task']})
    return result

class AlfPrefixManager(PilotManager):
    def __init__(self, task, config, prefix):
        super().__init__(task, config)
        self.prefix, self.ready = prefix, False

    def reset(self, kwargs=None):
        super().reset(kwargs)
        checks = []
        for i, session in enumerate(self.sessions):
            for source in self.prefix['prefix']:
                before = self.text[i]
                action = source['parsed_action']  # Actual executed command, never reparse thought text.
                obs, reward, done, info = session.step(action)
                self.history[i].append((before, action))
                self.text[i], self.infos[i], self.done[i] = obs, info, bool(done)
                state = canonical_state(obs, info)
                expected = canonical_state(source['observation_post'], source['native_info'])
                check = {'record_kind': 'replay', 'prefix_id': self.prefix['prefix_id'], 'arm': self.config['condition'],
                    'replicate': i, 'source_step_id': source['analysis_step_id'], 'parsed_action': action,
                    'observation': obs, 'info': info, 'done': bool(done),
                    'fields_agree': {k: state[k] == expected[k] for k in state},
                    'facts_available': state['facts'] is not None}
                append_jsonl(Path(self.config['run_dir'])/'replay_checks.jsonl', check)
                checks.append(check)
                if done or reward != 0: raise RuntimeError('Replay unexpectedly reached an endpoint')
            observed = digest(json.dumps(canonical_state(self.text[i],self.infos[i]),sort_keys=True))
            history = digest(json.dumps(self.history[i]))
            if observed != self.prefix['state_fingerprint'] or history != self.prefix['history_fingerprint']:
                raise RuntimeError('Prefix final state/history mismatch; excluded from matched P1')
            if holding_cloth(self.infos[i]) != ['cloth 4']: raise RuntimeError('Prefix no longer holds original cloth')
        if not all(c['facts_available'] and all(c['fields_agree'].values()) for c in checks):
            raise RuntimeError('Replay differs from recorded facts/feedback/menu; excluded from matched P1')
        self.ready = True
        return self.observations(), self.infos

    def observations(self):
        obs = super().observations()
        hint = HINTS[self.config['condition']]
        if self.ready and hint:
            for i in range(len(obs['text'])):
                if len(self.history[i]) == self.prefix['prefix_steps_consumed']:
                    obs['text'][i] += '\n' + hint
        return obs

class PostObserver(Observer):
    def start(self, obs, infos, group_ids, traj_ids):
        super().start(obs, infos, group_ids, traj_ids)
        self.previous_info = copy.deepcopy(infos)
        self.last_done = [False]*len(infos)
        for i, ep in enumerate(self.episodes):
            ep.update({k:self.config.get(k) for k in ['phase','execution_kind','prompt_profile','prefix_id','source_traj_uid','history_offset','condition']})
            ep['policy_updated'] = False
            ep['prefix_steps_consumed'] = self.config.get('history_offset',0)
            ep['remaining_turns'] = self.config['max_steps']
            ep['continuation_seed'] = 10000+i if self.config['env'] == 'alfworld' else None
            ep['replay_status'] = 'recorded_state_and_history_verified' if self.config['env'] == 'alfworld' else None
            ep['replay_scope'] = 'all recorded typed facts, public feedback, menu, moves, history; no hidden RNG snapshot'
            ep['initial_native_score'] = infos[i].get('native_score')
            ep['initial_look'] = infos[i].get('look')

    def before_step(self, step, batch, obs, active):
        ids = super().before_step(step,batch,obs,active)
        for i, row in enumerate(self.pending):
            offset = self.config.get('history_offset',0)
            hint = HINTS.get(self.config.get('condition'),'') if step == 0 else ''
            actual = self.tokenizer.decode(row['actual_input_ids'],skip_special_tokens=False)
            interface = self.previous_info[i].get('interface_metadata',{})
            row.update(record_kind='continuation' if offset else 'rollout', absolute_step_index=offset+step,
                suffix_step_index=step if offset else None, condition=self.config.get('condition'),
                prefix_id=self.config.get('prefix_id'), continuation_seed=10000+i if offset else None,
                generation_seed=self.config['seed']+i+100003*step,
                hint_text=hint, hint_scope='first_continuation_only' if offset else None,
                hint_visible_first_step=(hint in actual) if hint else None,
                actor_prompt_before_hint=obs['text'][i][:-len('\n'+hint)] if hint else obs['text'][i],
                native_info_pre=copy.deepcopy(self.previous_info[i]),
                interface_metadata=interface,
                interface_block_token_visible=interface.get('actor_interface_block','') in actual if interface else None)
            if active[i] and hint and hint not in actual:
                raise RuntimeError('First-decision hint was truncated; cannot enter matched analysis')
        return ids

    def after_step(self, next_obs, rewards, dones, infos):
        for i, row in enumerate(self.pending):
            action = infos[i].get('parsed_action','')
            row.update(parser_metadata(row.get('raw_response',''),action,self.config['env']))
            row['format_compliance'] = infos[i].get('parse_valid')
            row['execution_outcome'] = execution_outcome(action,next_obs['anchor'][i],
                self.previous_info[i].get('electrical_observer'),infos[i].get('electrical_observer')) if self.config['env']=='scienceworld' else 'unknown'
        super().after_step(next_obs,rewards,dones,infos)
        self.previous_info = copy.deepcopy(infos)
        self.last_done = list(map(bool,dones))

    def finish(self,error=None):
        for i,ep in enumerate(self.episodes):
            info=self.previous_info[i]
            ep.update(outer_truncated=ep['agent_turns']>=self.config['max_steps'],
                native_budget_truncated=self.last_done[i] and info.get('termination_reason')=='budget_truncated',
                native_completed=info.get('native_completed',info.get('won')),
                label_known=error is None and info.get('strict_success') is not None,
                unknown_reason=error, native_score_scale='0-100 (failure -100)' if self.config['env']=='scienceworld' else 'native TextWorld score')
        super().finish(error)

def collector_batch(actor, manager, observer, config):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    n = config['rollouts_per_task']
    actor.config, actor.observer = config, observer
    collector = TrajectoryCollector(OmegaConf.create({'data':{'max_prompt_length':config['prompt_tokens'],'truncation':'left','return_raw_chat':True},
                    'env':{'rollout':{'n':n},'max_steps':config['max_steps']}}),actor.tokenizer)
    collector.pilot_observer = observer
    raw = np.empty(n,dtype=object)
    for i in range(n): raw[i]=[{'role':'user','content':''}]
    batch = DataProto.from_single_dict({'input_ids':torch.zeros((n,1),dtype=torch.long),'raw_prompt':raw,
                'data_source':np.array([config['env']]*n,dtype=object)})
    try: collector.vanilla_multi_turn_loop(batch,actor,manager)
    except Exception as exc:
        observer.finish(error=f'{type(exc).__name__}: {exc}')
        raise
    finally: manager.close()

def analyze(out):
    from .post_analysis import analyze_run
    analyze_run(out)
    # Fresh process prevents Torch/Matplotlib OpenMP collisions on Windows.
    subprocess.run([sys.executable,'-m','recipe.dependency_pilot.post_analysis','plot','--run-dir',str(out)],check=True)

def run_alf(out, config, source, prefix_ids):
    prefixes = prepare_prefixes(source,out)
    order = ['clean_pickup_p3','clean_pickup_p1','clean_pickup_p2','clean_pickup_p4']
    prefixes.sort(key=lambda p:order.index(p['prefix_id']))
    if prefix_ids: prefixes = [p for p in prefixes if p['prefix_id'] in prefix_ids]
    from .runtime import TransformersWorker
    actor = TransformersWorker(config)
    schedule = []
    for prefix in prefixes:
        arms = list(HINTS);random.Random(20261012+order.index(prefix['prefix_id'])).shuffle(arms)
        schedule.extend((prefix,arm) for arm in arms)
    write_json(out/'schedule.json', [{'prefix_id':p['prefix_id'],'arm':a,'seeds':[10000,10001]} for p,a in schedule])
    errors = []
    for batch, (prefix,arm) in enumerate(schedule):
        child = {**config,'condition':arm,'prefix_id':prefix['prefix_id'],'source_traj_uid':prefix['source_traj_uid'],
            'history_offset':prefix['prefix_steps_consumed'],'max_steps':prefix['remaining_turns'],
            'seed':10000+100003*prefix['prefix_steps_consumed'],'rollouts_per_task':2}
        task = prefix['task']
        if file_hash(task['source_path']) != task['source_hash']:
            raise RuntimeError('ALFWorld task asset differs from source P0')
        # Environment reset/step budget/seed remains the source protocol; generation seed is independent.
        manager = AlfPrefixManager(task,{**child,'max_steps':50,'seed':0},prefix)
        observer = PostObserver(out,child,task,actor.tokenizer,f"{config['run_id']}:{prefix['prefix_id']}:{arm}:batch0")
        group_ok=True
        try: collector_batch(actor,manager,observer,child)
        except Exception as exc:
            group_ok=False
            errors.append(f'{prefix["prefix_id"]}/{arm}: {exc}')
            append_jsonl(out/'errors.jsonl',{'prefix_id':prefix['prefix_id'],'arm':arm,'error':str(exc),'traceback':traceback.format_exc()})
        for ep in observer.episodes:
            append_jsonl(out/'continuations.jsonl',{**ep,'arm':arm,'hint_text':HINTS[arm],
                'matched_pair_id':f"{prefix['prefix_id']}:{ep['continuation_seed']}",
                'hint_scope':'first_continuation_only'})
        analyze(out)
        print(f"{'COMPLETED' if group_ok else 'FAILED'}_P1_GROUP {batch+1}/{len(schedule)} {prefix['prefix_id']} {arm}",flush=True)
    return errors

def run_science(out, config, source):
    tasks = read_jsonl(source/'tasks_manifest.jsonl')
    assert {t['task_instance_id'] for t in tasks} == {'power-component/7','power-component-renewable-vs-nonrenewable-energy/5'}
    write_json(out/'environment_provenance.json',science_provenance())
    for task in tasks: append_jsonl(out/'tasks_manifest.jsonl',task)
    if config['execution_kind'] == 'science_smoke': tasks = sorted(tasks,key=lambda t:t['variation_id'])
    from .runtime import TransformersWorker
    actor = TransformersWorker(config)
    from .run import save_config
    refs=Path(__file__).with_name('post_p0_source_refs.json')
    config['interface_source_refs']=json.loads(refs.read_text(encoding='utf8'))
    write_json(out/'source_provenance.json',config['interface_source_refs'])
    save_config(out,config)
    for batch, task in enumerate(tasks):
        manager = PilotManager(task,config)
        observer = PostObserver(out,config,task,actor.tokenizer,f"{config['run_id']}:batch{batch}")
        collector_batch(actor,manager,observer,config)
        analyze(out)
        print(f"COMPLETED_SCI_GROUP {batch+1}/2 {task['task_instance_id']}",flush=True)
    return []

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['prepare-prefixes','alf-p1','science-smoke','science-rep0','analyze'])
    p.add_argument('--source-run',type=Path)
    p.add_argument('--output-root',type=Path,default=Path(os.environ.get('PILOT_OUTPUT_ROOT',str(ROOT/'outputs/dependency_pilot'))))
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--model-path',default=os.environ.get('MODEL_PATH'))
    p.add_argument('--prompt-profile',choices=PROFILES[1:],default=PROFILES[1])
    p.add_argument('--prefix-ids',nargs='+',choices=[p[0] for p in PREFIXES])
    p.add_argument('--swanlab-mode',choices=['cloud','offline','disabled'],default='cloud')
    p.add_argument('--lock-file',type=Path,help='Required for re-P0; produced by lock-interface after real smoke')
    args=p.parse_args()
    if args.command=='analyze': analyze(args.run_dir);return
    if not args.source_run: p.error('--source-run is required')
    source=args.source_run.resolve()
    kind={'alf-p1':'alfworld-p1','science-smoke':'scienceworld-interface-smoke','science-rep0':'scienceworld-rep0','prepare-prefixes':'alfworld-prefixes'}[args.command]
    run_id=kind+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    out=args.run_dir or args.output_root/run_id
    out.mkdir(parents=True,exist_ok=False)
    if args.command=='prepare-prefixes': prepare_prefixes(source,out);print(out);return
    old=json.loads((source/'resolved_config.json').read_text(encoding='utf8'))
    env='alfworld' if args.command=='alf-p1' else 'scienceworld'
    assert old['env']==env
    config={**old,'run_id':run_id,'run_dir':str(out.resolve()),'phase':'post_p0','policy_updated':False,'training_updates':0,
        'parent_run_ids':[old['run_id']],'comparison_set_id':'post-p0-20261012','execution_kind':args.command.replace('-','_'),
        'swanlab_mode':args.swanlab_mode,'prompt_profile':args.prompt_profile if env=='scienceworld' else 'p0_original',
        'parser_profile':'p0_original_unchanged','max_total_episodes':24 if env=='alfworld' else 8,
        'rollouts_per_task':2 if env=='alfworld' else (1 if args.command=='science-smoke' else 4),
        'max_steps':12 if args.command=='science-smoke' else 50,
        'model_path':args.model_path or old['model_path'],'seed':0,'condition':'root_original',
        'seed_rule':'P1: 10000 + rollout_index + 100003 * absolute_step_index; P0: original seed rule',
        'source_steps_sha256':file_hash(source/'steps.jsonl'),'prefix_ids_requested':args.prefix_ids}
    if args.command=='science-smoke': config['max_total_episodes']=2
    if args.command=='alf-p1' and args.prefix_ids:config['max_total_episodes']=len(args.prefix_ids)*6
    if args.command=='science-rep0':
        if not args.lock_file: raise ValueError('Full re-P0 requires --lock-file after smoke')
        lock=json.loads(args.lock_file.read_text(encoding='utf8'))
        if lock['selected_profile']!=args.prompt_profile or lock['status']!='locked_after_real_smoke': raise ValueError('Lock/profile mismatch')
        if lock['jar_sha256']!=science_provenance()['jar_sha256']: raise ValueError('Smoke and re-P0 JAR mismatch')
        config['interface_lock']=lock
        from .post_audit import MATCHED_PROTOCOL
        if any(config.get(k)!=v for k,v in MATCHED_PROTOCOL.items()):raise ValueError('Full re-P0 protocol differs from locked 8 x 50 matched protocol')
    from .run import save_config
    save_config(out,config)
    manifest={'run_id':run_id,'git':git_info(),'versions':versions(),'command':sys.argv,'status':'running',
              'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'policy_updated':False,'phase':'post_p0','execution_kind':config['execution_kind']}
    write_json(out/'run_manifest.json',manifest)
    print(f'RUN_DIR={out} EPISODES={config["max_total_episodes"]} MAX_STEPS={config["max_steps"]} RESPONSE_TOKENS={config["response_tokens"]}',flush=True)
    try:
        errors=run_alf(out,config,source,args.prefix_ids) if env=='alfworld' else run_science(out,config,source)
        manifest['status']=('ran_partial' if read_jsonl(out/'steps.jsonl') else 'blocked') if errors else 'completed_exploratory'
        manifest['errors']=errors
    except Exception as exc:
        manifest.update(status='ran_partial' if read_jsonl(out/'steps.jsonl') else 'blocked',error=str(exc))
        append_jsonl(out/'errors.jsonl',{'error':str(exc),'traceback':traceback.format_exc()})
        traceback.print_exc()
    finally:
        write_json(out/'run_manifest.json',manifest)
        try: analyze(out)
        except Exception as exc: write_json(out/'analysis_error.json',{'error':str(exc)})
        # Tracking is a separate process. Raw facts/prompts remain local.
        tracked=subprocess.run([sys.executable,'-m','recipe.dependency_pilot.post_analysis','track','--run-dir',str(out)])
        if tracked.returncode: write_json(out/'tracking_error.json',{'exit_code':tracked.returncode,'local_results_preserved':True})
    print(f'RESULT={manifest["status"]} RUN_DIR={out}',flush=True)
    return 0 if manifest['status']=='completed_exploratory' else 1

if __name__=='__main__': raise SystemExit(main())
