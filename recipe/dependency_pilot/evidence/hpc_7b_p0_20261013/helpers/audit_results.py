"""Read-only audit of recorded 7B P0. Never generate, re-execute or repair actions."""
import argparse,json,re,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser(__doc__)
    for arg in ['repo','alf-old','science-old','alf-new','science-new','tokenizer','output']:p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.repo))
    from recipe.dependency_pilot.environments import parse_action
    from recipe.dependency_pilot.post_analysis import table
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    runs={'alfworld':(args.alf_old,args.alf_new),'scienceworld':(args.science_old,args.science_new)}
    audit={};counts=[];actions=[];case_rows=[];rawlabels=[]
    for env,(old,new) in runs.items():
        cfg=json.loads((new/'resolved_config.json').read_text());ident=json.loads((new/'model_identity.json').read_text())
        for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
            assert sha(args.tokenizer/name)==ident['file_hashes'][name],name
        for name,expected in json.loads((new/'implementation_snapshot/hashes.json').read_text()).items():
            assert sha(new/'implementation_snapshot'/name)==expected,name
        steps=read(new/'steps.jsonl');eps=read(new/'episodes.jsonl');resets=read(new/'resets.jsonl')
        old_steps=read(old/'steps.jsonl');old_eps=read(old/'episodes.jsonl');old_resets=read(old/'resets.jsonl')
        assert len(eps)==8 and len(steps)==400 and len({s['analysis_step_id'] for s in steps})==400
        assert all(e['status']=='complete' and e['agent_turns']==50 for e in eps)
        assert all(s['active_mask'] and s['env_reward']==0 for s in steps)
        assert {e['checkpoint_id'] for e in eps}=={ident['checkpoint_id']}
        assert ident['actual_num_parameters']==7615616512 and cfg['policy_updated'] is False and cfg['training_updates']==0
        assert cfg['source_steps_sha256']==sha(old/'steps.jsonl')
        assert cfg['source_config_sha256']==sha(old/'resolved_config.json')
        assert cfg['source_tasks_sha256']==sha(old/'tasks_manifest.jsonl')
        epmap={e['traj_uid']:e for e in eps};old_epmap={e['traj_uid']:e for e in old_eps}
        old_first={s['task_instance_id']:s for s in old_steps if s['step_index']==0}
        old_reset={old_epmap[r['traj_uid']]['task_instance_id']:r for r in old_resets}
        group_members=defaultdict(set)
        reset_equal=[];prompt_equal=[]
        for r in resets:
            task=epmap[r['traj_uid']]['task_instance_id'];before=old_reset[task]
            reset_equal.append(r['observation']==before['observation'])
        for s in steps:
            uid=s['traj_uid'];group_members[s['episode_group_uid']].add(uid)
            assert s['generation_seed']==epmap[uid]['seed']+100003*s['step_index']
            assert not s.get('hint_text') and s.get('record_kind')=='rollout'
            decoded=tokenizer.decode(s['response_token_ids'],skip_special_tokens=True)
            assert decoded==s['raw_response'],s['analysis_step_id']
            actual=tokenizer.decode(s['actual_input_ids'],skip_special_tokens=False)
            assert actual==s['rendered_prompt'],s['analysis_step_id']
            assert s['truncated_tokens']==0 and not s['output_at_token_limit']
            a,valid=parse_action(s['raw_response'],env)
            assert (a,valid)==(s['parsed_action'],s['parse_valid'])
            if s['step_index']==0:
                prompt_equal.append(s['actor_prompt']==old_first[s['task_instance_id']]['actor_prompt'])
            raw=s['raw_response']
            label='complete_xml_action' if re.search(r'<action>.*?</action>',raw,re.S|re.I) else ('malformed_square_action_marker' if re.search(r'\[action\b',raw,re.I) else 'other_missing_or_incomplete_action_tags')
            rawlabels.append({'run_id':s['run_id'],'analysis_step_id':s['analysis_step_id'],'task_instance_id':s['task_instance_id'],
                'tag_category':label,'executed_command':a,'feedback':s['observation_post']})
        assert len(group_members)==2 and all(len(v)==4 for v in group_members.values())
        assert all(reset_equal) and all(prompt_equal)
        audit[env]={'run_id':cfg['run_id'],'episodes':8,'active_steps':400,'groups':2,'group_sizes':[len(v) for v in group_members.values()],
            'reset_observation_matches_baseline':sum(reset_equal),'first_actor_prompt_matches_baseline':sum(prompt_equal),
            'decoded_raw_responses_match':400,'decoded_inputs_match_full_rendered_prompts':400,
            'parser_commands_and_flags_match':400,'same_hash_tokenizer_used_for_decode':True,'policy_updated':False,'training_updates':0,
            'input_truncation_count':0,'response_limit_count':0,'model_identity':ident,'source_steps_sha256':sha(new/'steps.jsonl')}
        for version,path,ss in [('1.5B',old,old_steps),('7B',new,steps)]:
            buckets=defaultdict(list)
            for s in ss:
                if s['active_mask']:buckets[s['task_instance_id']].append(s)
            for task,rs in buckets.items():
                cats=Counter('complete_xml_action' if re.search(r'<action>.*?</action>',s['raw_response'],re.S|re.I) else ('malformed_square_action_marker' if re.search(r'\[action\b',s['raw_response'],re.I) else 'other_missing_or_incomplete_action_tags') for s in rs)
                counts.append({'model':version,'env':env,'task':task,'run_id':rs[0]['run_id'],'active_steps':len(rs),**{k:cats[k] for k in ['complete_xml_action','malformed_square_action_marker','other_missing_or_incomplete_action_tags']},
                    'format_compliant':sum(s['parse_valid'] for s in rs),'nothing_happens':sum(s['observation_post'].strip()=='Nothing happens.' for s in rs),
                    'no_known_action':sum(s['observation_post'].strip()=='No known action matches that input.' for s in rs),
                    'input_truncated':sum(s['truncated_tokens']>0 for s in rs),'response_at_limit':sum(s['output_at_token_limit'] for s in rs)})
                for (command,feedback),n in Counter((s['parsed_action'],s['observation_post']) for s in rs).most_common(12):
                    actions.append({'model':version,'env':env,'task':task,'command':command,'feedback':feedback,'count':n})
        chosen = ({'3cefd220-82ee-4092-81dd-3b86fc24a65e':range(0,4),
                   'f3bb8447-f3f6-445d-82fb-a338abfe2f9a':range(25,30),
                   '77d4edc4-4333-4afa-b993-7cfc6439740f':range(7,12)} if env=='alfworld' else
                  {'afadbee7-1282-4047-bfeb-01a1e2cbcce2':range(0,5),
                   '5e733fef-3c49-4242-947b-ab4d96dcf385':range(0,5)})
        for s in steps:
            if s['step_index'] in chosen.get(s['traj_uid'],[]):
                case_rows.append({k:s.get(k) for k in ['run_id','analysis_step_id','traj_uid','step_index','task_instance_id','actor_prompt','raw_response','parsed_action','observation_post','native_score_pre','native_score_post','native_info_pre','native_info']})
    save(args.output/'audit.json',audit)
    table(args.output/'format_failure_counts.csv',counts);save(args.output/'format_failure_counts.json',counts)
    table(args.output/'frequent_executed_commands.csv',actions)
    for name,rows in [('format_annotations.jsonl',rawlabels),('case_steps.jsonl',case_rows)]:
        (args.output/name).write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in rows),encoding='utf8')
    (args.output/'case_steps.md').write_text('# 实际轨迹案例（轮次为零起算 step_index + 1）\n\n'+
        '\n\n'.join(f"## {s['analysis_step_id']}\n\nRun: `{s['run_id']}`\n\n原始回答：\n```text\n{s['raw_response']}\n```\n\n实际执行：\n```text\n{s['parsed_action']}\n```\n\n环境反馈：\n```text\n{s['observation_post']}\n```" for s in case_rows),encoding='utf8')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
