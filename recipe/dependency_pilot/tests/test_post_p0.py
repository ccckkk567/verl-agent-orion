"""Small synthetic fixtures protect interpretation; never reported as experiment results."""
from recipe.dependency_pilot.post_p0 import PREFIXES,HINTS,fact_tuples,holding_cloth
from recipe.dependency_pilot.post_analysis import cloth_state,rate,ever,annotate
from recipe.dependency_pilot.post_interface import parser_metadata,execution_outcome,electrical_snapshot

def prop(name,*args):
    return {'name':name,'arguments':[{'name':a,'type':'object'} for a in args]}

def test_fixed_prefix_budget_and_non_command_hints():
    assert [p[3] for p in PREFIXES]==[39,35,47,34]
    assert all(p[2]+1+p[3]==50 for p in PREFIXES)
    assert list(HINTS)==['original','goal_reminder','clean_subgoal']
    assert not any(s in HINTS['clean_subgoal'] for s in ['sink','1','2','<action>'])

def test_native_facts_cleanable_does_not_mean_clean():
    info={'facts':[prop('cleanable','cloth 4'),prop('holds','agent1','cloth 4')]}
    assert holding_cloth(info)==['cloth 4']
    assert cloth_state(info)['clean']==set()
    info['facts'].append(prop('isClean','cloth 4'))
    assert cloth_state(info)['clean']=={'cloth 4'}
    assert cloth_state({}) is None

def test_fact_order_and_types():
    import copy
    a={'facts':[prop('holds','agent1','cloth 4'),prop('cleanable','cloth 4')]}
    b={'facts':copy.deepcopy(list(reversed(a['facts'])))}
    assert fact_tuples(a)==fact_tuples(b)
    b['facts'][0]['arguments'][0]['type']='changed'
    assert fact_tuples(a)!=fact_tuples(b)

def test_unknown_denominators_and_censoring():
    assert rate([True,False,None])=={'value':.5,'numerator':1,'denominator':2,'unknown_count':1}
    assert rate([])['value'] is None
    assert ever([False,None]) is None and ever([True,None]) is True

def test_parser_metadata_separates_language_format_from_command():
    raw='<think>检查</think><action>look around</action>'
    assert parser_metadata(raw,'look around','scienceworld')['parser_usable']
    assert not parser_metadata('<think>look around</think>','raw fallback','scienceworld')['parser_usable']
    assert parser_metadata('<think>x</think>','raw fallback','scienceworld')['parser_fallback_used']
    multi=parser_metadata('<action>a</action><action>b</action>','a','scienceworld')
    assert multi['parser_usable'] and not multi['response_action_structure_unambiguous']

def test_native_connection_not_focus_or_repeat():
    assert execution_outcome('connect a to b','No known action matches that input.')=='unmatched_command'
    assert execution_outcome('connect a to b','already connected, must be disconnected')=='precondition_blocked'
    r={'run_id':'synth','analysis_step_id':'t:0','traj_uid':'t','step_index':0,'env':'scienceworld',
       'native_info':{'native_score':38,'look':'workshop'},'parsed_action':'focus on green light bulb',
       'raw_response':'<action>focus on green light bulb</action>','observation_post':'You focus on the green light bulb.',
       'anchor_pre_raw':'No known action matches that input.'}
    e=annotate(r,{})
    assert e['target_focused'] and not e['connect_success'] and e['generic_pre_anchor']
    assert e['connection_newly_established'] is None

def test_read_only_edges_normalized():
    class E:
        def getObjectTree(self):return {'uuid':0,'name':'root','contents':{
            'a':{'uuid':1,'name':'terminal','propElectricalConnection':{'connectedTo':['2']},'contents':{}},
            'b':{'uuid':2,'name':'anode','propElectricalConnection':{'connectedTo':['1']},'contents':{}}}}
        def get_num_moves(self):return 3
    s=electrical_snapshot(E())
    assert s['edges']==[['1','2']] and s['moves_after_read']==3

def test_replay_restores_history_and_injects_hint_only_once(tmp_path):
    import copy,json
    from recipe.dependency_pilot.post_p0 import AlfPrefixManager,canonical_state
    from recipe.dependency_pilot.common import digest
    def info(n):
        return {'facts':[prop('holds','agent1','cloth 4')] if n>=2 else [],
            'admissible_commands':['look'],'strict_success':False,'native_score':0,'env_moves':n,
            'task_description':'put a clean cloth in toilet.'}
    class Session:
        def __init__(self,*args):self.n=0
        def reset(self):return 'reset',info(0)
        def step(self,a):self.n+=1;return 'obs'+str(self.n),0,False,info(self.n)
        def close(self):pass
    prefix={'prefix_id':'synthetic','prefix_steps_consumed':2,'prefix':[
        {'parsed_action':'go to toilet 1','observation_post':'obs1','native_info':info(1),'analysis_step_id':'s:0'},
        {'parsed_action':'take cloth 4 from toilet 1','observation_post':'obs2','native_info':info(2),'analysis_step_id':'s:1'}],
        'state_fingerprint':digest(json.dumps(canonical_state('obs2',info(2)),sort_keys=True)),
        'history_fingerprint':digest(json.dumps([('reset','go to toilet 1'),('obs1','take cloth 4 from toilet 1')]))}
    histories=[]
    for arm,hint in HINTS.items():
        cfg={'rollouts_per_task':2,'max_steps':50,'seed':0,'history_length':4,'run_dir':str(tmp_path),'condition':arm}
        m=AlfPrefixManager({'env':'alfworld'},cfg,prefix);m.session_cls=Session
        obs,infos=m.reset();histories.append(copy.deepcopy(m.history))
        assert all('go to toilet 1' in text and 'take cloth 4 from toilet 1' in text for text in obs['text'])
        if hint:assert all(text.endswith(hint) for text in obs['text'])
        obs,_,_,_=m.step(['<think>x</think><action>look</action>']*2)
        if hint:assert all(hint not in text for text in obs['text'])
        assert all(len(h)==3 for h in m.history)
    assert histories[0]==histories[1]==histories[2]
