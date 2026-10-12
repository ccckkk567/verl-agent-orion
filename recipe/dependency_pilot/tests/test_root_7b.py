"""Small interpretation/configuration fixtures, never experiment evidence."""
import json
from pathlib import Path
import pytest
from recipe.dependency_pilot.post_analysis import annotate, cloth_state
from recipe.dependency_pilot.post_p0 import root_7b_config, BASELINE_IDS, ROOT_TASKS, verify_local_model
from recipe.dependency_pilot.post_audit import MATCHED_PROTOCOL
from recipe.dependency_pilot.common import write_json, append_jsonl

def prop(name,*args):
    return {'name':name,'arguments':[{'name':a} for a in args]}

def row(env='alfworld',action='look',info=None,task='train/pick_clean_then_place_in_recep-Cloth-None-Toilet-402/t'):
    return {'run_id':'fixture','analysis_step_id':'a:0','traj_uid':'a','step_index':0,'env':env,
        'native_info':info or {},'parsed_action':action,'raw_response':'<action>'+action+'</action>',
        'observation_post':'Nothing happens.','anchor_pre_raw':'room','task_instance_id':task}

def test_sink_presence_is_not_holding_and_any_cloth_id_can_be_cleaned():
    pre={'facts':[prop('atlocation','agent1','loc1'),prop('receptacleatlocation','sinkbasin 2','loc1')],
         'admissible_commands':['look']}
    e=annotate(row(info=pre),pre)
    assert e['any_sink_reached'] is True and e['sink_reached'] is False
    after={'facts':pre['facts']+[prop('holds','agent1','cloth 7'),prop('isClean','cloth 7')],
        'admissible_commands':['clean cloth 7 with sinkbasin 2']}
    e=annotate(row(action='clean cloth 7 with sinkbasin 2',info=after),pre)
    assert e['sink_reached'] and e['clean_success'] and e['cleaned_ids']==['cloth 7']
    assert e['clean_command_available_while_holding']

def test_pick_cleaning_is_inapplicable_and_pickup_detected():
    e=annotate(row(action='take creditcard 2 from desk 1',info={'facts':[prop('holds','agent1','creditcard 2')]},
        task='train/pick_and_place_simple-CreditCard-None-Sofa-220/t'),{'facts':[]})
    assert e['object_pickup'] and e['take_success']
    assert e['clean_success'] is None and e['any_sink_reached'] is None

def test_connect_is_a_word_not_prose_prefix_and_focus_never_connection():
    for action,expected in [('connected to bulb',False),('connections',False),('connect a to b',True),(' CONNECT a to b ',True)]:
        e=annotate(row('scienceworld',action,{'electrical_observer':{'edges':[]}}),{'electrical_observer':{'edges':[]}})
        assert e['connect_attempt']==expected
        assert not e['connect_success']

@pytest.mark.parametrize('env',['alfworld','scienceworld'])
def test_7b_override_preserves_root_protocol_and_separates_historical_lock(tmp_path,env):
    old={**MATCHED_PROTOCOL,'env':env,'run_id':BASELINE_IDS[env],'seed':0,'dynamic_filtering':False,
        'model_path':'old-1.5B','checkpoint_id':'old@'+MATCHED_PROTOCOL['model_revision'],
        'chat_template':'old','history_offset':11,'prefix_id':'forbidden','prompt_profile':'reference_interface_v1' if env=='scienceworld' else 'p0_original',
        'parser_profile':'p0_original_unchanged','interface_lock':{'status':'locked_after_real_smoke',
        'selected_profile':'reference_interface_v1','parser_profile':'p0_original_unchanged','jar_sha256':'fixture',
        'model_revision':MATCHED_PROTOCOL['model_revision'],'matched_protocol':MATCHED_PROTOCOL}}
    write_json(tmp_path/'resolved_config.json',old)
    (tmp_path/'steps.jsonl').write_text('{}\n')
    for task in ROOT_TASKS[env]:append_jsonl(tmp_path/'tasks_manifest.jsonl',{'task_instance_id':task})
    cfg=root_7b_config(tmp_path,'new-7B','fixture-run',tmp_path/'out','disabled')
    assert cfg['model_id']=='Qwen/Qwen2.5-7B-Instruct'
    assert MATCHED_PROTOCOL['model_revision'] not in json.dumps(cfg)
    assert cfg['policy_updated'] is False and cfg['training_updates']==0
    assert cfg['max_steps']==50 and cfg['rollouts_per_task']==4 and cfg['history_length']==4
    assert cfg['additional_hint'] is None and 'history_offset' not in cfg and 'prefix_id' not in cfg
    assert old['checkpoint_id'].startswith('old@')
    old['run_id']='alfworld-p1-not-a-baseline';write_json(tmp_path/'resolved_config.json',old)
    with pytest.raises(ValueError,match='historical root P0'):root_7b_config(tmp_path,'new-7B','fixture',tmp_path/'out')

def test_local_model_refuses_old_revision_before_loading(tmp_path):
    write_json(tmp_path/'download_manifest.json',{'repo_id':'Qwen/Qwen2.5-1.5B-Instruct','revision':'old'})
    with pytest.raises(ValueError,match='pinned 7B'):
        verify_local_model({'model_path':str(tmp_path),'model_id':'Qwen/Qwen2.5-7B-Instruct','model_revision':'new'})
