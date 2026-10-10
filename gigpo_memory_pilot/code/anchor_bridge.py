"""Grouping-only bridge: owns one causal memory per live trajectory."""
import json
from predicate_memory import PredicateMemory
CARRY={'ContactsAddContact':['name_matches','phone_matches'],'ContactsNewContactDraft':['first_matches','last_matches','phone_matches','phone_label_matches'],'ClockTimerEntry':['hours_match','minutes_match','seconds_match'],'SimpleCalendarAddOneEvent':['title_matches','description_matches','date_matches','start_time_matches','duration_matches']}
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
class AnchorBridge:
    def __init__(self,arm):
        assert arm in ('obs','mem');self.arm=arm;self.sessions={}
    def reset(self,trajectory_id,task_family,predicate_names):
        self.sessions[trajectory_id]=(task_family,tuple(predicate_names),PredicateMemory(predicate_names,[n in CARRY[task_family] for n in predicate_names]))
    def observe(self,trajectory_id,step_idx,observation_anchor):
        family,names,mem=self.sessions[trajectory_id]
        assert tuple(observation_anchor['P_names'])==names
        state=mem.update(observation_anchor['P'],step_idx)
        obs=canonical([observation_anchor['P'],observation_anchor['M'],sorted(observation_anchor['F_guard'].items())])
        key=canonical([state['P_mem'],observation_anchor['M'],sorted(observation_anchor['F_guard'].items())])
        return dict(anchor_obs=obs if self.arm=='obs' else key,anchor_obs_baseline=obs,anchor_mem=key,**state)
    def finish(self,trajectory_id):del self.sessions[trajectory_id]
