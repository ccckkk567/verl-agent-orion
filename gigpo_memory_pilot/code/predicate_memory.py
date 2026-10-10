"""Causal last-observed T/F persistence. No actions, outcomes or future inputs."""
class PredicateMemory:
    def __init__(self, predicate_names, carry_mask):
        self.names=list(predicate_names); self.carry=list(carry_mask)
        assert len(self.names)==len(self.carry)
        self.reset()
    def reset(self):
        self.values=['U']*len(self.names); self.last_observed_step=[None]*len(self.names); self.step=-1
    def update(self,p_obs,step_idx):
        assert step_idx>self.step and len(p_obs)==len(self.names)
        self.step=step_idx; values=[]; carried=[]; ages=[]; types=[]; events=[]
        for i,(v,allow) in enumerate(zip(p_obs,self.carry)):
            assert v in ('T','F','U')
            old=self.values[i]
            if v!='U':
                transition='NEW_'+v
                if allow and old!='U' and old!=v:
                    transition='OVERWRITE_'+old+'_TO_'+v
                    events.append(dict(predicate=self.names[i],previous=old,current=v,last_observed_step=self.last_observed_step[i]))
                self.values[i]=v; self.last_observed_step[i]=step_idx
                out=v; carry=False; age=0
            elif allow and old!='U':
                out=old;carry=True;age=step_idx-self.last_observed_step[i];transition='CARRY_'+old
            else:
                out='U';carry=False;age=None;transition='STILL_U' if allow else 'NONCARRY_U'
                if not allow:self.values[i]='U';self.last_observed_step[i]=None
            values.append(out);carried.append(carry);ages.append(age);types.append(transition)
        return dict(P_mem=values,carried_mask=carried,P_age=ages,transition_types=types,overwrite_events=events)
