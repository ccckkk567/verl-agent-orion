import unittest
from anchor_bridge import AnchorBridge
class BridgeTest(unittest.TestCase):
 def test_reset_and_trajectory_isolation(self):
  b=AnchorBridge('mem');names=['first_matches','last_matches','phone_matches','phone_label_matches','editor_open']
  for t in ['a','b']:b.reset(t,'ContactsNewContactDraft',names)
  obs=lambda v:dict(P_names=names,P=v,M='NORMAL',F_guard={'activity':'UNKNOWN'})
  b.observe('a',0,obs(['T']*5))
  self.assertEqual(b.observe('b',0,obs(['U']*5))['P_mem'],['U']*5)
  self.assertEqual(b.observe('a',1,obs(['U']*5))['P_mem'],['T']*4+['U'])
  b.reset('a','ContactsNewContactDraft',names)
  self.assertEqual(b.observe('a',0,obs(['U']*5))['P_mem'],['U']*5)
 def test_opposite_observation_overwrites_and_age(self):
  b=AnchorBridge('mem');names=['name_matches','phone_matches'];b.reset('a','ContactsAddContact',names)
  obs=lambda v:dict(P_names=names,P=v,M='NORMAL',F_guard={})
  b.observe('a',0,obs(['T','F']))
  r=b.observe('a',1,obs(['U','T']));self.assertEqual(r['P_mem'],['T','T']);self.assertEqual(r['P_age'],[1,0]);self.assertEqual(r['transition_types'][1],'OVERWRITE_F_TO_T')
if __name__=='__main__':unittest.main()
