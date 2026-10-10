import unittest
from native_protocol import parse_uitars
class MobileSourceTests(unittest.TestCase):
 def test_mobile_duration(self):
  x=parse_uitars("Thought: Hold.\nAction: long_press(start_box='(500,500)', time='1.5')",(1080,1920))
  self.assertEqual(x,{'kind':'long_press','point':[540,960],'seconds':1.5})
 def test_reject_bad_duration(self):
  for v in ["''","'-1'","'nan'","'31'","True"]:
   with self.subTest(value=v),self.assertRaises(ValueError):
    parse_uitars("Thought: Hold.\nAction: long_press(start_box='(500,500)', time="+v+")",(1080,1920))
 def test_original_mobile_end_and_scroll(self):
  self.assertTrue(parse_uitars("Thought: End.\nAction: finished(content='done')",(1080,1920))['declared_complete'])
  self.assertEqual(parse_uitars("Thought: Scroll.\nAction: scroll(start_box='(500,800)', end_box='(500,200)')",(1080,1920))['end'],[540,384])
