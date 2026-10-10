import json
import unittest
from native_protocol import parse_openmobile, parse_uitars, terminal_reward


def o(args):
    return 'Thought: Next action.\nAction: Operate.\n<tool_call>' + json.dumps({'name':'mobile_use','arguments':args}) + '</tool_call>'


def u(action):
    return 'Thought: Next action.\nAction: ' + action


class NativeProtocolTests(unittest.TestCase):
    def test_coordinate_scales_and_edge(self):
        for value, expected in [(0, [0,0]),(500,[540,1200]),(999,[1078,2397])]:
            self.assertEqual(parse_openmobile(o({'action':'click','coordinate':[value,value]}),[1080,2400])['point'],expected)
        self.assertEqual(parse_uitars(u("click(start_box='<|box_start|>(1000,1000)<|box_end|>')"),[1080,2400])['point'],[1079,2399])

    def test_portrait_landscape_geometry(self):
        self.assertEqual(parse_uitars(u("click(start_box='(250,750)')"),[2400,1080])['point'],[600,810])

    def test_swipe_retains_both_endpoints(self):
        self.assertEqual(parse_openmobile(o({'action':'swipe','coordinate':[100,200],'coordinate2':[300,400]}),[1000,1000]),{'kind':'swipe','start':[100,200],'end':[300,400]})
        self.assertEqual(parse_uitars(u("scroll(start_box='(100,200)', end_box='(300,400)')"),[1000,1000]),{'kind':'swipe','start':[100,200],'end':[300,400]})

    def test_explicit_duration_and_menu_preserved(self):
        self.assertEqual(parse_openmobile(o({'action':'long_press','coordinate':[1,2],'time':3}),[1000,1000])['seconds'],3)
        self.assertEqual(parse_openmobile(o({'action':'wait','time':2.5}),[1000,1000])['seconds'],2.5)
        self.assertEqual(parse_openmobile(o({'action':'system_button','button':'Menu'}),[1000,1000])['button'],'Menu')

    def test_exact_unicode_text_no_enter_injection(self):
        value="他说：'你好'\n第二行"
        for result in [parse_openmobile(o({'action':'type','text':value}),[1,1]),parse_uitars(u('type(content='+repr(value)+')'),[1,1])]:
            self.assertEqual(result,{'kind':'type','text':value})

    def test_termination_does_not_create_success(self):
        success=parse_openmobile(o({'action':'terminate','status':'success'}),[1,1])
        fail=parse_openmobile(o({'action':'terminate','status':'failure'}),[1,1])
        finish=parse_uitars(u("finished(content='done')"),[1,1])
        for a in [success,finish]:
            self.assertEqual(terminal_reward(a,0),0)
            self.assertEqual(terminal_reward(a,1),1)
        self.assertEqual(terminal_reward(fail,1),0)
        self.assertEqual(terminal_reward({'kind':'type'},1),0)

    def test_malformed_json_duplicate_and_multiple_calls(self):
        valid=o({'action':'click','coordinate':[1,2]})
        bad=[valid+valid, valid[:-4], valid+'garbage',valid.replace('"action": "click"','"action":"click","action":"wait"')]
        for text in bad:
            with self.subTest(text=text),self.assertRaises((ValueError,TypeError)):
                parse_openmobile(text,[100,100])

    def test_out_of_bounds_nan_bool_and_unknown_fields(self):
        for xy in [[-1,0],[1000,0],[float('nan'),0],[True,0],[1],[1,2,3]]:
            with self.subTest(xy=xy),self.assertRaises(ValueError):
                parse_openmobile(o({'action':'click','coordinate':xy}),[1000,1000])
        with self.assertRaises(ValueError):parse_openmobile(o({'action':'click','coordinate':[1,2],'index':4}),[100,100])

    def test_no_code_execution_and_no_multi_action(self):
        for command in ["__import__('os').system('true')", "click(start_box=str((1,2)))", "click(start_box='(1,2)');press_back()", "click(**{'start_box':'(1,2)'})", "click(start_box='(-1,2)')", "click(start_box='(1001,2)')", "click(start_box='(True,2)')", "click(start_box='(1,2)', start_box='(3,4)')"]:
            with self.subTest(command=command),self.assertRaises(ValueError):parse_uitars(u(command),[100,100])

    def test_unknown_actions_never_become_wait(self):
        with self.assertRaises(ValueError):parse_openmobile(o({'action':'guess'}),[100,100])
        with self.assertRaises(ValueError):parse_uitars(u('guess()'),[100,100])


if __name__ == '__main__':
    unittest.main()
