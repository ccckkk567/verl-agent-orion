import gzip
import json
import tempfile
import unittest
from pathlib import Path
from export_web_analysis import export, summarize, read_rows

def adv(t, step, ret, ae=1, astep=1, group='g'):
    return dict(analysis_step_id=f'u0-train_rollout-1:{t}:{step}',episode_group_uid='ep',step_group_id=group,actual_step_return=ret,AE=ae,AS=astep,joint=ae+astep,step_weight=1,training_row_index=step)

class ExportTests(unittest.TestCase):
    def test_peer_and_denominators(self):
        rows=[adv('a',0,1),adv('a',1,2),adv('b',0,1,1,-1),adv('c',0,3,0,0,'single')]
        s,g=summarize(rows)
        self.assertEqual(s['cross_trajectory_peer_coverage'],.75)
        self.assertEqual(s['return_informative_peer_coverage'],.5)
        self.assertEqual(s['singleton_group_rate'],.5)
        self.assertEqual(s['singleton_step_rate'],.25)
        self.assertEqual(s['AE_AS_opposite_all_steps'],.25)
        self.assertAlmostEqual(s['AE_AS_opposite_both_nonzero'],1/3)
        self.assertEqual(summarize([adv('a',0,1),adv('a',1,2)])[0]['return_informative_peer_coverage'],0)

    def fixture(self, root):
        run=root/'run'; (run/'diagnostics').mkdir(parents=True)
        (run/'RUN_MANIFEST.json').write_text(json.dumps(dict(env='alfworld',run_id='test',code_commit='abc')))
        def write(kind, rows):
            with gzip.open(run/'diagnostics'/f'{kind}.pid-1.jsonl.gz','wt') as f:
                for r in rows:
                    f.write(json.dumps(dict(run_id='test',code_commit='abc',policy_version=0,**r))+'\n')
        batch='u0-train_rollout-1'
        write('updates',[dict(policy_update=1,native_actor_update_returned=True)])
        write('rollout_metrics',[dict(rollout_batch_id=batch,sample_kind='train_rollout',episodes=1,active_steps=2,metrics={'strict_success':dict(numerator=1,denominator=1,unknown_count=0,value=1,scope='episode')})])
        write('episodes',[dict(rollout_batch_id=batch,sample_kind='train_rollout',traj_uid='a',task_id='/train/pick_and_place-X/trial/game',active_steps=2,strict_success=True,reference_success=True)])
        a=adv('a',0,1); duplicate=dict(a,training_row_index=99)
        write('advantages',[a,duplicate,adv('a',1,2)])
        steps=[dict(analysis_step_id=adv('a',i,1)['analysis_step_id'],rollout_batch_id=batch,sample_kind='train_rollout',traj_uid='a',step_index=i,anchor_pre_hash='h',action_signature='x',feedback_short='You move the cd 2 to the drawer 1.',format_valid_native=True,events={'object_placed':False,'carrying_target_at_sink':None}) for i in (0,1)]
        write('steps_numeric',steps)
        write('detailed_steps',[dict(s,rendered_prompt='prompt',raw_response='response',prompt_token_ids=[1]) for s in steps])
        return run

    def test_end_to_end_read_only_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run=self.fixture(root)
            before={str(f):f.read_bytes() for f in run.rglob('*') if f.is_file()}
            z=export(run,root/'out',1,1)
            self.assertTrue(z.exists())
            rows=list(read_rows(root/'out/steps_joined.jsonl.gz',[]))
            self.assertEqual(len(rows),2)
            self.assertTrue(rows[0]['step']['offline_placement_feedback_seen'])
            self.assertIsNone(rows[0]['step']['events']['carrying_target_at_sink'])
            self.assertEqual(before,{str(f):f.read_bytes() for f in run.rglob('*') if f.is_file()})
            text=list(read_rows(root/'out/training_text_cases.jsonl.gz',[]))
            self.assertNotIn('prompt_token_ids',text[0])
            with self.assertRaises(FileExistsError): export(run,root/'out',1)
            with self.assertRaises(ValueError): export(run,run/'out',1)

    def test_partial_gzip(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'x.gz'
            f.write_bytes(gzip.compress(b'{"ok":1}\n')+gzip.compress(b'{"partial":2}\n')[:10])
            warnings=[]
            self.assertEqual(list(read_rows(f,warnings)),[{'ok':1}])
            self.assertTrue(warnings)

    def test_identity_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=self.fixture(root)
            with gzip.open(run/'diagnostics/updates.pid-2.jsonl.gz','wt') as f:
                f.write(json.dumps(dict(run_id='other',code_commit='abc'))+'\n')
            with self.assertRaisesRegex(ValueError,'Mixed run'): export(run,root/'out',1)

    def test_unfinished_rollout_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=self.fixture(root)
            r=dict(adv('a',0,3),run_id='test',code_commit='abc',policy_version=1)
            r['analysis_step_id']='u1-train_rollout-2:a:0'
            with gzip.open(run/'diagnostics/advantages.pid-1.jsonl.gz','at') as f:
                f.write(json.dumps(r)+'\n')
            export(run,root/'out',75)
            self.assertEqual(len(list(read_rows(root/'out/raw_advantages.jsonl.gz',[]))),3)
            self.assertEqual(json.loads((root/'out/PACKAGE_MANIFEST.json').read_text())['completed_updates'],[1])

    def test_inconsistent_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=self.fixture(root)
            r=dict(adv('a',0,99),run_id='test',code_commit='abc',policy_version=0)
            with gzip.open(run/'diagnostics/advantages.pid-1.jsonl.gz','at') as f:
                f.write(json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'Inconsistent duplicate'): export(run,root/'out',1)

if __name__=='__main__': unittest.main()
