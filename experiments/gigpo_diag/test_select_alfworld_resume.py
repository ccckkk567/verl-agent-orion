import gzip
import json
import tempfile
import unittest
from pathlib import Path
from select_alfworld_resume import select

class ResumeTests(unittest.TestCase):
    def fixture(self,root,step):
        (root/'diagnostics').mkdir(exist_ok=True)
        (root/'RUN_MANIFEST.json').write_text(json.dumps(dict(env='alfworld',job_id='160764',run_id='run',code_commit='abc')))
        ck=root/'checkpoints'/f'global_step_{step}'
        (ck/'actor').mkdir(parents=True)
        (ck/'data.pt').write_bytes(b'test fixture')
        for kind in ('model','optim','extra_state'):
            for rank in (0,1):
                (ck/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'test fixture')
        return ck

    def test_latest_and_partial_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root,100);latest=self.fixture(root,110)
            (root/'checkpoints/latest_checkpointed_iteration.txt').write_text('110')
            with gzip.open(root/'diagnostics/checkpoints.pid-1.jsonl.gz','wt') as f:
                f.write(json.dumps(dict(run_id='run',code_commit='abc',global_step=100,native_save_returned=True))+'\n')
            self.assertEqual(select(root)[0],110)
            (latest/'actor/optim_world_size_2_rank_1.pt').write_bytes(b'')
            self.assertEqual(select(root)[0],100)

    def test_uncommitted_directory_not_selected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root,110)
            with self.assertRaisesRegex(ValueError,'No committed'):select(root)

    def test_finished_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root,150)
            (root/'checkpoints/latest_checkpointed_iteration.txt').write_text('150')
            self.assertEqual(select(root)[0],150)

if __name__=='__main__':unittest.main()
