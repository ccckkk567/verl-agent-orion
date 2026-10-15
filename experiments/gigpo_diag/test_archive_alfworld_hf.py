import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
import archive_alfworld_hf as m

class ArchiveTests(unittest.TestCase):
    def test_hash_verification(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'x'; f.write_bytes(b'checkpoint')
            h=m.hashes(f)
            m.verify(h,N(size=10,lfs={'sha256':h['sha256']},blob_id='unused'))
            m.verify(h,N(size=10,lfs=None,blob_id=h['git_blob']))
            with self.assertRaises(ValueError): m.verify(h,N(size=10,lfs={'sha256':'wrong'},blob_id='unused'))
            with self.assertRaises(ValueError): m.verify(h,N(size=11,lfs=None,blob_id=h['git_blob']))
    def test_path_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            m.ROOT=Path(td)/'runs'; m.ROOT.mkdir()
            with self.assertRaises(ValueError): m.files(Path(td))
            p=m.ROOT/'a';p.mkdir(); (p/'x').write_bytes(b'1')
            self.assertEqual(len(m.files(p)),1)
    def test_complete_requires_optimizer_and_loader(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'actor').mkdir()
            self.assertFalse(m.complete(p))
            for kind in ('model','optim','extra_state'):
                for rank in (0,1): (p/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
            self.assertFalse(m.complete(p))
            (p/'data.pt').write_bytes(b'fixture')
            self.assertTrue(m.complete(p))
            (p/'actor/optim_world_size_2_rank_1.pt').write_bytes(b'')
            self.assertFalse(m.complete(p))
if __name__=='__main__':unittest.main()
