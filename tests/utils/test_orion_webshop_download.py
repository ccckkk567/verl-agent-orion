"""Download recovery must never publish partial/HTML data as a valid catalog."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/orion/games_prepare.py'
SPEC = importlib.util.spec_from_file_location('orion_games_prepare', SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class VerifiedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'items.json'
        self.payload = b'[{"name":"example product"}]'
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def download(self):
        prepare.download_verified('https://example.invalid/data', self.path,
                                  len(self.payload), self.digest)

    def test_existing_valid_file_needs_no_network(self):
        self.path.write_bytes(self.payload)
        with patch.object(prepare.subprocess, 'run') as run:
            self.download()
        run.assert_not_called()

    def test_existing_wrong_hash_is_rejected_without_overwriting(self):
        damaged = b'x' * len(self.payload)
        self.path.write_bytes(damaged)
        with patch.object(prepare.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                self.download()
        run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), damaged)

    def test_resume_does_not_reuse_drive_partial(self):
        old = self.path.with_name('items.json.part')
        old.write_bytes(b'<html>Drive error</html>')
        partial = self.path.with_name('items.json.hf.part')
        partial.write_bytes(self.payload[:8])

        def complete(args, **kwargs):
            self.assertIn('--continue-at', args)
            self.assertEqual(args[-1], str(partial))
            self.assertEqual(partial.read_bytes(), self.payload[:8])
            partial.write_bytes(self.payload)

        with patch.object(prepare.subprocess, 'run', side_effect=complete):
            self.download()
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertTrue(old.exists())
        self.assertFalse(partial.exists())

    def test_failed_transfer_leaves_partial_only(self):
        partial = self.path.with_name('items.json.hf.part')

        def fail(*args, **kwargs):
            partial.write_bytes(self.payload[:8])
            raise subprocess.CalledProcessError(18, 'curl')

        with patch.object(prepare.subprocess, 'run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                self.download()
        self.assertFalse(self.path.exists())
        self.assertEqual(partial.read_bytes(), self.payload[:8])

    def test_complete_partial_is_verified_without_redownload(self):
        self.path.with_name('items.json.hf.part').write_bytes(self.payload)
        with patch.object(prepare.subprocess, 'run') as run:
            self.download()
        run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_corrupt_download_is_not_promoted(self):
        partial = self.path.with_name('items.json.hf.part')

        def corrupt(*args, **kwargs):
            partial.write_bytes(b'x' * len(self.payload))

        with patch.object(prepare.subprocess, 'run', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                self.download()
        self.assertFalse(self.path.exists())


class IndexCountTests(unittest.TestCase):
    def test_reported_full_catalog_with_sixty_empty_documents(self):
        prepare.validate_index_count(1181430, 60, 1181370)

    def test_missing_nonempty_document_is_not_tolerated(self):
        with self.assertRaisesRegex(RuntimeError, 'Index count mismatch'):
            prepare.validate_index_count(1181430, 60, 1181369)

    def test_extra_document_is_not_tolerated(self):
        with self.assertRaisesRegex(RuntimeError, 'Index count mismatch'):
            prepare.validate_index_count(1181430, 60, 1181371)

    def test_empty_contents_matches_java_trim(self):
        for value in ['', '   ', '\t\r\n', '\x00\x1f ']:
            with self.subTest(value=value):
                self.assertTrue(prepare.empty_index_document(value))
        # Java String.trim does not remove these; using Python strip is wrong.
        for value in ['\u00a0', '\u2003', ' product ', '\x00x\x00']:
            with self.subTest(value=value):
                self.assertFalse(prepare.empty_index_document(value))


if __name__ == '__main__':
    unittest.main()
