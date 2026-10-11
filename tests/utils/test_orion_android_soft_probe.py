"""Validate diagnostic evidence and failure reporting without launching an AVD."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('android_soft_probe',
    Path(__file__).resolve().parents[2] / 'scripts/orion/android_soft_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def xml(texts, clock='12:00'):
    from xml.sax.saxutils import escape
    return '<hierarchy>' + ''.join(
        f'<node package="com.android.settings" text="{escape(t)}" bounds="[10,20][110,60]" />'
        for t in texts) + f'<node package="com.android.systemui" text="{clock}" /></hierarchy>'


class ProbeEvidenceTests(unittest.TestCase):
    def test_selects_center_of_visible_settings_entry(self):
        self.assertEqual(probe.settings_target(xml(['Network & internet'])),
                         ('Network & internet', 60, 40))

    def test_clock_change_does_not_count_as_navigation(self):
        before = xml(['Network & internet', 'Display', 'Connected devices'])
        after = xml(['Network & internet', 'Display', 'Connected devices'], '12:01')
        self.assertFalse(probe.navigated(before, after, 'Network & internet'))

    def test_settings_subpage_counts_as_navigation(self):
        before = xml(['Network & internet', 'Display', 'Connected devices'])
        after = xml(['Network & internet', 'Internet', 'Airplane mode'])
        self.assertTrue(probe.navigated(before, after, 'Network & internet'))

    def test_unrecognized_page_is_not_blindly_clicked(self):
        with self.assertRaises(RuntimeError):
            probe.settings_target(xml(['Permission required']))

    def test_screenshot_rejects_adb_error_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'screen.png'
            with self.assertRaises(RuntimeError):
                probe.save_png(path, b'error: device offline')
            self.assertFalse(path.exists())

    def test_screenshot_records_dimensions(self):
        data = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 1080, 2400) + bytes(128)
        with tempfile.TemporaryDirectory() as temp:
            result = probe.save_png(Path(temp) / 'screen.png', data)
        self.assertEqual((result['width'], result['height']), (1080, 2400))

    def test_preflight_failure_writes_report(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = ['probe', '--sdk', str(Path(temp) / 'absent-sdk'), '--avd', 'test', '--output', temp]
            with patch.object(sys, 'argv', argv), patch.dict(probe.os.environ, {'SLURM_JOB_ID': '123'}), \
                 patch.object(probe.platform, 'system', return_value='Linux'), \
                 patch.object(probe.signal, 'signal'), patch('builtins.print'):
                with self.assertRaises(SystemExit) as exit_status:
                    probe.main()
            self.assertEqual(exit_status.exception.code, 1)
            report = json.loads((Path(temp) / 'report.json').read_text())
            self.assertEqual(report['stage'], 'preflight')
            self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
