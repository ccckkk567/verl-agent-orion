"""Bounded single-AVD software-emulation diagnostic, not an AndroidWorld run."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import struct
import subprocess
import time
import xml.etree.ElementTree as ET


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def emulator_port():
    ports = list(range(5554, 5683, 2))
    random.shuffle(ports)
    for port in ports:
        with socket.socket() as console, socket.socket() as adb:
            try:
                console.bind(('127.0.0.1', port))
                adb.bind(('127.0.0.1', port + 1))
            except OSError:
                continue
            return port
    raise RuntimeError('No free emulator console/ADB port pair')


def stop_owned_process(process):
    if process is None:
        return
    # Every managed service starts in its own session; never use pkill or stop
    # the default/shared ADB server. Also reap children if its leader exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def settings_target(xml):
    root = ET.fromstring(xml)
    for title in ['Network & internet', 'Connected devices', 'Display', 'Sound & vibration']:
        for node in root.iter('node'):
            if node.get('package') != 'com.android.settings' or node.get('text') != title:
                continue
            match = re.fullmatch(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', node.get('bounds', ''))
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                if x2 > x1 and y2 > y1:
                    return title, (x1 + x2) // 2, (y1 + y2) // 2
    raise RuntimeError('Settings page has no recognized visible click target')


def ui_texts(xml):
    return {node.get('text') for node in ET.fromstring(xml).iter('node')
            if node.get('package') == 'com.android.settings' and node.get('text')}


def navigated(before, after, title):
    old, new = ui_texts(before), ui_texts(after)
    # Ignore clock/focus changes; require a settings subpage with new content.
    return title in new and len(old - new) >= 2 and bool(new - old)


def save_png(path, data):
    if len(data) < 128 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise RuntimeError('screencap did not produce a PNG image')
    width, height = struct.unpack('>II', data[16:24])
    if width == 0 or height == 0:
        raise RuntimeError('Screenshot has invalid dimensions')
    path.write_bytes(data)
    return {'path': str(path), 'width': width, 'height': height, 'bytes': len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--avd', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boot-seconds', type=int, default=1200)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or platform.system() != 'Linux':
        parser.error('Run inside an Orion Linux Slurm allocation')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'hostname': platform.node(),
        'partition': os.environ.get('SLURM_JOB_PARTITION'),
        'allocated_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
        'gpu_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'acceleration': 'off', 'graphics': 'swiftshader', 'avd': args.avd,
        'boot_limit_seconds': args.boot_seconds, 'status': 'running', 'stage': 'preflight',
        'scope': 'Emulator boot/ADB/screenshot/tap only; no AndroidWorld apps, agent or model',
        'timings_seconds': {}, 'sdk_properties': {},
    }
    start = time.monotonic()
    adb_process = emulator = None
    handles = []
    env = dict(os.environ)

    def interrupted(signum, frame):
        raise RuntimeError(f'Probe interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def command(argv, timeout=30, check=True):
        result = subprocess.run([str(a) for a in argv], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}): {argv}\n'
                               + result.stderr.decode(errors='replace')[-2000:]
                               + result.stdout.decode(errors='replace')[-2000:])
        return result

    def adb(*words, timeout=30, check=True):
        if adb_process is None or adb_process.poll() is not None:
            raise RuntimeError('Private ADB server exited; see adb.log')
        return command([args.sdk / 'platform-tools/adb', '-H', 'localhost', '-P', str(adb_port),
                        '-s', serial, *words], timeout=timeout, check=check)

    def dump_ui(name):
        adb('shell', 'uiautomator', 'dump', '/sdcard/soft-probe.xml', timeout=90)
        xml = adb('shell', 'cat', '/sdcard/soft-probe.xml').stdout.decode()
        ET.fromstring(xml)
        (args.output / name).write_text(xml, encoding='utf-8')
        return xml

    try:
        for relative in ['emulator/source.properties', 'platform-tools/source.properties',
                         'system-images/android-33/google_apis/x86_64/source.properties']:
            report['sdk_properties'][relative] = (args.sdk / relative).read_text()
        report['emulator_version'] = command([args.sdk / 'emulator/emulator', '-version']).stdout.decode(errors='replace')
        console_port = emulator_port()
        adb_port = free_port()
        while adb_port in (console_port, console_port + 1):
            adb_port = free_port()
        grpc_port = free_port()
        while grpc_port in (console_port, console_port + 1, adb_port):
            grpc_port = free_port()
        serial = f'emulator-{console_port}'
        env['ANDROID_ADB_SERVER_PORT'] = str(adb_port)
        # ADB's listener parser recognizes literal "localhost", but rejects
        # "127.0.0.1" here even though both normally denote the loopback host.
        adb_socket = f'tcp:localhost:{adb_port}'
        env['ADB_SERVER_SOCKET'] = adb_socket
        report['ports'] = {'adb_server': adb_port, 'emulator_console': console_port, 'grpc': grpc_port}
        report['stage'] = 'adb_start'
        adb_launch = [str(args.sdk / 'platform-tools/adb'),
                      '-L', adb_socket, 'server', 'nodaemon']
        report['adb_launch_command'] = adb_launch
        adb_log = (args.output / 'adb.log').open('wb')
        handles.append(adb_log)
        adb_process = subprocess.Popen(adb_launch,
            env=env, stdout=adb_log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)
        if adb_process.poll() is not None:
            raise RuntimeError('Private ADB server failed to start; see adb.log')
        emulator_log = (args.output / 'emulator.log').open('wb')
        handles.append(emulator_log)
        launch = [str(args.sdk / 'emulator/emulator'), '-avd', args.avd,
                  '-port', str(console_port), '-grpc', str(grpc_port),
                  '-accel', 'off', '-gpu', 'swiftshader', '-cores', '2', '-memory', '4096',
                  '-no-window', '-no-audio', '-no-boot-anim', '-no-snapshot']
        report['launch_command'] = launch
        report['stage'] = 'boot'
        print('Starting software emulator:', ' '.join(launch), flush=True)
        boot_start = time.monotonic()
        emulator = subprocess.Popen(launch, env=env, stdout=emulator_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        next_progress = boot_start
        while time.monotonic() - boot_start < args.boot_seconds:
            if emulator.poll() is not None:
                raise RuntimeError(f'Emulator exited with code {emulator.returncode}; see emulator.log')
            try:
                state = adb('get-state', timeout=5, check=False)
                if state.returncode == 0 and state.stdout.strip() == b'device':
                    report['timings_seconds'].setdefault('adb_ready', round(time.monotonic() - boot_start, 2))
                    ready = adb('shell', 'getprop', 'sys.boot_completed', timeout=5, check=False)
                    if ready.returncode == 0 and ready.stdout.strip() == b'1':
                        break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= next_progress:
                print(f'Waiting for boot: {time.monotonic() - boot_start:.0f}s', flush=True)
                next_progress = time.monotonic() + 60
            time.sleep(5)
        else:
            report['status'] = 'boot_timeout'
            raise RuntimeError(f'Android did not finish boot within {args.boot_seconds}s')
        report['timings_seconds']['boot_completed'] = round(time.monotonic() - boot_start, 2)
        print('ANDROID_BOOT_OK', report['timings_seconds'], flush=True)

        report['stage'] = 'settings_and_screenshot'
        step_start = time.monotonic()
        # Changes apply only to this fresh disposable diagnostic AVD.
        adb('shell', 'settings', 'put', 'global', 'device_provisioned', '1')
        adb('shell', 'settings', 'put', 'secure', 'user_setup_complete', '1')
        adb('shell', 'input', 'keyevent', '224')
        adb('shell', 'wm', 'dismiss-keyguard')
        adb('shell', 'input', 'keyevent', '3')
        adb('shell', 'am', 'start', '-W', '-a', 'android.settings.SETTINGS', timeout=90)
        before = dump_ui('before.xml')
        report['screenshots'] = {'before': save_png(args.output / 'before.png',
            adb('exec-out', 'screencap', '-p', timeout=90).stdout)}
        report['timings_seconds']['open_settings_dump_screenshot'] = round(time.monotonic() - step_start, 2)
        report['stage'] = 'tap_and_verify'
        title, x, y = settings_target(before)
        step_start = time.monotonic()
        adb('shell', 'input', 'tap', str(x), str(y), timeout=60)
        after = before
        for attempt in range(6):
            time.sleep(2)
            after = dump_ui('after.xml')
            if navigated(before, after, title):
                break
        else:
            raise RuntimeError('Tap returned but settings subpage navigation was not verified')
        report['timings_seconds']['tap_to_verified_ui_change'] = round(time.monotonic() - step_start, 2)
        report['screenshots']['after'] = save_png(args.output / 'after.png',
            adb('exec-out', 'screencap', '-p', timeout=90).stdout)
        report['tap'] = {'target': title, 'x': x, 'y': y, 'visible_ui_changed': True}
        report.update(status='passed', stage='complete')
    except Exception as exc:
        if report['status'] == 'running':
            report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        # Avoid a second TERM preventing cleanup/report during the outer timeout.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process in (emulator, adb_process):
            try:
                stop_owned_process(process)
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append(str(exc))
        for handle in handles:
            handle.close()
        if report['status'] != 'passed':
            # Include bounded service diagnostics directly in the Slurm output
            # through report.json, so a startup failure needs no extra log round trip.
            for filename in ('adb.log', 'emulator.log'):
                try:
                    with (args.output / filename).open('rb') as handle:
                        handle.seek(0, os.SEEK_END)
                        handle.seek(max(0, handle.tell() - 8192))
                        report.setdefault('log_tails', {})[filename] = handle.read().decode(errors='replace')
                except OSError:
                    pass
        report['timings_seconds']['total_probe'] = round(time.monotonic() - start, 2)
        path = args.output / 'report.json'
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        print('Report:', path, flush=True)
    if report['status'] != 'passed' or report.get('cleanup_errors'):
        raise SystemExit(1)
    print('ANDROID_SOFTWARE_PROBE_OK', flush=True)


if __name__ == '__main__':
    main()
