"""Inspect KVM access in a CPU or GPU allocation without changing settings."""

import argparse
from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import platform
import stat
import sys


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"error": type(exc).__name__, "errno": exc.errno}


def cpu_details():
    content = read_text("/proc/cpuinfo")
    if not isinstance(content, str):
        return content
    details = {}
    flag_sets = []
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in ("vendor_id", "model name"):
            details.setdefault(key, value)
        elif key == "flags":
            flag_sets.append(set(value.split()))
    common = set.intersection(*flag_sets) if flag_sets else set()
    details["common_virtualization_flags"] = sorted(common & {"vmx", "svm"})
    details["hypervisor_flag_present"] = any("hypervisor" in flags for flags in flag_sets)
    details["note"] = "CPU flags alone do not establish KVM device access."
    return details


def probe_kvm():
    result = {"device": "/dev/kvm", "vm_created": False}
    try:
        device_stat = os.stat("/dev/kvm")
    except FileNotFoundError:
        return dict(result, status="device_missing")
    except OSError as exc:
        return dict(result, status="device_stat_failed", errno=exc.errno, error=str(exc))
    result.update({
        "mode": stat.filemode(device_stat.st_mode),
        "owner_uid": device_stat.st_uid,
        "owner_gid": device_stat.st_gid,
        "read_write_access_hint": os.access("/dev/kvm", os.R_OK | os.W_OK),
    })
    if not stat.S_ISCHR(device_stat.st_mode):
        return dict(result, status="device_not_character")
    result["major_minor"] = [os.major(device_stat.st_rdev), os.minor(device_stat.st_rdev)]
    import fcntl

    # Stable Linux KVM ABI: GET_API_VERSION and CREATE_VM. The VM has no memory
    # or vCPUs; closing both descriptors immediately releases it.
    stage = "open"
    try:
        with ExitStack() as handles:
            device_fd = os.open("/dev/kvm", os.O_RDWR | os.O_CLOEXEC)
            handles.callback(os.close, device_fd)
            stage = "api_version"
            result["api_version"] = fcntl.ioctl(device_fd, 0xAE00, 0)
            if result["api_version"] != 12:
                return dict(result, status="api_version_unsupported")
            stage = "create_vm"
            vm_fd = fcntl.ioctl(device_fd, 0xAE01, 0)
            handles.callback(os.close, vm_fd)
            result.update(status="vm_creation_passed", vm_created=True)
    except OSError as exc:
        result.update(
            status="permission_denied" if exc.errno in (errno.EACCES, errno.EPERM) else "operation_failed",
            failed_stage=stage, errno=exc.errno, error=str(exc),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Submit scripts/orion/cpu-kvm.sbatch or gpu-kvm.sbatch on a compute node.")
    if platform.machine() != "x86_64" or platform.system() != "Linux":
        parser.error("This Android x86 emulator check targets x86_64 Linux nodes.")
    modules = {}
    for name in ("kvm", "kvm_intel", "kvm_amd"):
        directory = Path("/sys/module") / name
        modules[name] = {"visible_in_sysfs": directory.is_dir()}
        if name != "kvm" and directory.is_dir():
            modules[name]["nested_parameter"] = read_text(directory / "parameters/nested")
    report = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uid": os.getuid(), "gid": os.getgid(), "groups": os.getgroups(),
        "cpu": cpu_details(),
        "kernel_modules": modules,
        "kvm": probe_kvm(),
        "scope": "Current allocated node and user only; not an Android emulator boot test",
    }
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"KVM_CHECK_COMPLETE: {report['kvm']['status']}")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
