"""Inspect a Slurm compute allocation without installing or downloading packages."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


def command(argv):
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error_type": type(exc).__name__}


def network_check(url):
    # Fixed public endpoints only. Never print proxy settings or credentials.
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Orion-preflight"})
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read(32)
            return {"reachable": True, "status": response.status}
    except urllib.error.HTTPError as exc:
        return {"reachable": False, "status": exc.code}
    except (OSError, urllib.error.URLError) as exc:
        return {"reachable": False, "error_type": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this probe through scripts/orion/preflight.sbatch.")

    disk = shutil.disk_usage(args.output.parent)
    report = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "libc": platform.libc_ver(),
        "python": sys.version,
        "python_executable": sys.executable,
        "repository": str(args.repo.resolve()),
        "repository_commit": command(["git", "-C", str(args.repo), "rev-parse", "HEAD"]),
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": command([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
        "cuda_compiler": command(["nvcc", "--version"]),
        "c_compiler": command(["gcc", "--version"]),
        "kvm_exists": Path("/dev/kvm").exists(),
        "kvm_read_write_access": os.access("/dev/kvm", os.R_OK | os.W_OK),
        "filesystem_free_gib_not_project_quota": round(disk.free / 2**30, 2),
        "network": {
            "pypi": network_check("https://pypi.org/pypi/vllm/0.11.0/json"),
            "pytorch": network_check("https://download.pytorch.org/whl/cu128/"),
            "huggingface": network_check("https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct"),
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to {args.output}")
    if report["gpu"].get("returncode") != 0:
        raise SystemExit("GPU inspection failed; review this allocation before installation.")


if __name__ == "__main__":
    main()
