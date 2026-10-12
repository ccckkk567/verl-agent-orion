import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    if not isinstance(value, bytes):
        value = str(value).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def versions():
    result = {}
    for name in ["scienceworld", "alfworld", "appworld", "torch", "transformers", "swanlab", "ray", "tensordict"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_info():
    result = {}
    for key, args in {"commit": ["rev-parse", "HEAD"], "branch": ["branch", "--show-current"], "status": ["status", "--short"]}.items():
        proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        result[key] = proc.stdout.strip() if proc.returncode == 0 else None
    return result
