import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid
from .common import ROOT


class EnvProcess:
    def __init__(self, python, config):
        directory = Path(config["run_dir"]) / "env_workers"
        directory.mkdir(exist_ok=True)
        self.stderr_path = directory / (uuid.uuid4().hex + ".log")
        self.stderr = self.stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen([python, "-u", "-m", "recipe.dependency_pilot.env_worker"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding="utf-8", bufsize=1)
        self.messages = queue.Queue()
        def pump():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)
        threading.Thread(target=pump, daemon=True).start()

    def call(self, operation, **kwargs):
        self.process.stdin.write(json.dumps({"operation": operation, **kwargs}) + "\n")
        self.process.stdin.flush()
        try:
            line = self.messages.get(timeout=180)
        except queue.Empty:
            raise TimeoutError(f"Environment {operation} exceeded 180s; see {self.stderr_path}")
        if line is None:
            raise RuntimeError(f"Environment worker exited; see {self.stderr_path}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.call("close")
                except (BrokenPipeError, ConnectionResetError):
                    # The child can exit between poll() and the pipe write.
                    # Preserve the original environment error during cleanup.
                    pass
                self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate()  # only this child PID
                self.process.wait(timeout=10)
            self.stderr.close()


class RpcSession:
    def __init__(self, task, config):
        self.remote = EnvProcess(os.environ[task["env"].upper() + "_PYTHON"], config)
        try:
            self.remote.call("init", task=task, config=config)
        except Exception:
            self.remote.close()
            raise

    def reset(self): return self.remote.call("reset")
    def step(self, action): return self.remote.call("step", action=action)
    def close(self): self.remote.close()
