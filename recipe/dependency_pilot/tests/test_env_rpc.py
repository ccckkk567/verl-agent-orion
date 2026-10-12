"""Synthetic bridge regression fixtures; never model/environment evidence."""
import io
import json
from types import SimpleNamespace

import pytest

from recipe.dependency_pilot.common import json_default
from recipe.dependency_pilot import env_worker, environments
from recipe.dependency_pilot.env_rpc import EnvProcess


class Proposition:
    # Implements the public TextWorld 1.6.2 serialization contract; the full
    # ALFWorld engine runs on Linux and is not required by these local checks.
    __module__ = "textworld.logic"

    def serialize(self):
        return {"name": "in", "arguments": [
            {"name": "apple 1", "type": "o"},
            {"name": "fridge 1", "type": "r"},
        ]}


def run_worker(monkeypatch, session, operations):
    monkeypatch.setattr(environments, "AlfSession", lambda task, config: session)
    requests = [{"operation": "init", "task": {"env": "alfworld"}, "config": {}}]
    requests += [{"operation": operation, "action": "look"} for operation in operations]
    output = io.StringIO()
    monkeypatch.setattr(env_worker.sys, "stdin", io.StringIO(
        "\n".join(json.dumps(request) for request in requests) + "\n"))
    monkeypatch.setattr(env_worker.sys, "stdout", output)
    env_worker.main()
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_reset_and_step_transfer_structured_facts(monkeypatch):
    info = {"facts": [Proposition()], "admissible_commands": ["look"],
            "strict_success": None, "extra.gamefile": "/data/game.tw-pddl"}
    session = SimpleNamespace(reset=lambda: ("observation", info),
        step=lambda action: ("next", 0.0, False, info), close=lambda: None)
    replies = run_worker(monkeypatch, session, ["reset", "step", "close"])
    assert len(replies) == 4 and all("result" in reply for reply in replies)
    reset_info = replies[1]["result"][1]
    assert reset_info["facts"] == [{"__type__": "textworld.logic.Proposition",
                                    **Proposition().serialize()}]
    assert reset_info["admissible_commands"] == ["look"]
    assert reset_info["extra.gamefile"] == "/data/game.tw-pddl"
    assert reset_info["strict_success"] is None
    assert replies[2]["result"][1:3] == [0.0, False]
    assert replies[2]["result"][3] == reset_info


def test_encoding_failure_returns_traceback_and_worker_can_close(monkeypatch):
    closed = []
    session = SimpleNamespace(reset=lambda: ("observation", {"unsupported": object()}),
                              close=lambda: closed.append(True))
    replies = run_worker(monkeypatch, session, ["reset", "close"])
    assert replies[0] == {"result": True}
    assert "TypeError: object" in replies[1]["error"]
    assert "Traceback" in replies[1]["error"]
    assert replies[2] == {"result": True} and closed == [True]


def test_close_pipe_race_preserves_original_failure():
    remote = EnvProcess.__new__(EnvProcess)
    running = [True]
    remote.process = SimpleNamespace(poll=lambda: None if running[0] else 1,
        wait=lambda timeout: running.__setitem__(0, False))
    remote.stderr = io.StringIO()

    def disconnected(operation):
        raise BrokenPipeError("child exited after poll")

    remote.call = disconnected
    with pytest.raises(RuntimeError, match="original reset failure"):
        try:
            raise RuntimeError("original reset failure")
        finally:
            remote.close()
    assert remote.stderr.closed and running == [False]


def test_unknown_objects_are_not_silently_stringified():
    with pytest.raises(TypeError, match="object"):
        json.dumps({"unknown": object()}, default=json_default)
