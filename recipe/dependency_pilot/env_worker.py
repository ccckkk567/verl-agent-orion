"""JSON-lines bridge to an existing ALFWorld/AppWorld Python environment."""
import contextlib
import json
import sys
import traceback
from .common import json_default


def main():
    session = None
    for line in sys.stdin:
        request = json.loads(line)
        operation = request["operation"]
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from .environments import AlfSession, AppSession, enumerate_tasks
                if operation == "enumerate":
                    config = {**request["config"], "inside_env_worker": True}
                    result = enumerate_tasks(request["env"], config)
                elif operation == "init":
                    cls = {"alfworld": AlfSession, "appworld": AppSession}[request["task"]["env"]]
                    session = cls(request["task"], request["config"])
                    result = True
                elif operation == "reset":
                    result = session.reset()
                elif operation == "step":
                    result = session.step(request["action"])
                elif operation == "close":
                    if session: session.close()
                    result = True
                else:
                    raise ValueError(operation)
            # Encoding is part of the operation: unsupported native objects
            # must return a traceback to the parent rather than kill this worker.
            response = json.dumps({"result": result}, default=json_default)
        except Exception:
            response = json.dumps({"error": traceback.format_exc()})
        print(response, flush=True)
        if operation == "close": break


if __name__ == "__main__":
    main()
