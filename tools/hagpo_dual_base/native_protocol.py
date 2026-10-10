"""Pure native-action parsing for the dual-base qualification; no device actions.

This is an intermediate representation, not a validated Android executor.
No eval, action repair, coordinate snapping, implicit Enter, or oracle access.
"""
import ast
import json
import math
import re

VERSION = "dual_native_ir_v2"


def _number(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("numeric value required")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError("value outside declared range")
    return float(value)


def point(value, size, upper):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("one two-dimensional point required")
    if len(size) != 2 or any(type(s) is not int or s <= 0 for s in size):
        raise ValueError("positive integer screen dimensions required")
    xy = [_number(v, 0, upper) for v in value]
    # 1000 is the UI-TARS right/bottom edge; map that edge to the final pixel.
    # Out-of-range coordinates were rejected above, never clipped or snapped.
    return [min(int(v * side / 1000), side - 1) for v, side in zip(xy, size)]


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _fields(args, required, optional=()):
    if not isinstance(args, dict) or set(args) - set(required) - set(optional):
        raise ValueError("unknown action field")
    if not set(required) <= set(args):
        raise ValueError("missing action field")


def _text(value):
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    return value


def parse_openmobile(response, size):
    if not isinstance(response, str):
        raise ValueError("response must be text")
    if response.count("<tool_call>") != 1 or response.count("</tool_call>") != 1:
        raise ValueError("exactly one complete tool call required")
    before, payload = response.split("<tool_call>")
    payload, after = payload.split("</tool_call>")
    if after.strip():
        raise ValueError("trailing output after tool call")
    if not re.fullmatch(r"\s*Thought:\s*.+?\s+Action:\s*.+?\s*", before, re.S):
        raise ValueError("native Thought/Action prefix required")
    obj = json.loads(payload, object_pairs_hook=_pairs,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _fields(obj, ("name", "arguments"))
    if obj["name"] != "mobile_use":
        raise ValueError("wrong tool")
    args = obj["arguments"]
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    action = args.get("action")
    if action in {"click", "long_press"}:
        required = ("action", "coordinate") + (("time",) if action == "long_press" else ())
        _fields(args, required)
        result = {"kind": action, "point": point(args["coordinate"], size, 999)}
        if action == "long_press":
            result["seconds"] = _number(args["time"], 0, 30)
        return result
    if action == "swipe":
        _fields(args, ("action", "coordinate", "coordinate2"))
        return {"kind": "swipe", "start": point(args["coordinate"], size, 999),
                "end": point(args["coordinate2"], size, 999)}
    if action in {"type", "answer"}:
        _fields(args, ("action", "text"))
        return {"kind": action, "text": _text(args["text"])}
    if action == "wait":
        _fields(args, ("action", "time"))
        return {"kind": action, "seconds": _number(args["time"], 0, 30)}
    if action == "system_button":
        _fields(args, ("action", "button"))
        if args["button"] not in {"Back", "Home", "Menu", "Enter"}:
            raise ValueError("unsupported system button")
        return {"kind": action, "button": args["button"]}
    if action == "terminate":
        _fields(args, ("action", "status"))
        if args["status"] not in {"success", "failure"}:
            raise ValueError("invalid termination status")
        return {"kind": "terminate", "declared_complete": args["status"] == "success"}
    raise ValueError("unsupported native action")


def _box(value, size):
    text = _text(value)
    m = re.fullmatch(r"(?:<\|box_start\|>)?\s*(\([^()]+\))\s*(?:<\|box_end\|>)?", text)
    if not m:
        raise ValueError("expected native point box")
    return point(ast.literal_eval(m[1]), size, 1000)


def parse_uitars(response, size):
    if not isinstance(response, str) or len(re.findall(r"(?m)^Action:", response)) != 1:
        raise ValueError("exactly one native Action required")
    before, source = re.split(r"(?m)^Action:", response)
    if not before.strip().startswith("Thought:"):
        raise ValueError("native Thought required")
    try:
        call = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError("invalid native action syntax") from exc
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
        raise ValueError("one whitelisted keyword-only action required")
    args = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in args:
            raise ValueError("duplicate/expanded action argument")
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ValueError("only literal action arguments allowed") from exc
    name = call.func.id
    if name in {"click", "long_press"}:
        _fields(args, ("start_box",) + (("time",) if name == "long_press" else ()))
        result = {"kind": name, "point": _box(args["start_box"], size)}
        if name == "long_press":
            value = args["time"]
            # Original mobile template declares time as a quoted value.
            if isinstance(value, str):
                if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                    raise ValueError("invalid native duration")
                value = float(value)
            result["seconds"] = _number(value, 0, 30)
        return result
    if name == "scroll":
        _fields(args, ("start_box", "end_box"))
        return {"kind": "swipe", "start": _box(args["start_box"], size),
                "end": _box(args["end_box"], size)}
    if name == "type":
        _fields(args, ("content",))
        return {"kind": "type", "text": _text(args["content"])}
    if name in {"press_home", "press_back"}:
        _fields(args, ())
        return {"kind": "system_button", "button": "Home" if name == "press_home" else "Back"}
    if name == "finished":
        _fields(args, ("content",))
        _text(args["content"])
        return {"kind": "terminate", "declared_complete": True}
    raise ValueError("unsupported native action")


def terminal_reward(action, raw_score):
    """A completion declaration is necessary but never sufficient for reward."""
    score = _number(raw_score, 0, 1)
    return score if action.get("kind") == "terminate" and action.get("declared_complete") else 0.0
