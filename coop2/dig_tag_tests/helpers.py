"""Shared builders for the golden tests.

Import as `from helpers import ...`: works under pytest (the test directory
is inserted on sys.path) and under direct runs (`python tests/test_x.py`).
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Sequence

# The repo root, so `experiments` imports under direct runs as under pytest.
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dig_tag.base import ToolCall
from dig_tag.dig import DIGActivation, DIGEvent, Environment
from dig_tag.tag import TaskSpec


def spec(goal: str = "goal", rule: Any = None) -> TaskSpec:
    """A task spec with a short goal and an optional rule."""
    return TaskSpec(goal=goal, rule=rule)


class StubEnvironment(Environment):
    """Three tools: `echo` returns one feedback event carrying the call's
    args; `relay` returns one event that DECLARES the calling agent as its
    origin; `fail` raises (the call must then leave no trace). Every call
    is logged on `calls`."""

    tools: FrozenSet[str] = frozenset({"echo", "relay", "fail"})

    def __init__(self) -> None:
        self.calls: List[ToolCall] = []

    def step(self, activation: DIGActivation, call: ToolCall) -> Sequence[DIGEvent]:
        self.calls.append(call)
        if call.tool == "fail":
            raise RuntimeError("the environment rejected the call")
        if call.tool == "relay":
            return [DIGEvent(payload=dict(call.args), origin=activation.agent_id)]
        return [DIGEvent(payload=dict(call.args))]


def run_golden(module_globals: Dict[str, Any], banner: str) -> None:
    """Run a module's test_* functions in definition order (direct-run mode).

    pytest discovers the same functions itself; this keeps every golden file
    runnable as a plain script without a hand-maintained call list. A test
    taking `tmp_path` gets a real temporary directory.
    """
    for name, fn in list(module_globals.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as tmp:
                fn(tmp_path=Path(tmp))
        else:
            fn()
    print(f"{banner}: PASS")
