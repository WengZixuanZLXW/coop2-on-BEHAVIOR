"""TAGParallelInterface: the interface over T -- agents acting on the task
graph directly -- and what a TAG tool hands back."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Union

from ..base import ToolCall
from .action import ATTACH_TOOL, OBSERVE_TOOL, TAG_TOOLS, TaskAction
from .evidence import Evidence
from .graph import TAGGraph
from .spec import TaskSpec
from .version import Version
from .views import build_observation, evidence_for, history, relations

SCHEMA = "tag.v2"

VersionRef = Union[Version, str]

# What a TAG tool hands back to the agent that called it: information from
# T, as a dict, computed after the tool's effect from the graph, the action
# it recorded (None for attach and observe), and the evidence it attached
# (None but for attach). In DIG-TAG that dict is the event D records.
Return = Callable[[TAGGraph, Optional[TaskAction], Optional[Evidence]], Dict[str, Any]]


def observation(graph: TAGGraph, action: Optional[TaskAction] = None,
                evidence: Optional[Evidence] = None) -> Dict[str, Any]:
    """The task configuration as agents see it: the open identities and
    their current versions, the frontier of the open tasks. The default
    return of every tool."""
    seen = build_observation(graph)
    return {"open": list(seen.open_identities), "frontier": [q.to_dict() for q in seen.open_frontier]}


def observation_with_history(graph: TAGGraph, action: Optional[TaskAction] = None,
                             evidence: Optional[Evidence] = None) -> Dict[str, Any]:
    """The observation with, per open identity, its work history (each
    version with the tool, issuer, and call index that returned it) and
    its relations to the other identities (what it came from, what came
    from it), and the evidence on each version of the open frontier."""
    handed = observation(graph, action, evidence)
    handed["histories"] = {
        k: [{"version": q.id, "tool": a.label.value if a is not None else "initial",
             "issuer": a.issuer if a is not None else None, "index": a.call.index if a is not None else None}
            for a, q in history(graph, k)]
        for k in handed["open"]
    }
    handed["relations"] = {k: relations(graph, k) for k in handed["open"]}
    handed["evidence"] = {
        q["id"]: [{"id": phi.id, "source": phi.source, "payload": phi.payload}
                  for phi in evidence_for(graph, graph.tasks[q["id"]])]
        for q in handed["frontier"]
    }
    return handed


def produced(graph: TAGGraph, action: Optional[TaskAction] = None,
             evidence: Optional[Evidence] = None) -> Dict[str, Any]:
    """What the tool itself produced: the versions an action returned, the
    attachment an attach added, nothing for close and observe."""
    if action is not None and action.outputs:
        return {"versions": [q.to_dict() for q in action.outputs]}
    if evidence is not None:
        return {"evidence": evidence.to_dict()}
    return {}


DEFAULT_RETURNS: Dict[str, Return] = {tool: observation for tool in TAG_TOOLS}


def observed(handed: Mapping[str, Any]) -> List[Version]:
    """The versions a tool's return presents, as an agent reads them: the
    frontier of an observation, or the versions a `produced` return lists.
    Copies built from the information, not T's objects; a call names one
    by its id."""
    listed = handed.get("frontier") if "frontier" in handed else handed.get("versions", [])
    return [Version.from_dict(item) for item in listed or []]


def current(handed: Mapping[str, Any], identity: str) -> Version:
    """The version of the identity a return presents, when it presents one."""
    for version in observed(handed):
        if version.identity == identity:
            return version
    raise ValueError(f"the return presents no version of {identity!r}")


def returns_for(overrides: Optional[Mapping[str, Return]] = None) -> Dict[str, Return]:
    """The return policy per tool: the observation for every tool, with
    `overrides` (tool name -> Return) in place of it."""
    unknown = [tool for tool in (overrides or {}) if tool not in TAG_TOOLS]
    if unknown:
        raise ValueError(f"no such TAG tool to set a return for: {unknown}; the tools are {list(TAG_TOOLS)}")
    return {**DEFAULT_RETURNS, **(overrides or {})}


def apply(graph: TAGGraph, issuer: str, call: ToolCall, returns: Mapping[str, Return]) -> Dict[str, Any]:
    """One TAG tool call, applied to the graph and answered: a task tool
    transforms T and records the action under `issuer`; attach adds
    evidence; observe changes nothing. Returns what the tool's return
    policy hands back. Validated in full before anything changes, so a
    rejected call leaves the graph as it was. Not locked: the caller's
    critical section."""
    if call.tool not in TAG_TOOLS:
        raise ValueError(f"unknown tool {call.tool!r}: the TAG offers {list(TAG_TOOLS)}")
    action: Optional[TaskAction] = None
    evidence: Optional[Evidence] = None
    if call.tool == OBSERVE_TOOL:
        pass
    elif call.tool == ATTACH_TOOL:
        target = call.args.get("target")
        if not isinstance(target, str):
            raise ValueError("attach requires `target`: the id of a version")
        graph.expect_task(target)
        evidence = graph.attach(Evidence(
            source=call.args.get("source") or issuer, payload=call.args.get("payload"), target=target,
        ))
    else:
        outputs = graph.transition(call)
        action = graph.record(TaskAction(issuer=issuer, call=call, outputs=outputs))
    return returns[call.tool](graph, action, evidence)


class TAGParallelInterface:
    """The interface over T, as `DIGParallelInterface` is over D and Z: one
    shared task graph that agents act on through the TAG's tools at any
    moment, in parallel, for runtimes that schedule their own agents and
    record nothing else. Every step records who issued it and when; a lock
    serializes the steps, so callers may be threads; a rejected call
    leaves the graph as it was. Concurrent actions on one version leave
    two tasks, not two versions of one: an identity is continued only
    from its frontier version, and an action on a consumed version
    branches.

    A step hands back information from T by the tool's return policy: the
    observation, the frontier of the open tasks, for every tool unless
    `returns` says otherwise for one. It is not an environment in the
    sense of Z: nothing here is a world a call is applied to; the graph is
    the record itself, and what agents were handed is not kept here."""

    def __init__(self, returns: Optional[Mapping[str, Return]] = None) -> None:
        self.tag = TAGGraph()
        self.returns = returns_for(returns)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[], None]] = []
        self._issued = 0

    @property
    def tools(self) -> FrozenSet[str]:
        """The action space: the six task tools, attach, and observe."""
        return frozenset(TAG_TOOLS)

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Call `listener()` after every step: the hook a runtime uses to
        wake or interrupt agents when the tasks move."""
        self._listeners.append(listener)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the graph still while the block runs: no step lands until
        it ends, so a reader on another thread (a live figure) sees one
        consistent graph."""
        with self._lock:
            yield

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()

    def step(self, agent_id: str, tool: str, args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Realize one call by the agent, a tool name and its arguments as
        an LLM tool call arrives, and hand back what the tool returns.
        Raise to reject it: the graph is unchanged."""
        if not isinstance(agent_id, str) or not agent_id:
            raise TypeError(f"agent id must be a non-empty str; got {agent_id!r}")
        call = ToolCall(tool, dict(args or {}))
        with self._lock:
            call.index, call.at = self._issued, time.time()
            handed = apply(self.tag, agent_id, call, self.returns)
            self._issued += 1
        self._notify()
        return handed

    def observe(self) -> Dict[str, Any]:
        """What the observe tool hands back, now and off the record: by
        default the frontier of the open identities. The `observe` tool is
        the same, as a recorded step."""
        with self._lock:
            return self.returns[OBSERVE_TOOL](self.tag, None, None)

    # -- the action space, one method per tool -------------------------

    @staticmethod
    def _id(ref: VersionRef, what: str) -> str:
        if isinstance(ref, Version):
            if ref.id is None:
                raise ValueError(f"{what} must be a recorded version")
            return ref.id
        if isinstance(ref, str) and ref:
            return ref
        raise TypeError(f"{what} must be a recorded version or its id; got {ref!r}")

    @staticmethod
    def _spec_dict(spec: TaskSpec) -> Dict[str, Any]:
        if not isinstance(spec, TaskSpec):
            raise TypeError(f"task spec must be a TaskSpec; got {spec!r}")
        return spec.to_dict()

    def open(self, agent_id: str, spec: TaskSpec, state: Any = None) -> Dict[str, Any]:
        """OPEN(delta, sigma): a new task with a fresh identity."""
        return self.step(agent_id, "open", {"spec": self._spec_dict(spec), "state": state})

    def edit(self, agent_id: str, task: VersionRef, spec: TaskSpec) -> Dict[str, Any]:
        """EDIT(q, delta'): a new version with the spec changed."""
        return self.step(agent_id, "edit", {"task": self._id(task, "task"), "spec": self._spec_dict(spec)})

    def update(self, agent_id: str, task: VersionRef, state: Any) -> Dict[str, Any]:
        """UPDATE(q, sigma'): a new version with the state changed."""
        return self.step(agent_id, "update", {"task": self._id(task, "task"), "state": state})

    def split(self, agent_id: str, task: VersionRef, parts: Sequence[Any]) -> Dict[str, Any]:
        """SPLIT(q, parts): (identity, spec, state) per part, m >= 2."""
        shaped = []
        for part in parts:
            if not isinstance(part, (list, tuple)) or len(part) != 3:
                raise TypeError("each split part must be an (identity, spec, state) tuple")
            identity, spec, state = part
            shaped.append({"identity": identity, "spec": self._spec_dict(spec), "state": state})
        return self.step(agent_id, "split", {"task": self._id(task, "task"), "parts": shaped})

    def join(self, agent_id: str, tasks: Sequence[VersionRef], spec: TaskSpec, *, state: Any = None,
             identity: Optional[str] = None) -> Dict[str, Any]:
        """JOIN(qs, k', delta', sigma'): one version from versions of
        distinct identities."""
        return self.step(agent_id, "join", {
            "tasks": [self._id(task, "task") for task in tasks], "identity": identity,
            "spec": self._spec_dict(spec), "state": state,
        })

    def close(self, agent_id: str, task: VersionRef) -> Dict[str, Any]:
        """CLOSE(k): mark the identity closed; the frontier is unchanged."""
        identity = task.identity if isinstance(task, Version) else task
        return self.step(agent_id, "close", {"identity": identity})

    def attach(self, agent_id: str, target: VersionRef, payload: Any, *, source: Optional[str] = None) -> Dict[str, Any]:
        """attach(phi): evidence on an exact version; the source defaults to
        the attaching agent."""
        return self.step(agent_id, "attach", {"target": self._id(target, "target"), "payload": payload, "source": source})

    # -- the record ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"schema": SCHEMA, "issued": self._issued, "tag": self.tag.to_dict(calls=True)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TAGParallelInterface":
        schema = data.get("schema") if isinstance(data, Mapping) else None
        if schema != SCHEMA:
            raise ValueError(f"unsupported schema {schema!r}; expected {SCHEMA!r}")
        tag = cls()
        tag.tag = TAGGraph.from_dict(data.get("tag") or {})
        tag._issued = int(data.get("issued") or 0)
        return tag

    def save_json(self, path: "str | Path") -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: "str | Path") -> "TAGParallelInterface":
        return cls.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
