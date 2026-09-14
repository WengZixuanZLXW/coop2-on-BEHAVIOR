"""DIGTAG: the collaboration interface DT = (D, T, Z, U) -- the DIG
interface with the task graph beside the environment."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Union

from ..base import ToolCall
from ..dig import DIGActivation, DIGEvent, DIGParallelInterface, Environment
from ..tag import ATTACH_TOOL, OBSERVE_TOOL, TAG_TOOLS, Return, TAGGraph, TaskAction, TaskSpec, Version, apply, returns_for

SCHEMA = "dig-tag.v3"
TAG_ORIGIN = "tag"        # the declared origin of what a TAG tool hands back


class DIGTAG(DIGParallelInterface):
    """DT = (D, T, Z, U) over the shared agent identities I.

    `dig` is D, `tag` is T, `environment` is Z, and the tool families U are
    the TAG's tools, send, and whatever the environment declares. D and T
    are separate records that may describe the same information in their
    own ways: a task call transforms T (its action and versions are T's),
    and what the tool hands back -- information from T by its return
    policy, the observation unless `returns` says otherwise for a tool --
    is what D records, as one event returned by the call, the way it
    records what any tool returns. Nothing in T is also in D.

    There is ONE write path: `issue(activation, call)` realizes any call,
    and the tool methods (`open_task`, `edit`, `update`, `split`, `join`,
    `close`, `attach`, `observe`, `send`, `act`) only build a `ToolCall`
    and hand it to `issue`, returning the event D recorded. A TAG call is
    applied and recorded inside the interface's critical section, so D
    and T move together. A rejected call leaves D, T, and Z untouched. A
    DIG event a call names (send's event) must be in Avail(h, r); a task
    reference (an input version, an attach target, a close's identity)
    must exist in T."""

    def __init__(self, agents: Iterable[str] = (), environment: Optional[Environment] = None,
                 returns: Optional[Mapping[str, Return]] = None) -> None:
        super().__init__(agents, environment)
        self.tag = TAGGraph()
        self.returns = returns_for(returns)

    @property
    def reserved_tools(self) -> FrozenSet[str]:
        return super().reserved_tools | frozenset(TAG_TOOLS)

    # -- the root funnel for T ------------------------------------------

    def inject_task(self, version: Version, *, to: Sequence[str]) -> DIGEvent:
        """Admit an initial task: the version enters T with no action, and
        the agents in `to` are handed the observation as a root event."""
        with self._lock:
            self.tag.record_root(version)
            handed = self.returns[OBSERVE_TOOL](self.tag, None, None)
        return self.inject(DIGEvent(payload=handed, origin=TAG_ORIGIN), to=to, origin=TAG_ORIGIN)

    # -- the TAG's tools in the write path --------------------------------

    def _realize(self, activation: DIGActivation, call: ToolCall) -> None:
        """A TAG tool call: T (+) its effect, then D (+)_h (u, {e}) with e
        carrying what the tool handed back."""
        if call.tool not in TAG_TOOLS:
            super()._realize(activation, call)
        handed = apply(self.tag, activation.id, call, self.returns)
        self.dig.append(activation, call, returned=[DIGEvent(payload=handed, origin=TAG_ORIGIN)], origin=TAG_ORIGIN)

    # -- the TAG's tools: sugar over `issue`, each returning D's event -----

    @staticmethod
    def _spec_dict(spec: TaskSpec) -> Dict[str, Any]:
        if not isinstance(spec, TaskSpec):
            raise TypeError(f"task spec must be a TaskSpec; got {spec!r}")
        return spec.to_dict()

    @staticmethod
    def _task_id(ref: Union[Version, str], what: str) -> str:
        if isinstance(ref, Version):
            if ref.id is None:
                raise ValueError(f"{what} must be a recorded version")
            return ref.id
        if isinstance(ref, str) and ref:
            return ref
        raise TypeError(f"{what} must be a recorded version or its id; got {ref!r}")

    def _tag_call(self, activation: DIGActivation, tool: str, args: Dict[str, Any]) -> DIGEvent:
        (event,) = self.issue(activation, ToolCall(tool, args))
        return event

    def open_task(self, activation: DIGActivation, spec: TaskSpec, state: Any = None) -> DIGEvent:
        """OPEN(delta, sigma): {} -> q."""
        return self._tag_call(activation, TaskAction.Label.OPEN.value, {"spec": self._spec_dict(spec), "state": state})

    def edit(self, activation: DIGActivation, task: Union[Version, str], spec: TaskSpec) -> DIGEvent:
        """EDIT(q, delta'): q -> (k, delta', sigma)."""
        return self._tag_call(activation, TaskAction.Label.EDIT.value,
                              {"task": self._task_id(task, "task"), "spec": self._spec_dict(spec)})

    def update(self, activation: DIGActivation, task: Union[Version, str], state: Any) -> DIGEvent:
        """UPDATE(q, sigma'): q -> (k, delta, sigma')."""
        return self._tag_call(activation, TaskAction.Label.UPDATE.value,
                              {"task": self._task_id(task, "task"), "state": state})

    def split(self, activation: DIGActivation, task: Union[Version, str], parts: Sequence[Any]) -> DIGEvent:
        """SPLIT(q, {(k_j, delta_j, sigma_j)}): q -> {q_j}, m >= 2. Each part
        is an (identity, spec, state) tuple; identity None mints a fresh
        one, the input identity may be kept."""
        shaped = []
        for part in parts:
            if not isinstance(part, (list, tuple)) or len(part) != 3:
                raise TypeError("each split part must be an (identity, spec, state) tuple")
            identity, spec, state = part
            shaped.append({"identity": identity, "spec": self._spec_dict(spec), "state": state})
        return self._tag_call(activation, TaskAction.Label.SPLIT.value,
                              {"task": self._task_id(task, "task"), "parts": shaped})

    def join(self, activation: DIGActivation, tasks: Sequence[Union[Version, str]], spec: TaskSpec, *,
             state: Any = None, identity: Optional[str] = None) -> DIGEvent:
        """JOIN({q_j}, k', delta', sigma'): {q_j} -> q', m >= 2. `identity`
        may continue an input identity or introduce a new one (None mints)."""
        return self._tag_call(activation, TaskAction.Label.JOIN.value, {
            "tasks": [self._task_id(task, "task") for task in tasks], "identity": identity,
            "spec": self._spec_dict(spec), "state": state,
        })

    def close(self, activation: DIGActivation, task: Union[Version, str]) -> DIGEvent:
        """CLOSE(k): mark a task identity closed. `task` is the identity, or
        a recorded version whose identity is meant; no version is consumed
        and the frontier is unchanged."""
        identity = task.identity if isinstance(task, Version) else task
        return self._tag_call(activation, TaskAction.Label.CLOSE.value, {"identity": identity})

    def attach(self, activation: DIGActivation, target: Union[Version, str], payload: Any, *,
               source: Optional[str] = None) -> DIGEvent:
        """attach_h(phi): attach evidence to an exact task version. The
        declared source defaults to this activation's id."""
        return self._tag_call(activation, ATTACH_TOOL, {
            "target": self._task_id(target, "target"), "payload": payload, "source": source or activation.id,
        })

    def observe(self, activation: DIGActivation) -> DIGEvent:
        """observe: the task configuration now, handed back and recorded."""
        return self._tag_call(activation, OBSERVE_TOOL, {})

    # -- the record ------------------------------------------------------

    @classmethod
    def schema(cls) -> str:
        return SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the whole DT: agents, the environment's tool names, D,
        and T (whose calls are D's, referenced by issuer and call index)."""
        with self._lock:
            out = super().to_dict()
            out["tag"] = self.tag.to_dict(calls=False)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DIGTAG":
        dt = super().from_dict(data)
        dt.tag = TAGGraph.from_dict(
            data.get("tag") or {}, calls=lambda issuer, index: dt.dig.activations[issuer].calls[index],
        )
        return dt
