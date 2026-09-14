"""TAGGraph: the representation T = (Q, A, Phi, L_T)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from ..base import Minter, ToolCall, expect_type
from .action import TaskAction, close_target, task_input_ids
from .evidence import Evidence
from .spec import TaskSpec
from .version import Version


class TAGGraph:
    """The TAG REPRESENTATION: the version store Q, the action store A, and
    the evidence store Phi. L_T is not stored -- it derives from each
    action's consumed (`input_ids`) and returned (`outputs`) versions; every
    other derived read (versions, lineage, frontier) lives in
    `dig_tag.tag.views`. The graph answers nothing.

    A task call is realized in two steps: `transition` checks the
    structural signature of the tool against Q and builds the fresh output
    versions; `record` stores the action with those outputs, minting their
    ids. `attach` adds evidence; `record_root` admits a version that
    entered with no action. T is its own record: nothing in it is also in
    D, which holds what the calls handed back."""

    tasks: Dict[str, Version]
    actions: Dict[str, TaskAction]
    evidence: Dict[str, Evidence]

    def __init__(self) -> None:
        self.tasks = {}
        self.actions = {}
        self.evidence = {}
        self._identities: Set[str] = set()   # an index over `tasks`
        self._consumed: Set[str] = set()     # an index over `actions`: every version some action consumed
        self._versions = Minter("q")
        self._actions = Minter("a")
        self._evidence = Minter("phi")
        self._identity_minter = Minter("k")

    # -- reads the substrate needs itself -------------------------------

    @property
    def identities(self) -> Set[str]:
        """Every task identity in use."""
        return set(self._identities)

    def expect_task(self, task_id: Any) -> Version:
        """The exact version named by `task_id`, which must be in Q."""
        if not isinstance(task_id, str) or task_id not in self.tasks:
            raise ValueError(f"version {task_id!r} is not in the TAG")
        return self.tasks[task_id]

    def on_frontier(self, task_id: str) -> bool:
        """Whether no recorded action has consumed this version: it is its
        identity's current version, the one an action may continue."""
        return task_id in self.tasks and task_id not in self._consumed

    def expect_identity(self, identity: Any) -> str:
        """A task identity in use, the target a CLOSE must name."""
        if not isinstance(identity, str) or identity not in self._identities:
            raise ValueError(f"task identity {identity!r} is not in the TAG")
        return identity

    def mint_identity(self) -> str:
        return self._identity_minter.next(self._identities)

    def _resolve_identity(self, supplied: Any, inputs: Sequence[Version]) -> str:
        """An output identity supplied by SPLIT or JOIN must continue the
        identity of an input that is on the frontier, or introduce one not
        yet in use; None mints. An identity is continued only from its
        current version: from a consumed version the action branches."""
        if supplied is None:
            return self.mint_identity()
        if not isinstance(supplied, str) or not supplied:
            raise ValueError(f"identity must be a non-empty str or None; got {supplied!r}")
        for task in inputs:
            if task.identity == supplied:
                if self.on_frontier(task.id):
                    return supplied
                raise ValueError(
                    f"identity {supplied!r} cannot continue from {task.id}, which is not its "
                    "frontier version: a consumed version only branches (identity None)"
                )
        if supplied not in self._identities:
            return supplied
        raise ValueError(
            f"identity {supplied!r} is in use by another task and is not an input "
            "identity of this call"
        )

    def _continued(self, task: Version) -> str:
        """The identity an EDIT or UPDATE of `task` carries: the task's own
        while it is the frontier version, a fresh one once it is consumed
        (the branch)."""
        return task.identity if self.on_frontier(task.id) else self.mint_identity()

    @staticmethod
    def _spec_arg(args: Mapping[str, Any], key: str = "spec") -> TaskSpec:
        value = args.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"`{key}` must be a spec mapping with goal and rule")
        return TaskSpec.from_dict(value)

    @staticmethod
    def _state_arg(args: Mapping[str, Any], key: str = "state") -> Any:
        if key not in args:
            raise ValueError(f"`{key}` is required")
        return args[key]

    # -- the task-tool transitions ---------------------------------------

    def transition(self, call: ToolCall) -> List[Version]:
        """Check a task call's structural signature against Q and build its
        fresh output versions (not yet in Q; `record` stores them). Every
        output is a new token; no input is touched.

        OPEN(delta, sigma): {} -> q, fresh identity.
        EDIT(q, delta'): q -> (k, delta', sigma).
        UPDATE(q, sigma'): q -> (k, delta, sigma').
        SPLIT(q, parts): q -> {q_j}, m >= 2, one identity per part: at
        most one part continues the input identity, the others are fresh
        (None mints) or previously unused, all distinct.
        JOIN(qs, k', delta', sigma'): {q_j} -> q', m >= 2 versions of
        distinct task identities (a join integrates tasks; it never joins
        two versions of one identity); k' may continue an input identity
        or introduce a new one.
        CLOSE(k): Q_in = Q_out = {}; k must be in use. Closing changes no
        version and no frontier: whether k is open derives from the
        recorded CLOSE targets (`views.build_observation`).

        An identity is continued only from its frontier version, so each
        identity has one line of versions and at most one current one. An
        action on a version some action has already consumed BRANCHES: an
        EDIT or UPDATE of it carries a fresh identity, a SPLIT of it gives
        every part a fresh identity, and a JOIN may take it as an input
        but not continue its identity. Two agents updating the same version
        concurrently therefore leave two tasks, not two versions of one."""
        expect_type(call, ToolCall, what="TAGGraph.transition")
        if call.label is not ToolCall.Label.TASK:
            raise ValueError(f"{call.tool!r} is not a task tool")
        inputs = [self.expect_task(task_id) for task_id in task_input_ids(call)]
        return self._TRANSITIONS[TaskAction.Label(call.tool)](self, call, inputs)

    def _open(self, call: ToolCall, inputs: List[Version]) -> List[Version]:
        """OPEN: one version with a fresh identity."""
        return [Version(identity=self.mint_identity(), spec=self._spec_arg(call.args),
                                  state=call.args.get("state"))]

    def _edit(self, call: ToolCall, inputs: List[Version]) -> List[Version]:
        """EDIT: the spec changed, the state carried over."""
        (task,) = inputs
        return [Version(identity=self._continued(task), spec=self._spec_arg(call.args), state=task.state)]

    def _update(self, call: ToolCall, inputs: List[Version]) -> List[Version]:
        """UPDATE: the state changed, the spec carried over."""
        (task,) = inputs
        return [Version(identity=self._continued(task), spec=task.spec, state=self._state_arg(call.args))]

    def _split(self, call: ToolCall, inputs: List[Version]) -> List[Version]:
        """SPLIT: at least two parts, each with its own identity."""
        parts = call.args.get("parts")
        if not isinstance(parts, (list, tuple)) or len(parts) < 2:
            raise ValueError("split requires `parts`: at least two (identity, spec, state) parts")
        out: List[Version] = []
        for part in parts:
            if not isinstance(part, Mapping):
                raise ValueError("each split part must be a mapping with identity, spec, state")
            out.append(Version(
                identity=self._resolve_identity(part.get("identity"), inputs),
                spec=self._spec_arg(part),
                state=part.get("state"),
            ))
        if len({node.identity for node in out}) < len(out):
            raise ValueError(
                "split parts must carry distinct identities: at most one part continues "
                "the input identity, the others are new"
            )
        return out

    def _join(self, call: ToolCall, inputs: List[Version]) -> List[Version]:
        """JOIN: one version from at least two versions of distinct identities."""
        if len(inputs) < 2 or len({task.identity for task in inputs}) < len(inputs):
            raise ValueError("join requires at least two input versions of distinct task identities")
        return [Version(
            identity=self._resolve_identity(call.args.get("identity"), inputs),
            spec=self._spec_arg(call.args),
            state=call.args.get("state"),
        )]

    def _close(self, call: ToolCall, inputs: List[Version]) -> List[Version]:
        """CLOSE: names an identity in use; no version."""
        self.expect_identity(close_target(call))
        return []

    _TRANSITIONS = {
        TaskAction.Label.OPEN: _open,
        TaskAction.Label.EDIT: _edit,
        TaskAction.Label.UPDATE: _update,
        TaskAction.Label.SPLIT: _split,
        TaskAction.Label.JOIN: _join,
        TaskAction.Label.CLOSE: _close,
    }

    def record(self, action: TaskAction) -> TaskAction:
        """T (+) (a, Q): store an action and the versions it returned,
        minting their ids."""
        expect_type(action, TaskAction, what="TAGGraph.record")
        if action.recorded:
            raise ValueError(f"task action {action.id} is already recorded")
        for task_id in action.input_ids:
            self.expect_task(task_id)
        for node in action.outputs:
            if node.id is not None and node.id in self.tasks:
                raise ValueError(f"version {node.id} is already in the TAG")
        action.id = self._actions.next(self.actions)
        self.actions[action.id] = action
        self._consumed.update(action.input_ids)
        now = time.time()
        for node in action.outputs:
            if node.id is None:
                node.id = self._versions.next(self.tasks)
                node.created_at = now
            self.tasks[node.id] = node
            self._identities.add(node.identity)
        return action

    def record_root(self, task: Version) -> Version:
        """Admit a version that entered with no action (an injected initial
        task) into Q. A missing id or identity is minted."""
        expect_type(task, Version, what="TAGGraph.record_root")
        if task.id is not None and task.id in self.tasks:
            raise ValueError(f"version {task.id} is already in the TAG")
        if task.id is None:
            task.id = self._versions.next(self.tasks)
            task.created_at = time.time()
        if task.identity is None:
            task.identity = self.mint_identity()
        self.tasks[task.id] = task
        self._identities.add(task.identity)
        return task

    def attach(self, evidence: Evidence) -> Evidence:
        """T (+) phi: add an attachment targeting an exact version in Q. No
        action node, no transition edge. Phi is a set of triples: repeating
        a recorded (source, payload, target) leaves it unchanged and
        returns the existing attachment (the repeated call stays in U_h)."""
        expect_type(evidence, Evidence, what="TAGGraph.attach")
        if evidence.recorded:
            raise ValueError(f"evidence {evidence.id} is already recorded")
        self.expect_task(evidence.target)
        for existing in self.evidence.values():
            if (existing.source, existing.target) == (evidence.source, evidence.target) \
                    and existing.payload == evidence.payload:
                return existing
        evidence.id = self._evidence.next(self.evidence)
        evidence.created_at = time.time()
        self.evidence[evidence.id] = evidence
        return evidence

    # -- the record ------------------------------------------------------

    def to_dict(self, *, calls: bool = True) -> Dict[str, Any]:
        """The record of T: its versions, actions, and evidence. The calls
        belong to it when it stands alone (`calls`); in DIG-TAG they are the
        DIG's, referenced by issuer and call index."""
        out: Dict[str, Any] = {
            "versions": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "actions": {action_id: action.to_dict() for action_id, action in self.actions.items()},
            "evidence": {evidence_id: evidence.to_dict() for evidence_id, evidence in self.evidence.items()},
            "counters": {
                "q": self._versions.counter, "a": self._actions.counter,
                "phi": self._evidence.counter, "k": self._identity_minter.counter,
            },
        }
        if calls:
            out["calls"] = {action_id: action.call.to_dict() for action_id, action in self.actions.items()}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, calls: Optional[Callable[[str, int], ToolCall]] = None) -> "TAGGraph":
        """Rebuild a recorded TAG. A record that stands alone carries its
        calls; one inside DIG-TAG references the DIG's, given as `calls`
        (by issuer and call index)."""
        graph = cls()
        counters = data.get("counters") or {}
        graph._versions = Minter("q", int(counters.get("q") or 0))
        graph._actions = Minter("a", int(counters.get("a") or 0))
        graph._evidence = Minter("phi", int(counters.get("phi") or 0))
        graph._identity_minter = Minter("k", int(counters.get("k") or 0))
        for task_id, payload in (data.get("versions") or {}).items():
            node = Version.from_dict(payload)
            graph.tasks[task_id] = node
            graph._identities.add(node.identity)
        for action_id, payload in (data.get("actions") or {}).items():
            if "calls" in data:
                call = ToolCall.from_dict(data["calls"][action_id], {})
            elif calls is not None:
                call = calls(payload["issuer"], int(payload["call_index"]))
            else:
                raise ValueError(f"action {action_id} references a call this record does not carry")
            outputs = [graph.expect_task(task_id) for task_id in payload.get("outputs") or []]
            action = TaskAction(issuer=payload["issuer"], call=call, id=action_id, outputs=outputs)
            graph.actions[action_id] = action
            graph._consumed.update(action.input_ids)
        for evidence_id, payload in (data.get("evidence") or {}).items():
            graph.evidence[evidence_id] = Evidence.from_dict(payload)
        return graph
