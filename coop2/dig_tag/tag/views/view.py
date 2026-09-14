"""The TAG view: L_T materialized, and the version / frontier / evidence
relations of one recorded task graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..graph import TAGGraph
from ...base import natural_sort_key
from ..action import TaskAction, close_target
from ..evidence import Evidence
from ..version import Version


@dataclass(frozen=True)
class TAGEdge:
    """One edge of L_T: `consume` is q -> a, `return` is a -> q."""

    source: str
    target: str
    relation: str

    def __str__(self) -> str:
        return f"{self.source} -> {self.target} ({self.relation})"


def tag_edges(graph: TAGGraph) -> List[TAGEdge]:
    """L_T = union over actions of {q -> a : q in Q_in} and {a -> q : q in Q_out}."""
    out: List[TAGEdge] = []
    for action in graph.actions.values():
        for task_id in action.input_ids:
            out.append(TAGEdge(task_id, action.id, "consume"))
        for task_id in action.output_ids:
            out.append(TAGEdge(action.id, task_id, "return"))
    return out


def inputs(graph: TAGGraph, action: TaskAction) -> List[Version]:
    """Q_in: the exact versions the action consumed."""
    return [graph.tasks[task_id] for task_id in action.input_ids]


def outputs(graph: TAGGraph, action: TaskAction) -> List[Version]:
    """Q_out: the fresh versions the action returned."""
    return [graph.tasks[task_id] for task_id in action.output_ids]


def versions(graph: TAGGraph, identity: str) -> List[Version]:
    """Every exact version carrying the identity, oldest first."""
    nodes = [task for task in graph.tasks.values() if task.identity == identity]
    return sorted(nodes, key=lambda task: (task.created_at or 0.0, natural_sort_key(task.id)))


def producing_action(graph: TAGGraph, task: Version) -> Optional[TaskAction]:
    """The action that returned this version, or None for an injected root."""
    for action in graph.actions.values():
        if task.id in action.output_ids:
            return action
    return None


def consuming_actions(graph: TAGGraph, task: Version) -> List[TaskAction]:
    """Every action that consumed this exact version (several: branching)."""
    return [action for action in graph.actions.values() if task.id in action.input_ids]


def frontier(graph: TAGGraph) -> List[Version]:
    """F_t, the structural frontier: versions no action has consumed, in Q
    order. CLOSE consumes nothing, so closing leaves it unchanged."""
    return [task for task_id, task in graph.tasks.items() if graph.on_frontier(task_id)]


@dataclass(frozen=True)
class TAGObservation:
    """The current task configuration:
    the open identities K_t and the structural frontier F_t, read from the
    record at the moment of building. Every identity starts open and stays
    so until it is closed (`closed_identities`); the frontier is every
    version no action consumed, which CLOSE leaves unchanged."""

    identities: List[str]          # every identity introduced, in order of introduction
    open_identities: List[str]     # K_t
    frontier: List[Version]       # F_t

    @property
    def closed_identities(self) -> List[str]:
        """Every introduced identity not in K_t."""
        open_set = set(self.open_identities)
        return [k for k in self.identities if k not in open_set]

    @property
    def open_frontier(self) -> List[Version]:
        """The frontier restricted to open tasks: {q in F_t : k in K_t}. An
        open identity need not have a frontier version."""
        open_set = set(self.open_identities)
        return [task for task in self.frontier if task.identity in open_set]

    def __str__(self) -> str:
        return (f"TAGObservation(open={self.open_identities}, closed={self.closed_identities}, "
                f"frontier={[q.id for q in self.frontier]})")


def closed_identities(graph: TAGGraph) -> List[str]:
    """The identities closed at the moment of reading, in order of
    introduction: those a CLOSE named and, reaching backward, what a
    closed task was made of -- a join's inputs once the identity it
    returned is closed, a split's input once every part it returned is.
    Closure never reaches back through an edit, an update, or a branch:
    the line they consumed from goes on."""
    done = {close_target(a.call) for a in graph.actions.values() if a.label is TaskAction.Label.CLOSE}
    while True:
        reached = set()
        for action in graph.actions.values():
            if action.label not in (TaskAction.Label.JOIN, TaskAction.Label.SPLIT):
                continue
            if all(q.identity in done for q in action.outputs):
                reached |= {graph.tasks[q].identity for q in action.input_ids} - done
        if not reached:
            break
        done |= reached
    return [k for k in dict.fromkeys(task.identity for task in graph.tasks.values()) if k in done]


def build_observation(graph: TAGGraph) -> TAGObservation:
    """Build the task configuration from the record: the identities
    introduced in task versions, K_t (those not closed), and F_t."""
    introduced = list(dict.fromkeys(task.identity for task in graph.tasks.values()))
    closed = set(closed_identities(graph))
    return TAGObservation(
        identities=introduced,
        open_identities=[k for k in introduced if k not in closed],
        frontier=frontier(graph),
    )


def history(graph: TAGGraph, identity: str) -> List[Tuple[Optional[TaskAction], Version]]:
    """The identity's work history: each of its versions in order with
    the action that returned it (None for an initial version)."""
    return [(producing_action(graph, task), task) for task in versions(graph, identity)]


def relations(graph: TAGGraph, identity: str) -> Dict[str, Any]:
    """How the identity stands to the others: `origin`, the action that
    introduced it and the versions it consumed (None for an open or an
    initial version), and `successors`, the actions that consumed one of
    its versions and returned another identity's, with those identities."""
    first = versions(graph, identity)[0]
    introduced = producing_action(graph, first)
    origin = None
    if introduced is not None and introduced.input_ids:
        origin = {
            "tool": introduced.label.value, "issuer": introduced.issuer,
            "from": [{"version": q, "identity": graph.tasks[q].identity} for q in introduced.input_ids],
        }
    successors = []
    for task in versions(graph, identity):
        for action in consuming_actions(graph, task):
            others = list(dict.fromkeys(q.identity for q in action.outputs if q.identity != identity))
            if others:
                successors.append({"tool": action.label.value, "issuer": action.issuer, "from": task.id,
                                   "identities": others})
    return {"origin": origin, "successors": successors}


def evidence_for(graph: TAGGraph, task: Version) -> List[Evidence]:
    """Attachments targeting this exact version -- never a predecessor's."""
    return [phi for phi in graph.evidence.values() if phi.target == task.id]


def evaluate(graph: TAGGraph, task: Version) -> Any:
    """c(g, sigma): apply the version's rule to its goal and state. Opt-in:
    the TAG itself never evaluates, and only a callable rule can be."""
    rule = task.spec.rule
    if not callable(rule):
        raise TypeError(
            f"task {task.id} has a non-callable rule {rule!r}; the TAG stores rules opaquely"
        )
    return rule(task.spec.goal, task.state)
