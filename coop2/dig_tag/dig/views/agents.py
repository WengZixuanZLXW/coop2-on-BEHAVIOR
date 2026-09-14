"""The agent view: one node per agent, the run's activations aggregated.

Two ways to draw an edge between agents, kept apart because they answer
different questions. INTERACTION edges come from send calls: X -> Y when
an activation of X sent an event to Y (a relay by X of someone else's
event is X's interaction). FLOW edges come from provenance: X -> Y when an
event returned in an activation of X was presented to an activation of Y
(a self pair is an agent re-reading its own returns). Roots have no agent,
so they draw no edge in either."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..interface import DIGParallelInterface
from ..activation import DIGActivation
from ..tools import SEND_TOOL


@dataclass
class AgentGraphEdge:
    """A directed edge between two agents and the events that drew it."""

    source: str
    target: str
    event_ids: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.event_ids)

    def __str__(self) -> str:
        return f"{self.source} -> {self.target} ({self.count})"


@dataclass
class AgentGraph:
    """Agents as nodes, aggregated edges between them."""

    agents: List[str]
    edges: List[AgentGraphEdge]

    @property
    def edge_set(self) -> Set[Tuple[str, str]]:
        return {(edge.source, edge.target) for edge in self.edges}

    def successors(self, agent_id: str) -> List[str]:
        return [edge.target for edge in self.edges if edge.source == agent_id]

    def predecessors(self, agent_id: str) -> List[str]:
        return [edge.source for edge in self.edges if edge.target == agent_id]

    def __str__(self) -> str:
        edges = ", ".join(str(edge) for edge in self.edges) or "-"
        return f"AgentGraph(agents={self.agents}, edges=[{edges}])"


def _aggregate(agents: List[str], pairs: List[Tuple[str, str, str]]) -> AgentGraph:
    edges: Dict[Tuple[str, str], AgentGraphEdge] = {}
    for source, target, event_id in pairs:
        edge = edges.setdefault((source, target), AgentGraphEdge(source, target))
        edge.event_ids.append(event_id)
    return AgentGraph(agents=list(agents), edges=list(edges.values()))


def interaction_graph(
    dt: DIGParallelInterface,
    *,
    activations: Optional[Iterable[DIGActivation]] = None,
    relabel: Optional[Callable[[str], str]] = None,
) -> AgentGraph:
    """Who sends to whom: one edge per (sender, recipient) pair
    across the send calls of the selected activations (all by default),
    carrying the ids of the events sent. `relabel` maps an identity to the
    vertex it is aggregated under -- its retained source-node label, say,
    which merges the workers a Send instantiated into the one node the
    graph declares."""
    label = relabel or (lambda agent_id: agent_id)
    selected = dt.dig.activations.values() if activations is None else activations
    pairs: List[Tuple[str, str, str]] = []
    for activation in selected:
        for call in activation.calls:
            if call.tool != SEND_TOOL:
                continue
            for recipient in call.args.get("to", []):
                pairs.append((label(activation.agent_id), label(recipient), call.args["event"]))
    return _aggregate(list(dict.fromkeys(label(agent_id) for agent_id in dt.agents)), pairs)


def flow_graph(dt: DIGParallelInterface) -> AgentGraph:
    """Whose returns reach whose activations: one edge per (producer agent,
    presented agent) pair, carrying the ids of the events that flowed."""
    pairs: List[Tuple[str, str, str]] = []
    for activation in dt.dig.activations.values():
        for event in activation.inputs:
            if event.producer is None:
                continue
            source = dt.dig.activations[event.producer.activation_id].agent_id
            pairs.append((source, activation.agent_id, event.id))
    return _aggregate(dt.agents, pairs)


def interaction_edges(dt: DIGParallelInterface) -> Set[Tuple[str, str]]:
    """The interaction graph's edge set: the run's topology as sent."""
    return interaction_graph(dt).edge_set
