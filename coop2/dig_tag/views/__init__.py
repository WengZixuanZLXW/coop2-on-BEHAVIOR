"""Ways of looking at one recorded DT, as plain functions taking the
`DIGTAG`. The DIG's views apply as they are (a DIGTAG is a DIG); the TAG's
views take its graph, so here each is handed the DT's T. Views never own
state and the interface never imports them."""

from ..dig.views import (
    AgentGraph,
    AgentGraphEdge,
    DIGEdge,
    EnvironmentTransition,
    Provenance,
    ancestry,
    available,
    delivery,
    dependents,
    dig_edges,
    environment_calls,
    feedback,
    flow_graph,
    interaction_edges,
    interaction_graph,
    presented_to,
    producer,
    provenance,
    recipients,
    roots,
    transitions,
)
from ..tag import views as _tag
from ..tag.views import TAGEdge, TAGObservation


def tag_edges(dt):
    """L_T of the DT's T."""
    return _tag.tag_edges(dt.tag)


def inputs(dt, action):
    """Q_in of an action in the DT's T."""
    return _tag.inputs(dt.tag, action)


def outputs(dt, action):
    """Q_out of an action in the DT's T."""
    return _tag.outputs(dt.tag, action)


def versions(dt, identity):
    """Every version carrying the identity in the DT's T, oldest first."""
    return _tag.versions(dt.tag, identity)


def producing_action(dt, task):
    """The action that returned the version, or None for a root."""
    return _tag.producing_action(dt.tag, task)


def consuming_actions(dt, task):
    """Every action that consumed the version."""
    return _tag.consuming_actions(dt.tag, task)


def frontier(dt):
    """F_t of the DT's T."""
    return _tag.frontier(dt.tag)


def build_observation(dt):
    """The task configuration of the DT's T: K_t and F_t."""
    return _tag.build_observation(dt.tag)


def closed_identities(dt):
    """The closed identities of the DT's T, closure reaching back through joins and splits."""
    return _tag.closed_identities(dt.tag)


def history(dt, identity):
    """The identity's versions in order, each with the action that returned it."""
    return _tag.history(dt.tag, identity)


def relations(dt, identity):
    """What the identity came from and what came from it."""
    return _tag.relations(dt.tag, identity)


def evidence_for(dt, task):
    """The attachments targeting the exact version."""
    return _tag.evidence_for(dt.tag, task)


def evaluate(dt, task):
    """Apply the version's rule to its goal and state (opt-in)."""
    return _tag.evaluate(dt.tag, task)


def lineage(dt, task):
    """The versions this one descends from, nearest first."""
    return _tag.lineage(dt.tag, task)


def descendants(dt, task):
    """The versions derived from this one, nearest first."""
    return _tag.descendants(dt.tag, task)


__all__ = [
    "DIGEdge", "Provenance", "dig_edges", "roots", "producer", "presented_to", "delivery", "recipients",
    "provenance", "available", "ancestry", "dependents",
    "AgentGraph", "AgentGraphEdge", "interaction_graph", "interaction_edges", "flow_graph",
    "EnvironmentTransition", "transitions", "environment_calls", "feedback",
    "TAGEdge", "TAGObservation", "tag_edges", "inputs", "outputs", "versions", "producing_action",
    "consuming_actions", "frontier", "build_observation", "closed_identities", "evidence_for", "evaluate", "lineage",
    "descendants", "history", "relations",
]
