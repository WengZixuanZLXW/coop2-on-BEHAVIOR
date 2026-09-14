"""Ways of looking at one recorded D, as plain functions taking the DIG
interface (the coupled DIG-TAG interface is one):

  view.py         L_D as a trace; roots, producers, deliveries, recipients, provenance
  ancestry.py     the availability-based dependency analysis
  agents.py       one node per agent: interaction (who sends to whom) and flow (whose returns reach whom)
  environment.py  the calls on Z, their feedback, the transition sequence
"""

from .agents import AgentGraph, AgentGraphEdge, flow_graph, interaction_edges, interaction_graph
from .ancestry import ancestry, available, dependents
from .environment import EnvironmentTransition, environment_calls, feedback, transitions
from .view import (
    DIGEdge,
    Provenance,
    delivery,
    dig_edges,
    presented_to,
    producer,
    provenance,
    recipients,
    roots,
)

__all__ = [
    "DIGEdge", "Provenance", "dig_edges", "roots", "producer", "presented_to", "delivery", "recipients",
    "provenance", "available", "ancestry", "dependents",
    "AgentGraph", "AgentGraphEdge", "interaction_graph", "interaction_edges", "flow_graph",
    "EnvironmentTransition", "transitions", "environment_calls", "feedback",
]
