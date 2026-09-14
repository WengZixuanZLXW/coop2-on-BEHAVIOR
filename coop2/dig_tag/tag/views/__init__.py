"""Ways of looking at one recorded T, as plain functions taking the
`TAGGraph` (DIG-TAG's `dig_tag.views` gives them the DT's T):

  view.py      L_T as a task graph; versions, the frontier, the task configuration, evidence
  lineage.py   the exact-version lineage analysis
"""

from .lineage import descendants, lineage
from .view import (
    TAGEdge,
    TAGObservation,
    build_observation,
    closed_identities,
    consuming_actions,
    evaluate,
    evidence_for,
    frontier,
    history,
    inputs,
    outputs,
    producing_action,
    relations,
    tag_edges,
    versions,
)

__all__ = [
    "TAGEdge", "TAGObservation", "tag_edges", "inputs", "outputs", "versions", "producing_action",
    "consuming_actions", "frontier", "build_observation", "closed_identities", "evidence_for", "evaluate", "lineage",
    "descendants", "history", "relations",
]
