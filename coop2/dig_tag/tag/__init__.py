"""T on its own: the Task Activity Graph.

  spec.py         TaskSpec, the goal and the rule
  version.py      Version, one exact task version
  action.py       TaskAction, a task call bound to its issuer; the TAG's tools: the six task tools, attach, observe
  evidence.py     Evidence, an attachment to an exact version
  graph.py        TAGGraph, T = (Q, A, Phi, L_T): the stores, the six transitions with the branch rule, attach
  interface.py    TAGParallelInterface, the interface over T, and what a tool hands back (its return policy)
  views/          ways of looking at one recorded T

`from dig_tag import tag` imports nothing of the DIG."""

from .action import ATTACH_TOOL, OBSERVE_TOOL, TAG_TOOLS, TASK_TOOLS, TaskAction, close_target, task_input_ids
from .interface import (
    DEFAULT_RETURNS, Return, TAGParallelInterface, apply, current, observation, observation_with_history, observed,
    produced, returns_for,
)
from .evidence import Evidence
from .graph import TAGGraph
from .spec import TaskSpec
from .version import Version

__all__ = [
    "TaskSpec", "Version", "TaskAction", "Evidence",
    "TASK_TOOLS", "ATTACH_TOOL", "OBSERVE_TOOL", "TAG_TOOLS", "task_input_ids", "close_target",
    "TAGGraph", "TAGParallelInterface",
    "Return", "observation", "observation_with_history", "produced", "DEFAULT_RETURNS", "returns_for", "apply",
    "observed", "current",
]
