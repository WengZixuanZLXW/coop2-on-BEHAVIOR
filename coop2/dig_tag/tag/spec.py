"""TaskSpec: what a task asks for and how it is judged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class TaskSpec:
    """delta = (g, c): the task goal and its task-specific evaluation rule.

    Both fields are opaque to the substrate -- language, predicates,
    programs, whatever the domain uses. Recording them establishes nothing
    about their correctness, and the TAG never evaluates the rule."""

    goal: Any
    rule: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "rule": self.rule}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        return cls(goal=data.get("goal"), rule=data.get("rule"))

    def __str__(self) -> str:
        rule = f", rule={self.rule!r}" if self.rule is not None else ""
        return f"TaskSpec(goal={self.goal!r}{rule})"
