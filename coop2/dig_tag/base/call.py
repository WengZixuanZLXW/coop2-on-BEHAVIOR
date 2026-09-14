"""ToolCall: one tool call and what it returned, the record element both
graphs share -- a call-return pair in an activation for the DIG, the call
an action realized for the TAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(eq=False)
class ToolCall:
    """(u_r, E_r): one occurrence-distinct tool call and its returned nodes.

    `tool` and `args` are the call as issued (references are by id, so args
    stay JSON-able); `index`, `returned`, and `at` are filled when the call
    is realized. Two calls with the same tool and arguments are still
    distinct occurrences."""

    class Label(str, Enum):
        """The tool family: U_T, U_Z, or U_phi."""

        TASK = "task"
        ENVIRONMENT = "environment"
        INFORMATION = "information"

        def __str__(self) -> str:
            return self.value

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    returned: List[Any] = field(default_factory=list)
    at: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool, str) or not self.tool:
            raise TypeError(f"ToolCall.tool must be a non-empty str; got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise TypeError(f"ToolCall.args must be a dict; got {type(self.args).__name__}")
        for node in self.returned:
            if not hasattr(node, "id"):
                raise TypeError(f"ToolCall.returned must contain recorded nodes; got {node!r}")

    @property
    def label(self) -> "ToolCall.Label":
        """The family derives from the tool name through the declared
        vocabulary (`declare_tools`): the DIG declares send, the TAG its
        task tools and attach; every other name is an environment tool."""
        return FAMILIES.get(self.tool, ToolCall.Label.ENVIRONMENT)

    @property
    def recorded(self) -> bool:
        return self.index is not None

    @property
    def returned_ids(self) -> List[str]:
        return [node.id for node in self.returned]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "index": self.index,
            "returned": self.returned_ids,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], nodes: Mapping[str, Any]) -> "ToolCall":
        """Rebuild a serialized call, re-linking returned nodes by id."""
        returned: List[Any] = []
        for node_id in data.get("returned") or []:
            if node_id not in nodes:
                raise ValueError(f"call references unknown node id {node_id!r}")
            returned.append(nodes[node_id])
        index = data.get("index")
        return cls(
            tool=data["tool"],
            args=dict(data.get("args") or {}),
            index=int(index) if index is not None else None,
            returned=returned,
            at=data.get("at"),
        )

    def __str__(self) -> str:
        returned = ",".join(node.id or "-" for node in self.returned) or "-"
        index = self.index if self.index is not None else "-"
        return (
            f"ToolCall(index={index}, tool={self.tool}, label={self.label.value}, "
            f"args={self.args}, returned=[{returned}])"
        )


FAMILIES: Dict[str, ToolCall.Label] = {}


def declare_tools(label: ToolCall.Label, names: Iterable[str]) -> None:
    """Enter tool names into the vocabulary under a family. A name declared
    twice keeps the later family; a name never declared is an environment
    tool."""
    for name in names:
        FAMILIES[name] = label
