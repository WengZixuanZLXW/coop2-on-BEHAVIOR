"""TaskAction: a task call bound to whoever issued it, and the TAG's tools:
the six task tools' names and argument shapes, attach, and observe."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..base import ToolCall, declare_tools
from .version import Version

ATTACH_TOOL = "attach"
OBSERVE_TOOL = "observe"


@dataclass(eq=False)
class TaskAction:
    """a = (issuer, u): a task call bound to whoever issued it -- the
    activation, in DIG-TAG; the agent, when the TAG stands alone.

    Consumed versions (Q_in) derive from the call's arguments; the versions
    the transition returned (Q_out) are the action's `outputs`. What the
    call handed back to the agent is something else: information from T
    by the tool's return policy, which the DIG records as an event."""

    class Label(str, Enum):
        """The task tool the action applied."""

        OPEN = "open"
        EDIT = "edit"
        UPDATE = "update"
        SPLIT = "split"
        JOIN = "join"
        CLOSE = "close"

        def __str__(self) -> str:
            return self.value

    issuer: str
    call: ToolCall
    id: Optional[str] = None
    outputs: List[Version] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.issuer, str) or not self.issuer:
            raise TypeError(f"TaskAction.issuer must be a non-empty str; got {self.issuer!r}")
        if not isinstance(self.call, ToolCall):
            raise TypeError(f"TaskAction.call must be a ToolCall; got {self.call!r}")
        if self.call.label is not ToolCall.Label.TASK:
            raise ValueError(f"TaskAction.call must be a task tool call; got tool {self.call.tool!r}")
        if not all(isinstance(version, Version) for version in self.outputs):
            raise TypeError("TaskAction.outputs must contain Version")

    @property
    def label(self) -> "TaskAction.Label":
        return TaskAction.Label(self.call.tool)

    @property
    def input_ids(self) -> List[str]:
        """Q_in: the exact versions the call consumed, from its arguments."""
        return task_input_ids(self.call)

    @property
    def output_ids(self) -> List[str]:
        """Q_out, by id."""
        return [version.id for version in self.outputs]

    @property
    def recorded(self) -> bool:
        return self.id is not None

    def to_dict(self) -> Dict[str, Any]:
        """The call is referenced by (issuer, call index), the outputs by id."""
        return {"id": self.id, "issuer": self.issuer, "call_index": self.call.index, "outputs": self.output_ids}

    def __str__(self) -> str:
        inputs = ",".join(self.input_ids) or "-"
        outputs = ",".join(self.output_ids) or "-"
        return (
            f"TaskAction(id={self.id or '-'}, issuer={self.issuer}, "
            f"label={self.label.value}, inputs=[{inputs}], outputs=[{outputs}])"
        )


TASK_TOOLS = tuple(label.value for label in TaskAction.Label)
TAG_TOOLS = TASK_TOOLS + (ATTACH_TOOL, OBSERVE_TOOL)     # the TAG's whole action space

declare_tools(ToolCall.Label.TASK, TASK_TOOLS)
declare_tools(ToolCall.Label.INFORMATION, [ATTACH_TOOL, OBSERVE_TOOL])


def task_input_ids(call: ToolCall) -> List[str]:
    """The version ids a task call names as inputs, per tool signature:
    OPEN and CLOSE consume nothing (CLOSE names an identity, see
    `close_target`); EDIT / UPDATE / SPLIT consume `task`; JOIN consumes
    `tasks`."""
    label = TaskAction.Label(call.tool)
    if label in (TaskAction.Label.OPEN, TaskAction.Label.CLOSE):
        return []
    if label is TaskAction.Label.JOIN:
        tasks = call.args.get("tasks")
        if not isinstance(tasks, (list, tuple)) or not all(isinstance(task_id, str) for task_id in tasks):
            raise ValueError("join requires `tasks`: a list of version ids")
        return list(tasks)
    task = call.args.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{call.tool} requires `task`: a version id")
    return [task]


def close_target(call: ToolCall) -> str:
    """The task identity a CLOSE call names: Q_in = Q_out = {} and the
    target k is in the call."""
    if TaskAction.Label(call.tool) is not TaskAction.Label.CLOSE:
        raise ValueError(f"{call.tool} is not a close call")
    identity = call.args.get("identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("close requires `identity`: the task identity to mark closed")
    return identity
