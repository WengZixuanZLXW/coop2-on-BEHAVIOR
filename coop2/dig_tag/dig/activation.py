"""DIGActivation: one activation of an agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..base import ToolCall
from .event import DIGEvent

DIG_METADATA = "dig"
DIG_ERROR = "error"


@dataclass(eq=False)
class DIGActivation:
    """h[t:t'] = (i, E_h, U_h): one activation of agent i.

    `inputs` is E_h, fixed when the activation opens; `calls` is U_h, the
    ordered call-return pairs appended as the activation issues them. The
    node is in H from the moment it opens (the record is prefix-realized),
    and closing only stamps t'. An activation that raised keeps everything
    it issued and notes the error under `metadata["dig"]["error"]`."""

    agent_id: str
    inputs: List[DIGEvent] = field(default_factory=list)
    calls: List[ToolCall] = field(default_factory=list)
    id: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise TypeError(
                f"DIGActivation.agent_id must be a non-empty str; got {self.agent_id!r}"
            )
        for event in self.inputs:
            if not isinstance(event, DIGEvent):
                raise TypeError(
                    f"DIGActivation.inputs must contain DIGEvent instances; got {event!r}"
                )
        for call in self.calls:
            if not isinstance(call, ToolCall):
                raise TypeError(
                    f"DIGActivation.calls must contain ToolCall instances; got {call!r}"
                )
        if not isinstance(self.metadata, dict):
            raise TypeError("DIGActivation.metadata must be a dict")
        dig_metadata = self.metadata.get(DIG_METADATA)
        if dig_metadata is not None and not isinstance(dig_metadata, dict):
            raise TypeError('DIGActivation.metadata["dig"] must be a dict')

    @property
    def recorded(self) -> bool:
        return self.id is not None

    @property
    def is_open(self) -> bool:
        """In H and not yet closed."""
        return self.recorded and self.ended_at is None

    @property
    def input_ids(self) -> List[str]:
        return [event.id for event in self.inputs]

    @property
    def returned(self) -> List[DIGEvent]:
        """Every event returned by this activation's calls, in call order."""
        return [event for call in self.calls for event in call.returned]

    def available(self, before: Optional[int] = None) -> List[DIGEvent]:
        """Avail(h, r): the inputs plus the returns of every call with index
        below r, in presentation order without duplicates. `before=None`
        means the next call to be issued."""
        upto = len(self.calls) if before is None else before
        out: List[DIGEvent] = []
        seen: set = set()
        for event in list(self.inputs) + [
            event for call in self.calls[:upto] for event in call.returned
        ]:
            if id(event) in seen:
                continue
            seen.add(id(event))
            out.append(event)
        return out

    def available_ids(self, before: Optional[int] = None) -> List[str]:
        return [event.id for event in self.available(before)]

    @property
    def error(self) -> Optional[Dict[str, Any]]:
        """The error noted by `mark_error`, if the activation raised."""
        value = self.metadata.get(DIG_METADATA, {}).get(DIG_ERROR)
        return dict(value) if value is not None else None

    def mark_error(self, error: BaseException) -> None:
        """Note that this activation raised. What it already issued stays
        recorded; the note keeps the attempt observable."""
        self.metadata.setdefault(DIG_METADATA, {})[DIG_ERROR] = {
            "type": type(error).__name__,
            "message": str(error),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Inputs are referenced by id; calls carry their returns by id."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "inputs": self.input_ids,
            "calls": [call.to_dict() for call in self.calls],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        events: Mapping[str, DIGEvent],
    ) -> "DIGActivation":
        """Rebuild a serialized activation, re-linking inputs and returns to
        the graph's event objects by id (one object per event)."""
        inputs: List[DIGEvent] = []
        for event_id in data.get("inputs") or []:
            if event_id not in events:
                raise ValueError(
                    f"activation {data.get('id')!r} references unknown event id {event_id!r}"
                )
            inputs.append(events[event_id])
        return cls(
            agent_id=data["agent_id"],
            inputs=inputs,
            calls=[ToolCall.from_dict(call, events) for call in data.get("calls") or []],
            id=data.get("id"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            metadata=dict(data.get("metadata") or {}),
        )

    def __str__(self) -> str:
        inputs = ",".join(event.id or "-" for event in self.inputs) or "-"
        status = "open" if self.is_open else ("error" if self.error else "closed")
        return (
            f"DIGActivation(id={self.id or '-'}, agent={self.agent_id}, "
            f"inputs=[{inputs}], calls={len(self.calls)}, {status})"
        )
