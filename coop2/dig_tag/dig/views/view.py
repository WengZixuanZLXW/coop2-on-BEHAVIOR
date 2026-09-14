"""The DIG view: L_D materialized, and the producer / recipient / provenance
relations of one recorded run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..interface import DIGParallelInterface
from ...base import ToolCall
from ..activation import DIGActivation
from ..event import DIGEvent
from ..tools import SEND_TOOL


@dataclass(frozen=True)
class DIGEdge:
    """One edge of L_D: `input` is e -> h (presentation), `return` is h -> e
    (returned during h by the call at `call_index`)."""

    source: str
    target: str
    relation: str
    call_index: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.source} -> {self.target} ({self.relation})"


def dig_edges(dt: DIGParallelInterface) -> List[DIGEdge]:
    """L_D = {e -> h : e in E_h} union {h -> e : e in E_r for some r}."""
    out: List[DIGEdge] = []
    for activation in dt.dig.activations.values():
        for event in activation.inputs:
            out.append(DIGEdge(event.id, activation.id, "input"))
        for call in activation.calls:
            for event in call.returned:
                out.append(DIGEdge(activation.id, event.id, "return", call.index))
    return out


def roots(dt: DIGParallelInterface) -> List[DIGEvent]:
    """Events with no producing call: the run's initial inputs and anything
    injected later."""
    return [event for event in dt.dig.events.values() if event.producer is None]


def producer(dt: DIGParallelInterface, event: DIGEvent) -> Optional[Tuple[DIGActivation, ToolCall]]:
    """The call-return pair that inserted the event, or None for a root."""
    if event.producer is None:
        return None
    activation = dt.dig.activations[event.producer.activation_id]
    return activation, activation.calls[event.producer.call_index]


def delivery(dt: DIGParallelInterface, activation: DIGActivation, event: DIGEvent) -> str:
    """How an input of the activation reached its agent: "send" (a send
    call named the agent), "root" (an initial presentation), the executor
    that routed it (runtime-defined routing), or "own" (the agent's own
    earlier return, retained as context)."""
    routed = dt.dig.routed.get(activation.agent_id, {})
    if event.id in routed:
        return routed[event.id]
    if event.id not in dt.dig.inbox.get(activation.agent_id, {}):
        return "own"
    return "root" if event.producer is None else "send"


def presented_to(dt: DIGParallelInterface, event: DIGEvent) -> List[DIGActivation]:
    """Every activation that had the event in E_h, in H order."""
    return [
        activation
        for activation in dt.dig.activations.values()
        if any(candidate is event for candidate in activation.inputs)
    ]


def recipients(dt: DIGParallelInterface, event: DIGEvent) -> List[str]:
    """The agents named by send calls of this event, in send order,
    without repeats: the DECLARED recipients (presentation is `presented_to`)."""
    out: List[str] = []
    for activation in dt.dig.activations.values():
        for call in activation.calls:
            if call.tool != SEND_TOOL or call.args.get("event") != event.id:
                continue
            for agent_id in call.args.get("to", []):
                if agent_id not in out:
                    out.append(agent_id)
    return out


@dataclass(frozen=True)
class Provenance:
    """Where an event comes from: its declared origin, and structurally the
    agent and tool of the producing call (None for a root)."""

    origin: Optional[str]
    agent_id: Optional[str]
    tool: Optional[str]
    producer: Optional[DIGEvent.Producer]

    def __str__(self) -> str:
        if self.producer is None:
            return f"root (origin {self.origin})"
        return f"{self.agent_id} via {self.tool} at {self.producer} (origin {self.origin})"


def provenance(dt: DIGParallelInterface, event: DIGEvent) -> Provenance:
    """Agent and tool provenance of an event, beside its declared origin."""
    pair = producer(dt, event)
    if pair is None:
        return Provenance(event.origin, None, None, None)
    activation, call = pair
    return Provenance(event.origin, activation.agent_id, call.tool, event.producer)
