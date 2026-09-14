"""The environment view: the calls that acted on Z, their feedback, and
the realized transition sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..interface import DIGParallelInterface
from ...base import ToolCall
from ..activation import DIGActivation
from ..event import DIGEvent


@dataclass(frozen=True)
class EnvironmentTransition:
    """One realized Z -(h, u) / E_Z-> Z': the issuing activation, the call,
    and the externalized feedback."""

    activation: DIGActivation
    call: ToolCall

    @property
    def feedback(self) -> List[DIGEvent]:
        return list(self.call.returned)

    def __str__(self) -> str:
        returned = ",".join(event.id for event in self.call.returned) or "-"
        return f"{self.activation.agent_id}@{self.activation.id}: {self.call.tool} -> [{returned}]"


def transitions(dt: DIGParallelInterface) -> List[EnvironmentTransition]:
    """Every environment call, in the order it was realized."""
    out = [
        EnvironmentTransition(activation, call)
        for activation in dt.dig.activations.values()
        for call in activation.calls
        if call.label is ToolCall.Label.ENVIRONMENT
    ]
    out.sort(key=lambda transition: transition.call.at or 0.0)
    return out


def environment_calls(dt: DIGParallelInterface) -> List[ToolCall]:
    """The environment calls themselves, in realization order."""
    return [transition.call for transition in transitions(dt)]


def feedback(dt: DIGParallelInterface, call: ToolCall) -> List[DIGEvent]:
    """E_Z of one environment call."""
    if call.label is not ToolCall.Label.ENVIRONMENT:
        raise ValueError(f"{call.tool!r} is not an environment tool")
    return list(call.returned)
