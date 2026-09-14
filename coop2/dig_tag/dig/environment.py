"""The environment seam Z."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import FrozenSet, Iterable, Sequence

from ..base import ToolCall
from .activation import DIGActivation
from .event import DIGEvent


class Environment(ABC):
    """The shared world Z on which the environment tools U_Z act.

    DIG-TAG never models Z itself. An environment declares its tool names
    and realizes one call at a time: `step` applies the call to its own
    state (Z -> Z', deterministic or stochastic, entirely the environment's
    business) and returns the feedback E_Z it externalizes -- fresh
    `DIGEvent`s, possibly none. The interface inserts that feedback into the
    DIG with this call as producer and, unless an event declares its own
    origin, `ENVIRONMENT_ORIGIN`. Autonomous evolution of Z reaches the DIG
    only when some later call externalizes it.

    `step` runs outside the interface's lock, so agents on threads may
    step the world at once: a world that cannot take that serializes
    inside its own `step`."""

    @property
    @abstractmethod
    def tools(self) -> FrozenSet[str]:
        """The names of U_Z. None may be a reserved task or information
        tool name."""

    @abstractmethod
    def step(self, activation: DIGActivation, call: ToolCall) -> Sequence[DIGEvent]:
        """Realize one environment call issued in `activation`; return the
        feedback events. Raise to reject the call: nothing is recorded."""


class NullEnvironment(Environment):
    """An environment with no tools: the run has no shared world beyond the
    tools of the interface. The default when one is built without an environment."""

    tools: FrozenSet[str] = frozenset()

    def step(self, activation: DIGActivation, call: ToolCall) -> Sequence[DIGEvent]:
        raise ValueError(f"the null environment has no tool {call.tool!r}")


class RecordedEnvironment(Environment):
    """The environment of a loaded record: it keeps the tool names the run
    declared, so the record round-trips, and realizes nothing -- a loaded
    run is finished."""

    def __init__(self, tools: Iterable[str]) -> None:
        self._tools = frozenset(tools)

    @property
    def tools(self) -> FrozenSet[str]:
        return self._tools

    def step(self, activation: DIGActivation, call: ToolCall) -> Sequence[DIGEvent]:
        raise ValueError("a loaded record is a finished run; its environment realizes no calls")
