"""The asynchronously activated agent."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, List, Optional

from .activation import DIGActivation
from .interface import DIGParallelInterface

Behavior = Callable[[DIGActivation, DIGParallelInterface], Any]


class AsyncAgent:
    """One agent i in I, activated asynchronously.

    The agent is woken through the interface's `on_arrival` hook whenever
    something is made available to it, and never scans the record. Its
    `run` loop fires one activation per batch of pending events: it opens
    the activation (E_h fixed to its inbox), runs the behavior, closes, and
    repeats while anything is still pending, then sleeps until the next
    wake. A behavior is `behavior(activation, dt)` and may be sync or
    async; awaiting inside the open activation (an LLM call, say) is how
    this agent's activation overlaps with others'.

    The interface stays synchronous and imports no asyncio; this is the
    only module that does. `run` is one firing policy -- fire as soon as
    anything is pending; a runtime with its own schedule may call
    `dt.open_activation` directly and never use this class."""

    def __init__(self, agent_id: str, dt: DIGParallelInterface, behavior: Behavior) -> None:
        if not callable(behavior):
            raise TypeError("behavior must be callable: behavior(activation, dt)")
        self.agent_id = agent_id
        self.dt = dt
        self._behavior = behavior
        self._wakeup = asyncio.Event()
        dt.add_agent(agent_id)
        dt.on_arrival(agent_id, self.wake)

    def wake(self) -> None:
        """Nudge the loop (a send does this automatically)."""
        self._wakeup.set()

    @property
    def active(self) -> bool:
        return self.dt.dig.open_activation_of(self.agent_id) is not None

    @property
    def activations(self) -> List[DIGActivation]:
        """This agent's activations, in H order."""
        return self.dt.dig.activations_of(self.agent_id)

    def should_fire(self) -> bool:
        """Not active, and something is pending."""
        return not self.active and bool(self.dt.pending(self.agent_id))

    async def fire(self) -> DIGActivation:
        """One activation: open, run the behavior, close. An exception in
        the behavior closes the activation with the error noted and
        propagates."""
        with self.dt.open_activation(self.agent_id) as activation:
            result = self._behavior(activation, self.dt)
            if inspect.isawaitable(result):
                await result
        return activation

    async def _wait(self, stop: Optional[asyncio.Event]) -> None:
        if stop is None:
            await self._wakeup.wait()
            self._wakeup.clear()
            return
        wakeup = asyncio.ensure_future(self._wakeup.wait())
        stopped = asyncio.ensure_future(stop.wait())
        try:
            await asyncio.wait({wakeup, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (wakeup, stopped):
                if not task.done():
                    task.cancel()
        self._wakeup.clear()

    async def run(self, stop: Optional[asyncio.Event] = None) -> None:
        """Fire whenever anything is pending, until `stop` is set."""
        while True:
            while self.should_fire() and (stop is None or not stop.is_set()):
                await self.fire()
            if stop is not None and stop.is_set():
                return
            await self._wait(stop)
