"""Dependency analysis of the DIG view: what a call could have depended on.

DIG incidence does not assert that every return depends on every input, so
ancestry is the OVER-approximation the record supports: an event depends on
whatever was available to the call that returned it, Avail(h, r), and
transitively on what those events depend on."""

from __future__ import annotations

from typing import List, Optional

from ..interface import DIGParallelInterface
from ..activation import DIGActivation
from ..event import DIGEvent


def available(dt: DIGParallelInterface, activation: DIGActivation, before: Optional[int] = None) -> List[DIGEvent]:
    """Avail(h, r): E_h plus the returns of calls before index r."""
    return activation.available(before)


def ancestry(dt: DIGParallelInterface, event: DIGEvent) -> List[DIGEvent]:
    """Every event the given one may depend on, in discovery order: the
    availability set of its producing call, then theirs. A root has none."""
    out: List[DIGEvent] = []
    seen = {event.id}
    queue = [event]
    while queue:
        current = queue.pop(0)
        if current.producer is None:
            continue
        activation = dt.dig.activations[current.producer.activation_id]
        for candidate in activation.available(current.producer.call_index):
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            out.append(candidate)
            queue.append(candidate)
    return out


def dependents(dt: DIGParallelInterface, event: DIGEvent) -> List[DIGEvent]:
    """Every event that may depend on the given one: returns of calls that
    had it available, transitively, in discovery order."""
    out: List[DIGEvent] = []
    seen = {event.id}
    queue = [event]
    while queue:
        current = queue.pop(0)
        for activation in dt.dig.activations.values():
            for call in activation.calls:
                if not any(candidate is current for candidate in activation.available(call.index)):
                    continue
                for returned in call.returned:
                    if returned.id in seen:
                        continue
                    seen.add(returned.id)
                    out.append(returned)
                    queue.append(returned)
    return out
