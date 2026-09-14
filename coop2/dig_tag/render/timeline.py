"""Where on the time axis each moment of a record goes."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..dig import DIGParallelInterface
from ..dig.views import roots
from ..tag import Evidence, TAGGraph, TaskAction

Key = Tuple[str, object]
Moment = Tuple[Optional[float], int, Key, float]     # (time, rank among equals, key, slots it advances)


class Timeline:
    """x, in slots, of every moment a record draws, in the order the moments
    happened; the T band asks it where an action and an attachment sit
    (a version hangs off its action), the D band where an activation
    opens, calls, and ends.
    By default each moment advances the axis by its width (a call one slot,
    an open half, an end three quarters), the paper's logical time; with
    `clock` (seconds per slot) moments sit at their wall-clock times
    instead, measured from `origin` (the first moment when not given) and
    reaching to `horizon` when given, so a live axis keeps its length
    while moments are added to it. Time is strict: moments in the same
    instant share an x, and an activation is as long as it lasted. A slot
    is left before the first moment for the root events when there are
    any. A moment without a time (an activation still open) sits at now,
    so it reaches further at every redraw."""

    def __init__(self, moments: Sequence[Moment], *, root_slot: bool, clock: Optional[float] = None,
                 origin: Optional[float] = None, horizon: Optional[float] = None) -> None:
        self.x: Dict[Key, float] = {}
        self.root_slot = root_slot
        self.base = 0.5 if root_slot else 0.0
        self.clock, self.origin, self.horizon = clock, origin, horizon
        ordered = sorted(moments, key=lambda m: (m[0] if m[0] is not None else float("inf"), m[1]))
        stamped = [m[0] for m in ordered if m[0] is not None]
        self.first: Optional[float] = min(stamped) if stamped else None    # the moments' wall-clock reach,
        self.last: Optional[float] = max(stamped) if stamped else None     # whatever the axis
        if clock is None:
            x = self.base
            for _, _, key, advance in ordered:
                x += advance
                self.x[key] = x
        else:
            now = time.time()
            times = [m[0] for m in ordered if m[0] is not None]
            self.origin = origin if origin is not None else (min(times) if times else now)
            for t, _, key, _ in ordered:
                self.x[key] = self.at(t if t is not None else now)

    def at(self, t: float) -> float:
        """The x, in slots, of a wall-clock time (clock timelines only)."""
        return self.base + 0.5 + (t - self.origin) / self.clock

    @property
    def extent(self) -> float:
        """The end of the axis, in slots: the horizon when there is one,
        else the last moment."""
        if self.clock is not None and self.horizon is not None:
            return self.at(self.horizon)
        return max(self.x.values(), default=0.0)

    @property
    def span(self) -> Optional[float]:
        """The seconds the axis covers, on a clock timeline."""
        if self.clock is None:
            return None
        end = self.horizon if self.horizon is not None else self.origin + (self.extent - self.base - 0.5) * self.clock
        return end - self.origin

    # what the T band asks -------------------------------------------------

    def action(self, action: TaskAction) -> float:
        raise NotImplementedError

    def attachments(self) -> List[Tuple[float, Evidence, Optional[str]]]:
        """Each attachment as (x, evidence, the attaching agent or None)."""
        raise NotImplementedError


class ActivationTimeline(Timeline):
    """A DIG's record: each activation's open, calls, and end, by their
    times. In DIG-TAG an action sits at its call and an attachment at its
    attach call."""

    def __init__(self, dig: DIGParallelInterface, tag: Optional[TAGGraph] = None, **options: Any) -> None:
        self.dig, self.tag = dig, tag
        moments: List[Moment] = []
        for a in dig.dig.activations.values():
            moments.append((a.started_at, 0, ("open", a.id), 0.5))
            for c in a.calls:
                moments.append((c.at, 1, ("call", (a.id, c.index)), 1.0))
            moments.append((a.ended_at, 2, ("end", a.id), 0.75))
        super().__init__(moments, root_slot=bool(roots(dig)), **options)

    def action(self, action: TaskAction) -> float:
        return self.x[("call", (action.issuer, action.call.index))]

    def attachments(self) -> List[Tuple[float, Evidence, Optional[str]]]:
        out: List[Tuple[float, Evidence, Optional[str]]] = []
        if self.tag is None:
            return out
        for activation in self.dig.dig.activations.values():
            for call in activation.calls:
                if call.tool != "attach":
                    continue
                phi = next(e for e in self.tag.evidence.values() if e.target == call.args["target"]
                           and e.source == (call.args.get("source") or activation.id))
                out.append((self.x[("call", (activation.id, call.index))], phi, activation.agent_id))
        return out


class ActionTimeline(Timeline):
    """A TAG on its own: each action and each attachment, by their times."""

    def __init__(self, tag: TAGGraph, **options: Any) -> None:
        self.tag = tag
        moments: List[Moment] = [(a.call.at, 0, ("action", a.id), 1.0) for a in tag.actions.values()]
        moments += [(phi.created_at, 1, ("attach", phi.id), 1.0) for phi in tag.evidence.values()]
        produced = {q for a in tag.actions.values() for q in a.output_ids}
        super().__init__(moments, root_slot=any(q not in produced for q in tag.tasks), **options)

    def action(self, action: TaskAction) -> float:
        return self.x[("action", action.id)]

    def attachments(self) -> List[Tuple[float, Evidence, Optional[str]]]:
        return [(self.x[("attach", phi.id)], phi, None) for phi in self.tag.evidence.values()]
