"""DIGGraph: the representation D = (E, H, L_D)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..base import Minter, ToolCall, expect_type, natural_sort_key
from .activation import DIGActivation
from .event import EXTERNAL_ORIGIN, DIGEvent


class DIGGraph:
    """The DIG REPRESENTATION: the event store E, the activation store H, and
    the per-agent inboxes that realize send. L_D is not stored -- it derives
    from `inputs` (e -> h) and `calls[*].returned` (h -> e); every other
    derived read lives in `dig_tag.dig.views`. The graph answers nothing.

    Writes come through four funnels, all called by the interface:
    `record_root` (an event with no producer), `open` (an activation enters
    H with E_h fixed), `append` (one realized call-return pair, its returned
    events inserted with this pair as their unique producer), and
    `make_available` (send's effect: the existing event enters the recipient
    inboxes without copying), and `route` (runtime-defined routing: the
    same effect on behalf of an identified executor, without a send
    call; `routed` remembers who). `end` stamps t'.

    E_h of a new activation is the agent's whole CONTEXT at that instant:
    every event ever made available to it by a send or a root (its inbox),
    plus every event its own earlier activations returned, ordered by when
    each became available. An agent's context accumulates, as an LLM
    agent's history does, so a task node it received or returned once stays
    available to its later calls. Only the inbox is wake-worthy: `pending`
    is what the inbox holds beyond what earlier activations were presented."""

    events: Dict[str, DIGEvent]
    activations: Dict[str, DIGActivation]
    inbox: Dict[str, Dict[str, float]]

    def __init__(self) -> None:
        self.events = {}
        self.activations = {}
        self.inbox = {}          # agent id -> {event id: available_at}, in arrival order
        self.routed: Dict[str, Dict[str, str]] = {}   # agent id -> {event id: executor}, for runtime routing
        self._by_agent: Dict[str, List[DIGActivation]] = {}   # an index over `activations`
        self._presented: Dict[str, Dict[str, None]] = {}       # agent id -> event ids presented, first-presentation order
        self._events = Minter("e")
        self._activations = Minter("h")
        self._created_at = time.time()

    # -- reads the substrate needs itself -------------------------------

    def activations_of(self, agent_id: str) -> List[DIGActivation]:
        """The agent's activations, in H order."""
        return list(self._by_agent.get(agent_id, []))

    def open_activation_of(self, agent_id: str) -> Optional[DIGActivation]:
        """The agent's open activation, if it has one (at most one)."""
        own = self._by_agent.get(agent_id)
        if own and own[-1].is_open:
            return own[-1]
        return None

    def inbox_events(self, agent_id: str) -> List[DIGEvent]:
        """What sends and roots made available to the agent, in arrival order."""
        return [self.events[event_id] for event_id in self.inbox.get(agent_id, {})]

    def context(self, agent_id: str) -> List[DIGEvent]:
        """Everything available to the agent right now: its inbox plus the
        returns of its own activations, ordered by when each became
        available (arrival for the inbox, creation for own returns)."""
        available: Dict[str, Tuple[float, DIGEvent]] = {}
        for activation in self._by_agent.get(agent_id, []):
            for event in activation.returned:
                available.setdefault(event.id, (event.created_at, event))
        for event_id, at in self.inbox.get(agent_id, {}).items():
            known = available.get(event_id)
            if known is None or at < known[0]:
                available[event_id] = (at, self.events[event_id])
        ordered = sorted(
            available.items(),
            key=lambda item: (item[1][0], natural_sort_key(item[0])),
        )
        return [event for _, (_, event) in ordered]

    def presented(self, agent_id: str) -> List[DIGEvent]:
        """Events already presented to some activation of the agent, in
        first-presentation order."""
        return [self.events[event_id] for event_id in self._presented.get(agent_id, {})]

    def pending(self, agent_id: str) -> List[DIGEvent]:
        """Inbox events not yet presented to any activation of the agent --
        what a woken agent checks. Own returns never count: an agent does
        not wake because it produced something."""
        presented = self._presented.get(agent_id, {})
        return [event for event in self.inbox_events(agent_id) if event.id not in presented]

    def expect_recorded(self, event: DIGEvent) -> DIGEvent:
        expect_type(event, DIGEvent, what="DIGGraph.expect_recorded")
        if event.id is None or self.events.get(event.id) is not event:
            raise ValueError(f"event {event.id or '-'} is not recorded in this DIG")
        return event

    def expect_open(self, activation: DIGActivation) -> DIGActivation:
        expect_type(activation, DIGActivation, what="DIGGraph.expect_open")
        if activation.id is None or self.activations.get(activation.id) is not activation:
            raise ValueError("activation is not recorded in this DIG")
        if not activation.is_open:
            raise ValueError(f"activation {activation.id} is closed")
        return activation

    def expect_available(
        self,
        activation: DIGActivation,
        event_ids: Sequence[str],
    ) -> None:
        """A call may name only objects in Avail(h, r) for the next call."""
        available = set(activation.available_ids())
        missing = [event_id for event_id in event_ids if event_id not in available]
        if missing:
            raise ValueError(
                f"activation {activation.id} ({activation.agent_id}) cannot name "
                f"{', '.join(missing)}: not available to it (Avail(h, r) is its "
                "inputs plus the returns of its earlier calls)"
            )

    # -- the funnels -----------------------------------------------------

    def _insert(self, event: DIGEvent, *, producer: Optional[DIGEvent.Producer], origin: str) -> DIGEvent:
        event.id = self._events.next(self.events)
        event.producer = producer
        if event.origin is None:
            event.origin = origin
        event.created_at = time.time()
        self.events[event.id] = event
        return event

    def record_root(
        self,
        event: DIGEvent,
        *,
        to: Sequence[str] = (),
        origin: str = EXTERNAL_ORIGIN,
    ) -> DIGEvent:
        """Insert an event with no producing call (a root: the run's initial
        input, or anything supplied from outside) and make it available to
        the agents in `to`. The event keeps a declared origin it already
        carries; otherwise it gets `origin`."""
        expect_type(event, DIGEvent, what="DIGGraph.record_root")
        if event.recorded:
            raise ValueError(f"event {event.id} is already recorded")
        self._insert(event, producer=None, origin=origin)
        self.make_available(event, to)
        return event

    def open(
        self,
        agent_id: str,
        inputs: Optional[Sequence[DIGEvent]] = None,
        by: str = "runtime",
    ) -> DIGActivation:
        """Open an activation of the agent: it enters H now, with E_h fixed.
        By default E_h is the agent's whole context at this instant. A
        runtime that knows exactly what it presented passes `inputs`
        (recorded events, in presentation order): they become E_h as given,
        and any of them not yet available to the agent -- other than its own
        earlier returns -- is routed on the runtime's behalf first."""
        if not isinstance(agent_id, str) or not agent_id:
            raise TypeError(f"agent_id must be a non-empty str; got {agent_id!r}")
        already = self.open_activation_of(agent_id)
        if already is not None:
            raise ValueError(
                f"agent {agent_id} already has an open activation ({already.id}); "
                "an agent fires one activation at a time"
            )
        if inputs is None:
            presented_now = self.context(agent_id)
        else:
            own = {event.id for a in self._by_agent.get(agent_id, []) for event in a.returned}
            presented_now = []
            for event in inputs:
                self.expect_recorded(event)
                if any(known is event for known in presented_now):
                    continue
                if event.id not in self.inbox.get(agent_id, {}) and event.id not in own:
                    self.route(event, [agent_id], by)
                presented_now.append(event)
        activation = DIGActivation(
            agent_id=agent_id,
            inputs=presented_now,
            started_at=time.time(),
        )
        activation.id = self._activations.next(self.activations)
        self.activations[activation.id] = activation
        self._by_agent.setdefault(agent_id, []).append(activation)
        presented = self._presented.setdefault(agent_id, {})
        for event in activation.inputs:
            presented.setdefault(event.id, None)
        return activation

    def append(
        self,
        activation: DIGActivation,
        call: ToolCall,
        *,
        returned: Sequence[DIGEvent],
        origin: str,
    ) -> ToolCall:
        """D (+)_h (u, E_u): append a realized call to U_h and insert its
        returned events, each fresh, with this call-return pair as its unique
        producer. `origin` is the declared provenance for returned events
        that do not declare their own."""
        self.expect_open(activation)
        expect_type(call, ToolCall, what="DIGGraph.append")
        if call.recorded:
            raise ValueError(f"call {call.tool} is already recorded (index {call.index})")
        for event in returned:
            expect_type(event, DIGEvent, what="DIGGraph.append returned")
            if event.recorded:
                raise ValueError(
                    f"returned event {event.id} is already recorded: every returned "
                    "event is fresh, with this call as its unique producer"
                )
        index = len(activation.calls)
        producer = DIGEvent.Producer(activation.id, index)
        for event in returned:
            self._insert(event, producer=producer, origin=origin)
        call.index = index
        call.returned = list(returned)
        call.at = time.time()
        activation.calls.append(call)
        return call

    def make_available(self, event: DIGEvent, to: Sequence[str]) -> List[str]:
        """Send's effect: the existing event enters each recipient's inbox
        without copying, stamped with its arrival. Returns the recipients
        for whom this was new (an event already in an inbox stays where it
        is)."""
        self.expect_recorded(event)
        added: List[str] = []
        now = time.time()
        for agent_id in to:
            if not isinstance(agent_id, str) or not agent_id:
                raise TypeError(f"recipient must be a non-empty agent id; got {agent_id!r}")
            inbox = self.inbox.setdefault(agent_id, {})
            if event.id in inbox:
                continue
            inbox[event.id] = now
            added.append(agent_id)
        return added

    def route(self, event: DIGEvent, to: Sequence[str], by: str) -> List[str]:
        """Runtime-defined routing: the event enters the inboxes in `to` as
        `make_available` does, and the executor `by` is remembered for the
        recipients for whom it was new. No call records it; the receiving
        activation's inputs and the flow graph show it, the interaction
        graph (sends) does not."""
        if not isinstance(by, str) or not by:
            raise TypeError(f"the routing executor must be a non-empty str; got {by!r}")
        added = self.make_available(event, to)
        for agent_id in added:
            self.routed.setdefault(agent_id, {})[event.id] = by
        return added

    def end(self, activation: DIGActivation, error: Optional[BaseException] = None) -> DIGActivation:
        """Stamp t'. What the activation issued stays; an error is noted."""
        self.expect_open(activation)
        if error is not None:
            activation.mark_error(error)
        activation.ended_at = time.time()
        return activation

    # -- the record ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self._created_at,
            "events": {event_id: event.to_dict() for event_id, event in self.events.items()},
            "activations": {
                activation_id: activation.to_dict()
                for activation_id, activation in self.activations.items()
            },
            "inbox": {
                agent_id: dict(arrivals)
                for agent_id, arrivals in self.inbox.items()
            },
            "routed": {agent_id: dict(by) for agent_id, by in self.routed.items()},
            "counters": {"e": self._events.counter, "h": self._activations.counter},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DIGGraph":
        """Rebuild a recorded DIG: nodes as-is, references re-linked by id.
        No stamps are re-applied; the loaded graph is the record of a run."""
        graph = cls()
        graph._created_at = float(data.get("created_at", graph._created_at))
        counters = data.get("counters") or {}
        graph._events = Minter("e", int(counters.get("e") or 0))
        graph._activations = Minter("h", int(counters.get("h") or 0))
        graph.events = {event_id: DIGEvent.from_dict(payload) for event_id, payload in (data.get("events") or {}).items()}
        graph.activations = {
            activation_id: DIGActivation.from_dict(payload, graph.events)
            for activation_id, payload in (data.get("activations") or {}).items()
        }
        graph.routed = {
            agent_id: dict(by) for agent_id, by in (data.get("routed") or {}).items()
        }
        for activation in graph.activations.values():
            graph._by_agent.setdefault(activation.agent_id, []).append(activation)
            presented = graph._presented.setdefault(activation.agent_id, {})
            for event in activation.inputs:
                presented.setdefault(event.id, None)
        for agent_id, arrivals in (data.get("inbox") or {}).items():
            for event_id in arrivals:
                if event_id not in graph.events:
                    raise ValueError(f"inbox of {agent_id!r} references unknown event {event_id!r}")
            graph.inbox[agent_id] = {event_id: float(at) for event_id, at in arrivals.items()}
        return graph
