"""DIG: the interface over D and Z alone -- agents, their activations and
sends, calls on the environment, and the record of all of it."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from ..base import ToolCall, expect_type
from .activation import DIGActivation
from .environment import Environment, NullEnvironment, RecordedEnvironment
from .event import ENVIRONMENT_ORIGIN, EXTERNAL_ORIGIN, DIGEvent
from .graph import DIGGraph
from .tools import SEND_TOOL

SCHEMA = "dig.v2"

EventRef = Union[DIGEvent, str]


class DIGParallelInterface:
    """(D, Z, U) over the shared agent identities I: agents fire
    activations, issue sends and environment calls, and everything is
    recorded in D.

    There is ONE write path: `issue(activation, call)` realizes any call,
    and the tool methods (`send`, `act`) only build a `ToolCall` and hand
    it to `issue`. `inject` is the root funnel for events with no producing
    call. `open_activation` is the firing lifecycle: the activation enters
    H on entry, with E_h fixed to the agent's context, and is closed on
    exit. A call is validated in full before anything changes: a rejected
    call leaves D and Z untouched (the environment is asked only after the
    DIG-side checks pass). The one reference condition: an event a send
    names must be in Avail(h, r).

    An interface with more stores than D extends `reserved_tools`,
    `_realize`; DIG-TAG does, for the TAG's tools.

    Parallel: agents act through it at once, on asyncio or on threads.
    Every write -- inject, route, open, end, and each issued call -- is one
    critical section under a lock, released before listeners are told, so
    two agents' steps never interleave inside the record; `hold()` gives a
    reader the same stillness."""

    def __init__(self, agents: Iterable[str] = (), environment: Optional[Environment] = None) -> None:
        self._agents: Dict[str, None] = {}
        self.environment: Environment = environment if environment is not None else NullEnvironment()
        expect_type(self.environment, Environment, what="DIG environment")
        clash = set(self.environment.tools) & set(self.reserved_tools)
        if clash:
            raise ValueError(f"environment tools {sorted(clash)} collide with the interface's own tool names")
        self.dig = DIGGraph()
        self._lock = threading.RLock()          # every write is one critical section: callers may be threads
        self._listeners: List[Callable[[], None]] = []
        self._arrival_listeners: Dict[str, List[Callable[[], None]]] = {}
        for agent_id in agents:
            self.add_agent(agent_id)

    @property
    def reserved_tools(self) -> FrozenSet[str]:
        """The tool names the interface realizes itself; an environment may
        not declare them."""
        return frozenset({SEND_TOOL})

    # -- agents and observers -------------------------------------------

    @property
    def agents(self) -> List[str]:
        """I, in registration order."""
        return list(self._agents)

    def add_agent(self, agent_id: str) -> None:
        """Admit an identity to I (idempotent)."""
        if not isinstance(agent_id, str) or not agent_id:
            raise TypeError(f"agent id must be a non-empty str; got {agent_id!r}")
        self._agents.setdefault(agent_id, None)

    def expect_agent(self, agent_id: Any) -> str:
        if agent_id not in self._agents:
            raise ValueError(f"unknown agent id {agent_id!r}; agents are {self.agents}")
        return agent_id

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Call `listener()` after every change (inject, open, issue, end).
        The interface's only outward hook; agents and observers attach
        here, and the interface never imports them."""
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()

    def on_arrival(self, agent_id: str, listener: Callable[[], None]) -> None:
        """Call `listener()` whenever an event is made available to the
        agent (a send or an injection naming it): the wake signal an agent
        runtime needs, without scanning every agent on every change."""
        self._arrival_listeners.setdefault(self.expect_agent(agent_id), []).append(listener)

    def _notify_arrival(self, agents: Sequence[str]) -> None:
        for agent_id in agents:
            for listener in self._arrival_listeners.get(agent_id, ()):
                listener()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the record still while the block runs: no write lands until
        it ends, so a reader on another thread (a view over a run in
        progress, a live figure) sees one consistent record."""
        with self._lock:
            yield

    def pending(self, agent_id: str) -> List[DIGEvent]:
        """Events available to the agent that no activation of it has been
        presented yet -- the wake condition."""
        with self._lock:
            return self.dig.pending(self.expect_agent(agent_id))

    def inbox(self, agent_id: str) -> List[DIGEvent]:
        """Everything ever made available to the agent, in arrival order."""
        with self._lock:
            return self.dig.inbox_events(self.expect_agent(agent_id))

    def received(self, activation: DIGActivation) -> List[DIGEvent]:
        """The activation's inputs that arrived through the agent's inbox --
        sent to it or injected for it -- as opposed to its own earlier
        returns, which E_h also carries. What a behavior usually reacts to."""
        with self._lock:
            arrivals = self.dig.inbox.get(activation.agent_id, {})
            return [event for event in activation.inputs if event.id in arrivals]

    # -- the root funnel and the firing lifecycle ------------------------

    def inject(self, event: DIGEvent, *, to: Sequence[str], origin: str = EXTERNAL_ORIGIN) -> DIGEvent:
        """Admit an externally supplied event with no producing call (a root)
        and make it available to the agents in `to`."""
        with self._lock:
            for agent_id in to:
                self.expect_agent(agent_id)
            self.dig.record_root(event, to=to, origin=origin)
        self._notify()
        self._notify_arrival(to)
        return event


    def route(self, event: EventRef, *, to: Sequence[str], by: str = "runtime") -> List[str]:
        """Runtime-defined routing: make a recorded event available to the
        agents in `to` on behalf of the executor `by`, without a send call
        -- what a framework's shared state presents to an invocation beyond
        the routes its agents chose. The interaction graph does not see it;
        the receiving activation's inputs and the flow graph do. Returns
        the agents for whom it was new."""
        with self._lock:
            for agent_id in to:
                self.expect_agent(agent_id)
            event_id = self._event_id(event, "event")
            if event_id not in self.dig.events:
                raise ValueError(f"route requires a recorded event; {event_id!r} is not in the DIG")
            added = self.dig.route(self.dig.events[event_id], to, by)
        self._notify()
        self._notify_arrival(added)
        return added

    def open(self, agent_id: str, inputs: Optional[Sequence[EventRef]] = None, by: str = "runtime") -> DIGActivation:
        """The explicit form of the firing lifecycle: open an activation of
        the agent now and close it later with `end`. E_h is the agent's
        whole context, or exactly `inputs` when the runtime states what it
        presented (events it had not made available are routed on its
        behalf, see `route`). A runtime that interleaves several agents'
        activations (a superstep engine, a framework adapter) uses this
        pair; `open_activation` wraps it for the common block-shaped case."""
        with self._lock:
            self.expect_agent(agent_id)
            events = None
            if inputs is not None:
                events = []
                for ref in inputs:
                    event_id = self._event_id(ref, "input")
                    if event_id not in self.dig.events:
                        raise ValueError(f"open requires recorded inputs; {event_id!r} is not in the DIG")
                    events.append(self.dig.events[event_id])
            activation = self.dig.open(agent_id, events, by)
        self._notify()
        return activation

    def end(self, activation: DIGActivation, error: Optional[BaseException] = None) -> DIGActivation:
        """Close an open activation: stamp t', note an error if there was
        one. Everything it issued stays recorded."""
        with self._lock:
            self.dig.end(activation, error=error)
        self._notify()
        return activation

    @contextmanager
    def open_activation(self, agent_id: str) -> Iterator[DIGActivation]:
        """One activation of the agent, as a lifecycle block. Entering opens
        it (`open`) and yields the recorded node itself; the body issues
        calls through this interface; leaving closes it (`end`). An
        exception inside the block closes the activation with the error
        noted and re-raises -- everything it issued stays recorded (the
        record is prefix-realized)."""
        activation = self.open(agent_id)
        try:
            yield activation
        except BaseException as error:
            self.end(activation, error=error)
            raise
        self.end(activation)

    # -- the one write path ---------------------------------------------

    def issue(self, activation: DIGActivation, call: ToolCall) -> List[DIGEvent]:
        """Realize one call issued in `activation`: the one write path.

        send:        (D (+)_h (u, {}), then e enters the inboxes of J; Z unchanged)
        env call:    (D (+)_h (u, E_Z), Z -> Z')

        Any other tool is handed to `_realize`, which an interface with
        more stores extends. Returns the call's returned events E_u.

        The record's checks and writes are critical sections under the
        lock; the world's step is not: it runs between them, so a tool
        that takes long holds no other agent, and agents on threads step
        the world at once (its thread safety is the environment's own).
        The call is recorded once the world has answered, so the record
        stays prefix-realized."""
        arrived: Sequence[str] = ()
        with self._lock:
            self.dig.expect_open(activation)
            expect_type(call, ToolCall, what="DIGParallelInterface.issue")
            if call.recorded:
                raise ValueError("a ToolCall is one occurrence; build a new one per call")
            environment_call = call.tool != SEND_TOOL and call.label is ToolCall.Label.ENVIRONMENT
            if environment_call and call.tool not in self.environment.tools:
                raise ValueError(
                    f"unknown tool {call.tool!r}: the environment declares {sorted(self.environment.tools)}"
                )
        if environment_call:
            feedback = self._step_environment(activation, call)      # outside the lock: the world's own business
            with self._lock:
                self.dig.expect_open(activation)
                self.dig.append(activation, call, returned=feedback, origin=ENVIRONMENT_ORIGIN)
        else:
            with self._lock:
                if call.tool == SEND_TOOL:
                    arrived = self._issue_send(activation, call)
                else:
                    self._realize(activation, call)
        self._notify()
        self._notify_arrival(arrived)
        return list(call.returned)

    def _realize(self, activation: DIGActivation, call: ToolCall) -> None:
        """A call to a tool this interface does not own."""
        raise ValueError(
            f"unknown tool {call.tool!r}: this interface realizes send and the environment's "
            f"{sorted(self.environment.tools)}"
        )

    def _issue_send(self, activation: DIGActivation, call: ToolCall) -> List[str]:
        """Realize a send; returns the agents for whom the event was new."""
        event_id = call.args.get("event")
        recipients = call.args.get("to")
        if not isinstance(event_id, str) or event_id not in self.dig.events:
            raise ValueError("send requires `event`: the id of a recorded event")
        if not isinstance(recipients, (list, tuple)) or not recipients:
            raise ValueError("send requires `to`: a non-empty list of agent ids")
        for agent_id in recipients:
            self.expect_agent(agent_id)          # J subset of I
        self.dig.expect_available(activation, [event_id])
        self.dig.append(activation, call, returned=[], origin=activation.agent_id)
        return self.dig.make_available(self.dig.events[event_id], recipients)

    def _step_environment(self, activation: DIGActivation, call: ToolCall) -> List[DIGEvent]:
        """Z -> Z': apply the call to the world and take its feedback E_Z."""
        feedback = list(self.environment.step(activation, call))
        for event in feedback:
            expect_type(event, DIGEvent, what="environment feedback")
        return feedback

    # -- the tool methods: sugar over `issue` ---------------------------

    @staticmethod
    def _event_id(ref: EventRef, what: str) -> str:
        if isinstance(ref, DIGEvent):
            if ref.id is None:
                raise ValueError(f"{what} must be a recorded event")
            return ref.id
        if isinstance(ref, str) and ref:
            return ref
        raise TypeError(f"{what} must be a recorded event or its id; got {ref!r}")

    def send(self, activation: DIGActivation, event: EventRef, *, to: Sequence[str]) -> None:
        """send_h(e, J): make an available event available to the agents in J."""
        self.issue(activation, ToolCall(SEND_TOOL, {"event": self._event_id(event, "event"), "to": list(to)}))

    def act(self, activation: DIGActivation, tool: str, **args: Any) -> List[DIGEvent]:
        """An environment call u in U_Z; returns the feedback E_Z."""
        return self.issue(activation, ToolCall(tool, dict(args)))

    # -- the record ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the run: agents, the environment's tool names, and D.
        Derived reads are query answers, recomputed by the views, not
        stored."""
        return {
            "schema": self.schema(),
            "agents": self.agents,
            "environment_tools": sorted(self.environment.tools),
            "dig": self.dig.to_dict(),
        }

    @classmethod
    def schema(cls) -> str:
        return SCHEMA

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DIGParallelInterface":
        """Rebuild a recorded run. The loaded interface is the record of a
        finished run, not a live bus: its environment keeps the recorded
        tool names and realizes nothing."""
        schema = data.get("schema") if isinstance(data, Mapping) else None
        if schema != cls.schema():
            raise ValueError(f"unsupported schema {schema!r}; expected {cls.schema()!r}")
        dt = cls(agents=data.get("agents") or [], environment=RecordedEnvironment(data.get("environment_tools") or []))
        dt.dig = DIGGraph.from_dict(data.get("dig") or {})
        return dt

    def save_json(self, path: "str | Path") -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: "str | Path") -> "DIGParallelInterface":
        return cls.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
