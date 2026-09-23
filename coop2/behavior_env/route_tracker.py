"""Which sub-goal of a route has been met, judged on the world's state.

COOHAVIOR records a checkpoint when its placement tool succeeds on the right
marker. This tracker asks the world instead, at every macro-step boundary: is
the cargo `ontop` the next node's support? Three reasons: our `place_on_top`
teleports and welds,
so the predicate *is* the ground truth once the object is kept awake; a state
check covers every path that puts cargo on a support -- an arm's placement, a
drone's release, an unload -- without instrumenting each verb; and it is the
same instrument ``check_goal`` uses, so the two cannot disagree about whether a
box is on a shelf.

Order is what makes it a route. A node counts only when it is the next one
expected; a box put on C4 while C2 is next is recorded ``out_of_order`` and
nothing advances -- not punished, because an exploratory mistake and an
inability are different things and what is measured here is the difference
between cooperation topologies (user, 2026-09-12). Progress resumes the instant
C2 holds. Nodes already scored are never re-evaluated: under state detection a
box still sitting where it scored is not a duplicate visit, it is just still
there.

Pure Python over the world model's three questions -- ``relation_holds``,
``held_objects``, ``entity_id_of_name`` -- so the CPU test drives it with a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from coop2.behavior_env.route_spec import Route, RouteNode, RouteSpec

__all__ = ["RouteEvent", "RouteProgress", "RouteTracker"]


@dataclass(frozen=True)
class RouteEvent:
    """One node judged at one macro-step."""

    env_step: int
    route_id: str
    node_id: str
    #: ``completed`` -- the expected node holds; ``out_of_order`` -- a later
    #: node holds while the expected one does not.
    status: str
    goal: Tuple[str, str, str]
    #: The node that *was* expected, for an out_of_order event; equals node_id
    #: for a completion.
    expected: str
    #: Who put it there, when that can be said honestly: the one agent whose
    #: primitive terminated this macro-step and who was holding the cargo at
    #: the previous one. Otherwise None -- "unattributed" is a real value and
    #: a guess is not.
    by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "route": self.route_id,
            "node": self.node_id,
            "status": self.status,
            "goal": list(self.goal),
            "expected": self.expected,
            "by": self.by,
        }


@dataclass
class RouteProgress:
    """Where one route stands. What the prompt and the metrics read."""

    route_id: str
    cargo: str
    completed: List[str] = field(default_factory=list)
    next: Optional[str] = None
    remaining: List[str] = field(default_factory=list)
    out_of_order: int = 0

    @property
    def done(self) -> bool:
        return self.next is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route_id,
            "cargo": self.cargo,
            "completed": list(self.completed),
            "next": self.next,
            "remaining": list(self.remaining),
            "out_of_order": self.out_of_order,
            "done": self.done,
        }


class RouteTracker:
    """Advance each route by evaluating its next node against the world."""

    def __init__(self, world: Any, spec: RouteSpec):
        self.world = world
        self.spec = spec
        #: route id -> index of the node expected next; == len(nodes) when done.
        self._next: Dict[str, int] = {route.id: 0 for route in spec.routes}
        self._events: List[RouteEvent] = []
        self._out_of_order: Dict[str, int] = {route.id: 0 for route in spec.routes}
        #: ``{cargo entity id: robot name}`` at the previous macro-step, for
        #: attributing a completion to the robot that let go of the cargo.
        self._held_before: Dict[str, str] = {}
        self._steps = 0

    # -- reading --------------------------------------------------------------

    def next_node(self, route_id: str) -> Optional[RouteNode]:
        route = self._route(route_id)
        index = self._next[route_id]
        return route.nodes[index] if index < len(route.nodes) else None

    def complete(self) -> bool:
        """Every route's last node has been credited, in order."""
        return all(self._next[route.id] >= len(route.nodes) for route in self.spec.routes)

    def completed_count(self) -> int:
        return sum(self._next[route.id] for route in self.spec.routes)

    def progress(self) -> Dict[str, RouteProgress]:
        out: Dict[str, RouteProgress] = {}
        for route in self.spec.routes:
            index = self._next[route.id]
            out[route.id] = RouteProgress(
                route_id=route.id,
                cargo=route.cargo,
                completed=[n.id for n in route.nodes[:index]],
                next=route.nodes[index].id if index < len(route.nodes) else None,
                remaining=[n.id for n in route.nodes[index + 1:]],
                out_of_order=self._out_of_order[route.id],
            )
        return out

    def history(self) -> List[RouteEvent]:
        return list(self._events)

    def summary(self) -> Dict[str, Any]:
        """COOHAVIOR's numbers under COOHAVIOR's names, so a table can hold
        both projects' runs. ``Y_task`` counts nodes (user, 2026-09-12)."""
        required = self.spec.required_nodes
        completed = self.completed_count()
        return {
            "required_nodes": required,
            "completed_nodes": completed,
            "Y_task": (completed / required) if required else 0.0,
            "out_of_order_visits": sum(self._out_of_order.values()),
            "task_complete": self.complete(),
            "routes": {rid: p.to_dict() for rid, p in self.progress().items()},
        }

    # -- advancing ------------------------------------------------------------

    def step(self, env_step: int, acting: Optional[Dict[str, Any]] = None) -> List[RouteEvent]:
        """Judge every route once. Call at each macro-step boundary, after
        ``world.step()``, with ``acting`` = the agents whose primitive terminated
        (the same shape ``CoopTaskTracker.step`` takes).

        Evaluates at most ``routes x remaining nodes`` predicates -- ten for any
        V4 task -- and never a node already scored.
        """
        self._steps += 1
        held_now = self._held_by_entity_id()
        terminated = set(acting or {})
        events: List[RouteEvent] = []

        for route in self.spec.routes:
            index = self._next[route.id]
            if index >= len(route.nodes):
                continue

            # The expected node, then anything it unlocked: one placement can
            # only satisfy one support, but the loop costs nothing and is what
            # handles a node whose support is also the next node's.
            while index < len(route.nodes) and self._holds(route.nodes[index]):
                node = route.nodes[index]
                events.append(RouteEvent(
                    env_step=env_step, route_id=route.id, node_id=node.id,
                    status="completed", goal=node.goal, expected=node.id,
                    by=self._credit(route.cargo, held_now, terminated),
                ))
                index += 1
                self._next[route.id] = index
            if index >= len(route.nodes):
                continue

            # Later nodes, once each: a box on C4 while C2 is next is recorded
            # and credited to nobody. Nothing advances; it will the moment C2
            # holds. Earlier nodes are not asked -- a box still where it scored
            # is not a duplicate visit under state detection.
            expected = route.nodes[index]
            for later in route.nodes[index + 1:]:
                if self._holds(later):
                    self._out_of_order[route.id] += 1
                    events.append(RouteEvent(
                        env_step=env_step, route_id=route.id, node_id=later.id,
                        status="out_of_order", goal=later.goal, expected=expected.id,
                    ))

        self._events.extend(events)
        self._held_before = held_now
        return events

    # -- internals ------------------------------------------------------------

    def _route(self, route_id: str) -> Route:
        for route in self.spec.routes:
            if route.id == route_id:
                return route
        raise KeyError(route_id)

    def _holds(self, node: RouteNode) -> bool:
        try:
            return bool(self.world.relation_holds(*node.goal))
        except Exception:  # noqa: BLE001 - an undefined pair is "not there"
            return False

    def _held_by_entity_id(self) -> Dict[str, str]:
        """``{cargo entity id: robot name}`` right now, or {} if the world
        cannot say (a stub without inventories)."""
        held = getattr(self.world, "held_objects", None)
        to_id = getattr(self.world, "entity_id_of_name", None)
        if not callable(held) or not callable(to_id):
            return {}
        out: Dict[str, str] = {}
        try:
            for name, robot in (held() or {}).items():
                entity_id = to_id(name)
                if entity_id:
                    out[entity_id] = robot
        except Exception:  # noqa: BLE001
            return {}
        return out

    def _credit(self, cargo: str, held_now: Dict[str, str], terminated: Iterable[str]) -> Optional[str]:
        """The robot that let go of the cargo this macro-step, if exactly one
        candidate exists: it held the cargo last step, does not now, and its
        primitive terminated now. Anything less certain is None."""
        holder = self._held_before.get(cargo)
        if holder is None or held_now.get(cargo) == holder:
            return None
        return holder if holder in set(terminated) else None
