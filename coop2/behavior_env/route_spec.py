"""The ordered sub-goals of a COOHAVIOR task, read from a sidecar next to its BDDL.

A COOHAVIOR task is not "put the box on the bed". It is "carry the box through
these nine supports, in this order, then onto the bed", and BDDL can only state
the last of those -- which is why ``v4_s1_v4_hl`` ends at env_step 0: every one
of its goals is an initial condition. This module reads the part BDDL cannot
carry, ``route.json``, found by the activity name beside ``problem0.bddl``. An
activity without one behaves exactly as before.

Design in ``coop2/ROUTE_SUPERVISION_PLAN.md`` section 5.1. The rules enforced
here are the ones that fail *late* without it: a support the BDDL never
declared is a support the scene may not have bound, and shows up as a task
that is quietly unachievable; a destination that disagrees with the BDDL goal
is the LL bug in COOHAVIOR's own files, where the BDDL says a bedroom floor and
the route they actually run says a bed.

Pure Python over ``bddl.parsing``. No OmniGibson import, so the CPU test can
load the real file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LIFT_ROLES",
    "POLICIES",
    "Route",
    "RouteNode",
    "RouteSpec",
    "load_route_spec",
    "parse_route_spec",
    "route_file_for",
]

#: What the ``lift`` table may name. ``arm`` is a robot with a hand, ``drone`` a
#: robot whose layout says so, ``carrier`` a robot with no arm. They are ours,
#: not COOHAVIOR's role words, because they are what the engine can tell apart.
LIFT_ROLES = ("arm", "drone", "carrier")

#: ``serial``: one route, one cargo (L*). ``parallel``: independent routes,
#: each with its own cargo (H*). Descriptive -- the tracker treats every route
#: independently either way -- so the prompt can say which it is.
POLICIES = ("serial", "parallel")


@dataclass(frozen=True)
class RouteNode:
    """One sub-goal: a BDDL predicate the cargo must satisfy, in turn."""

    id: str
    #: ``("ontop", "die.n.01_1", "bed.n.01_1")`` -- the same shape as a line of
    #: an activity's ``:goal`` and as ``BehaviorTaskState.goal``, so
    #: ``relation_holds`` evaluates it as it is.
    goal: Tuple[str, str, str]
    #: Room *type* of the support, from ``inroom`` in the BDDL ``:init``. For
    #: the prompt; the route file does not repeat it.
    room: Optional[str] = None

    @property
    def predicate(self) -> str:
        return self.goal[0]

    @property
    def cargo(self) -> str:
        return self.goal[1]

    @property
    def support(self) -> str:
        return self.goal[2]


@dataclass(frozen=True)
class Route:
    id: str
    cargo: str
    nodes: Tuple[RouteNode, ...]

    @property
    def destination(self) -> RouteNode:
        return self.nodes[-1]

    @property
    def checkpoints(self) -> Tuple[RouteNode, ...]:
        return self.nodes[:-1]


@dataclass(frozen=True)
class RouteSpec:
    activity: str
    policy: str
    routes: Tuple[Route, ...]
    #: cargo synset -> roles allowed to grasp it. A synset left out means anyone
    #: with a hand.
    lift: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    source: str = ""
    #: Loader warnings that are not errors -- kept on the spec so a run can
    #: print them once rather than lose them.
    warnings: Tuple[str, ...] = ()

    @property
    def required_nodes(self) -> int:
        """COOHAVIOR's denominator for ``Y_task``: every checkpoint plus every
        destination, ten for each of the twelve V4 tasks."""
        return sum(len(route.nodes) for route in self.routes)

    def may_lift(self, cargo_synset: str, role: str) -> bool:
        allowed = self.lift.get(cargo_synset)
        if allowed is None:
            return role != "carrier"
        return role in allowed


# ---------------------------------------------------------------------------
# Finding and reading the file
# ---------------------------------------------------------------------------


def route_file_for(activity: str) -> str:
    """``.../activity_definitions/<activity>/route.json``, beside problem0.bddl.

    Resolved through the same helper BDDL uses for the problem file, so the two
    cannot be found in different places.
    """
    from bddl.config import get_definition_filename  # noqa: PLC0415

    return os.path.join(os.path.dirname(get_definition_filename(activity, 0)), "route.json")


def load_route_spec(activity: Optional[str], domain: str = "behavior-1k") -> Optional[RouteSpec]:
    """The activity's route, or None when it has none.

    None is the common case and is not an error: every activity that predates
    this file, and every one whose goal really is a single final state.
    """
    if not activity:
        return None
    path = route_file_for(activity)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)

    from bddl.parsing import parse_problem  # noqa: PLC0415

    _name, objects, init, goal = parse_problem(activity, 0, domain)
    return parse_route_spec(payload, objects, goal, init, activity=activity)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _fail(where: str, message: str) -> ValueError:
    return ValueError(f"route.json {where}: {message}")


def _synset_of(instance: str) -> str:
    """``bed.n.01_2`` -> ``bed.n.01``. An id with no ``_N`` is its own synset."""
    base, sep, suffix = instance.rpartition("_")
    return base if sep and suffix.isdigit() else instance


def _declared_instances(objects: Any) -> Dict[str, str]:
    """``{instance: synset}`` from ``parse_problem``'s ``{synset: [instances]}``."""
    out: Dict[str, str] = {}
    for synset, instances in (objects or {}).items():
        for instance in instances or []:
            out[str(instance)] = str(synset)
    return out


def _rooms_from_init(init: Optional[Sequence]) -> Dict[str, str]:
    rooms: Dict[str, str] = {}
    for clause in init or []:
        if isinstance(clause, list) and len(clause) == 3 and clause[0] == "inroom":
            rooms[str(clause[1])] = str(clause[2])
    return rooms


def _goal_triples(goal: Optional[Sequence]) -> List[Tuple[str, str, str]]:
    """The flat ``(predicate, a, b)`` clauses of a BDDL goal.

    Only flat ones: a routed task's final state is one predicate per cargo, and
    a quantified goal has no single destination for a route to end on. Anything
    else is skipped rather than mis-read -- see CLAUDE.md, "A goal is a tree".
    """
    out: List[Tuple[str, str, str]] = []
    for clause in goal or []:
        if isinstance(clause, list) and len(clause) == 3 and all(isinstance(t, str) for t in clause):
            out.append((clause[0], clause[1], clause[2]))
    return out


def _parse_triple(raw: Any, where: str) -> Tuple[str, str, str]:
    if (not isinstance(raw, list) or len(raw) != 3
            or not all(isinstance(t, str) and t for t in raw)):
        raise _fail(where, f"'goal' must be a predicate triple like "
                           f"[\"ontop\", \"<cargo>\", \"<support>\"], got {raw!r}")
    return raw[0].lower(), raw[1], raw[2]


def _parse_node(raw: Any, where: str, declared: Dict[str, str], rooms: Dict[str, str],
                cargo: str) -> RouteNode:
    if not isinstance(raw, dict):
        raise _fail(where, f"must be an object, got {type(raw).__name__}")
    node_id = raw.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise _fail(where, f"'id' must be a non-empty string, got {node_id!r}")
    goal = _parse_triple(raw.get("goal"), f"{where} ({node_id})")
    for instance in goal[1:]:
        if instance not in declared:
            # The rule the whole file exists for. COOHAVIOR's own checkpoints
            # carry "add this object to BDDL :objects / :init" for the same
            # reason: a support the sampler did not bind is a support the scene
            # may not have, and nothing raises when a goal names one.
            raise _fail(f"{where} ({node_id})",
                        f"{instance} is not declared in the BDDL :objects; every id a "
                        f"node names must be, or the instance cannot bind it -- add it "
                        f"to :objects with an inroom line in :init")
    if goal[1] != cargo:
        # COOHAVIOR binds a box to its route by a name regex; ours is the
        # `cargo` field, and a node about a different object is a load error.
        raise _fail(f"{where} ({node_id})",
                    f"goal names {goal[1]} but the route's cargo is {cargo}")
    unknown = set(raw) - {"id", "goal", "comment", "_comment", "source"}
    if unknown:
        raise _fail(f"{where} ({node_id})",
                    f"unknown fields {sorted(unknown)}; order is list order, so a "
                    f"'sequence' or 'depends_on' field would encode it twice")
    return RouteNode(id=node_id, goal=goal, room=rooms.get(goal[2]))


def _parse_route(raw: Any, index: int, declared: Dict[str, str], rooms: Dict[str, str],
                 goal_triples: List[Tuple[str, str, str]],
                 warnings: List[str]) -> Route:
    where = f"routes[{index}]"
    if not isinstance(raw, dict):
        raise _fail(where, f"must be an object, got {type(raw).__name__}")
    route_id = raw.get("id")
    if not isinstance(route_id, str) or not route_id:
        raise _fail(where, f"'id' must be a non-empty string, got {route_id!r}")
    where = f"{where} ({route_id})"
    cargo = raw.get("cargo")
    if not isinstance(cargo, str) or not cargo:
        raise _fail(where, f"'cargo' must be a BDDL instance id, got {cargo!r}")
    if cargo not in declared:
        raise _fail(where, f"cargo {cargo} is not declared in the BDDL :objects")
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise _fail(where, "'nodes' must be a non-empty list")

    nodes = [_parse_node(n, f"{where}.nodes[{i}]", declared, rooms, cargo)
             for i, n in enumerate(raw_nodes)]

    seen: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        if node.id in seen:
            raise _fail(where, f"node id {node.id!r} appears twice (nodes {seen[node.id]} and {i})")
        seen[node.id] = i
    for earlier, later in zip(nodes, nodes[1:]):
        if earlier.goal == later.goal:
            # Not refused: a route may legitimately return to a support. But in
            # Merom_1_int every room has one floor object, so two floor nodes in
            # a row collapse to one predicate and the second scores for free.
            warnings.append(f"{where}: nodes {earlier.id} and {later.id} share the goal "
                            f"{list(earlier.goal)} -- the second will be satisfied the "
                            f"instant the first is")

    # The last node is the destination, and the BDDL :goal must say the same
    # for this cargo. Refused, not warned: this mismatch is exactly the LL
    # disagreement in COOHAVIOR's own files, and a run against both would end
    # on one and be scored on the other. The route is the authority (user,
    # 2026-09-12): when they disagree the BDDL is what gets corrected, so the
    # message says so.
    destination = nodes[-1]
    stated = [t for t in goal_triples if t[1] == cargo]
    if not stated:
        raise _fail(where, f"the BDDL :goal states no flat predicate for {cargo}, so the "
                           f"route has nothing to end on")
    if destination.goal not in stated:
        raise _fail(where, f"destination {destination.id} is {list(destination.goal)} but the "
                           f"BDDL :goal for {cargo} is {[list(t) for t in stated]}; the two "
                           f"files disagree on where the route ends -- the route is the task, "
                           f"so correct the BDDL :goal to match it")
    return Route(id=route_id, cargo=cargo, nodes=tuple(nodes))


def _parse_lift(raw: Any, declared: Dict[str, str]) -> Dict[str, Tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _fail("lift", f"must be an object of synset -> roles, got {type(raw).__name__}")
    known_synsets = set(declared.values())
    out: Dict[str, Tuple[str, ...]] = {}
    for synset, roles in raw.items():
        if synset not in known_synsets:
            raise _fail("lift", f"{synset} is not the synset of any declared object")
        if (not isinstance(roles, list) or not roles
                or not all(isinstance(r, str) for r in roles)):
            raise _fail("lift", f"{synset}: roles must be a non-empty list of strings, got {roles!r}")
        bad = [r for r in roles if r not in LIFT_ROLES]
        if bad:
            raise _fail("lift", f"{synset}: unknown roles {bad}; known: {list(LIFT_ROLES)}")
        out[synset] = tuple(roles)
    return out


def parse_route_spec(payload: Any, objects: Any, goal: Optional[Sequence],
                     init: Optional[Sequence] = None, activity: str = "") -> RouteSpec:
    """Validate a route.json payload against the parsed BDDL it sits beside.

    @objects, @goal and @init are ``parse_problem``'s three outputs. Kept as
    parameters rather than re-read here so a test can hand in a fabricated BDDL
    and check each rule in isolation.
    """
    if not isinstance(payload, dict):
        raise _fail("", f"top level must be an object, got {type(payload).__name__}")
    declared = _declared_instances(objects)
    if not declared:
        raise _fail("", "the BDDL declares no objects, so nothing can be routed")
    rooms = _rooms_from_init(init)
    goal_triples = _goal_triples(goal)

    name = payload.get("activity", activity)
    if activity and name != activity:
        raise _fail("activity", f"file says {name!r} but sits beside {activity!r}")

    policy = payload.get("policy", "serial")
    if policy not in POLICIES:
        raise _fail("policy", f"must be one of {list(POLICIES)}, got {policy!r}")

    raw_routes = payload.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise _fail("routes", "must be a non-empty list")
    warnings: List[str] = []
    routes = [_parse_route(r, i, declared, rooms, goal_triples, warnings)
              for i, r in enumerate(raw_routes)]

    ids = [r.id for r in routes]
    if len(set(ids)) != len(ids):
        raise _fail("routes", f"route ids must be unique, got {ids}")
    cargos = [r.cargo for r in routes]
    if len(set(cargos)) != len(cargos):
        raise _fail("routes", f"two routes share a cargo: {cargos}; a box is on one route")
    if policy == "serial" and len(routes) != 1:
        raise _fail("policy", f"'serial' means one route; there are {len(routes)}")

    unknown = set(payload) - {"activity", "source", "policy", "routes", "lift", "_comment", "comment"}
    if unknown:
        raise _fail("", f"unknown top-level fields {sorted(unknown)}")

    return RouteSpec(
        activity=str(name),
        policy=str(policy),
        routes=tuple(routes),
        lift=_parse_lift(payload.get("lift"), declared),
        source=str(payload.get("source", "")),
        warnings=tuple(warnings),
    )
