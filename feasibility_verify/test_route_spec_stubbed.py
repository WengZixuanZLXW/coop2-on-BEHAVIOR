"""CPU-only test for the route.json sidecar and its loader.

No Isaac, no GPU: `bddl.parsing` reads the real activity definition and the
loader validates the real route file against it. Worth its own suite because
every rule here fails *late* without it -- a support the BDDL never declared is
a task that is quietly unachievable, and a destination that disagrees with the
BDDL goal is a run that ends on one file and is scored on the other.

Run:
    python feasibility_verify/test_route_spec_stubbed.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.route_spec import (  # noqa: E402
    LIFT_ROLES,
    load_route_spec,
    parse_route_spec,
    route_file_for,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


def rejects(payload, objects, goal, init, needle: str) -> str:
    """Assert parse fails, and that the message says why."""
    try:
        parse_route_spec(payload, objects, goal, init, activity="fake")
    except ValueError as error:
        message = str(error)
        assert needle in message, f"rejected, but for the wrong reason:\n  {message}\n  wanted {needle!r}"
        return message
    raise AssertionError(f"accepted a payload that should have been refused ({needle!r})")


# A small fabricated BDDL, in the shapes parse_problem returns.
OBJECTS = {
    "die.n.01": ["die.n.01_1"],
    "floor.n.01": ["floor.n.01_1"],
    "shelf.n.01": ["shelf.n.01_1"],
    "bed.n.01": ["bed.n.01_1", "bed.n.01_2"],
    "notebook.n.01": ["notebook.n.01_1"],
    "agent.n.01": ["agent.n.01_1"],
}
INIT = [
    ["ontop", "die.n.01_1", "floor.n.01_1"],
    ["inroom", "floor.n.01_1", "childs_room"],
    ["inroom", "shelf.n.01_1", "kitchen"],
    ["inroom", "bed.n.01_2", "bedroom"],
    ["inroom", "bed.n.01_1", "childs_room"],
]
GOAL = [["ontop", "die.n.01_1", "bed.n.01_1"]]
PAYLOAD = {
    "activity": "fake",
    "policy": "serial",
    "routes": [{
        "id": "R",
        "cargo": "die.n.01_1",
        "nodes": [
            {"id": "C1", "goal": ["ontop", "die.n.01_1", "shelf.n.01_1"]},
            {"id": "C2", "goal": ["ontop", "die.n.01_1", "bed.n.01_2"]},
            {"id": "D", "goal": ["ontop", "die.n.01_1", "bed.n.01_1"]},
        ],
    }],
    "lift": {"die.n.01": ["arm", "drone"]},
}


def main() -> int:
    print("test 1: the real v4_s1_v4_ll route loads against its real BDDL")
    path = route_file_for("v4_s1_v4_ll")
    assert os.path.exists(path), path
    spec = load_route_spec("v4_s1_v4_ll")
    assert spec is not None
    assert spec.policy == "serial" and len(spec.routes) == 1
    route = spec.routes[0]
    assert route.id == "S1-M5" and route.cargo == "die.n.01_1"
    assert [n.id for n in route.nodes] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "D"]
    assert spec.required_nodes == 10, "COOHAVIOR's Y_task denominator for every V4 task"
    assert route.destination.goal == ("ontop", "die.n.01_1", "bed.n.01_1")
    # Every node's room comes from the BDDL :init, not from the route file.
    rooms = [n.room for n in route.nodes]
    assert rooms == ["childs_room", "kitchen", "kitchen", "dining_room", "dining_room",
                     "living_room", "living_room", "bedroom", "bedroom", "childs_room"], rooms
    assert spec.warnings == (), spec.warnings
    ok("ten nodes in COOHAVIOR's order, each with its room, ending on bed.n.01_1")

    print("test 1b: every S1 task's route loads against its BDDL, in COOHAVIOR's shape")
    # LL/LH: one box through ten supports. HL/HH: five boxes, one station on
    # each. The heavy tasks' notebook may be lifted by an arm only.
    expect = {
        "v4_s1_v4_ll": ("serial", 1, 10, {"die.n.01": ("arm", "drone")}),
        "v4_s1_v4_lh": ("serial", 1, 10, {"notebook.n.01": ("arm",)}),
        "v4_s1_v4_hl": ("parallel", 5, 10, {"die.n.01": ("arm", "drone")}),
        "v4_s1_v4_hh": ("parallel", 5, 10, {"notebook.n.01": ("arm",)}),
    }
    for activity, (policy, n_routes, n_nodes, lift) in expect.items():
        spec = load_route_spec(activity)
        assert spec is not None, activity
        assert (spec.policy, len(spec.routes), spec.required_nodes) == (policy, n_routes, n_nodes), (
            activity, spec.policy, len(spec.routes), spec.required_nodes)
        assert spec.lift == lift, (activity, spec.lift)
        assert spec.warnings == (), (activity, spec.warnings)
        # Every node has a room: the BDDL declares inroom for every support.
        assert all(n.room for r in spec.routes for n in r.nodes), activity
    # LH is LL's route with the cargo swapped and nothing else.
    ll = load_route_spec("v4_s1_v4_ll").routes[0]; lh = load_route_spec("v4_s1_v4_lh").routes[0]
    assert [n.support for n in ll.nodes] == [n.support for n in lh.nodes]
    assert lh.cargo == "notebook.n.01_1" and ll.cargo == "die.n.01_1"
    # HL and HH are the same five routes with the cargo swapped; they chain --
    # M5's destination is M3's checkpoint -- and no node is a floor, so nothing
    # can be true at load.
    hl = load_route_spec("v4_s1_v4_hl"); hh = load_route_spec("v4_s1_v4_hh")
    assert [(r.id, [n.support for n in r.nodes]) for r in hl.routes] == \
           [(r.id, [n.support for n in r.nodes]) for r in hh.routes]
    by_id = {r.id: r for r in hl.routes}
    assert by_id["S1-M5"].destination.support == by_id["S1-M3"].checkpoints[0].support == "bookcase.n.01_1"
    assert by_id["S1-M3"].destination.support == by_id["S1-M2"].checkpoints[0].support == "armchair.n.01_1"
    assert not any(n.support.startswith("floor.") for r in hl.routes for n in r.nodes)
    ok("four S1 route files, each the shape COOHAVIOR's tasks.modified.json gives it")

    print("test 1c: every S1 definition survives BehaviorTask's wildcard pass")
    # `_strip_wildcards` treats any line containing `*` as a wildcard scope
    # line and indexes split(" - ")[1] -- so a `*` in a comment with a ` -- ` in
    # it raises IndexError inside Environment.__init__, after the scene loads.
    # Two GPU runs found that; this finds it in a millisecond.
    from bddl.knowledge_base.models import Task as _Task
    for activity in expect:
        text = open(os.path.dirname(route_file_for(activity)) + "/problem0.bddl").read()
        assert "*" not in text, f"{activity}: a '*' anywhere in the file is a wildcard to BehaviorTask"
        _Task._strip_wildcards(_Task.__new__(_Task), text)
    ok("no '*' in any S1 definition, and the pass runs clean on all four")

    print("test 2: an activity without a route file is simply unrouted")
    assert load_route_spec("coop_two_apples_pomaria") is None
    assert load_route_spec(None) is None
    ok("None, not an error: every activity that predates this file")

    print("test 3: a node may name only objects the BDDL declares")
    bad = copy.deepcopy(PAYLOAD)
    bad["routes"][0]["nodes"][0]["goal"][2] = "armchair.n.01_1"
    rejects(bad, OBJECTS, GOAL, INIT, "not declared in the BDDL :objects")
    ok("an undeclared support is refused up front, not discovered as an unbindable task")

    print("test 4: the destination must be the BDDL goal, exactly")
    bad = copy.deepcopy(PAYLOAD)
    bad["routes"][0]["nodes"][-1]["goal"][2] = "bed.n.01_2"
    rejects(bad, OBJECTS, GOAL, INIT, "the two files disagree on where the route ends")
    # And a goal that states nothing flat for the cargo has nothing to end on.
    rejects(PAYLOAD, OBJECTS, [["forall", ["?d", "-", "die.n.01"], ["ontop", "?d", "bed.n.01_1"]]],
            INIT, "states no flat predicate")
    ok("COOHAVIOR's own LL bug -- BDDL says a floor, the route says a bed -- is a load error")

    print("test 5: a node is about the route's cargo and nothing else")
    bad = copy.deepcopy(PAYLOAD)
    bad["routes"][0]["nodes"][1]["goal"][1] = "notebook.n.01_1"
    rejects(bad, OBJECTS, GOAL, INIT, "route's cargo is die.n.01_1")
    ok("a node naming another object is refused")

    print("test 6: a goal is a predicate triple and nothing else")
    for wrong in (["ontop", "die.n.01_1"], "ontop(die, bed)", ["ontop", "die.n.01_1", "bed.n.01_1", "extra"],
                  {"predicate": "ontop"}):
        bad = copy.deepcopy(PAYLOAD)
        bad["routes"][0]["nodes"][0]["goal"] = wrong
        rejects(bad, OBJECTS, GOAL, INIT, "predicate triple")
    ok("no marker ids, no coordinates, no runtime names")

    print("test 7: order is list order, so a node may not carry its own")
    for field_name in ("sequence", "depends_on"):
        bad = copy.deepcopy(PAYLOAD)
        bad["routes"][0]["nodes"][0][field_name] = 1
        rejects(bad, OBJECTS, GOAL, INIT, "encode it twice")
    ok("'sequence' and 'depends_on' are refused")

    print("test 8: ids are unique, routes are unique, a cargo is on one route")
    bad = copy.deepcopy(PAYLOAD)
    bad["routes"][0]["nodes"][1]["id"] = "C1"
    rejects(bad, OBJECTS, GOAL, INIT, "appears twice")
    two = copy.deepcopy(PAYLOAD)
    two["policy"] = "parallel"
    two["routes"].append(copy.deepcopy(two["routes"][0]))
    rejects(two, OBJECTS, GOAL, INIT, "route ids must be unique")
    two["routes"][1]["id"] = "R2"
    rejects(two, OBJECTS, GOAL, INIT, "share a cargo")
    ok("duplicates refused at every level")

    print("test 9: 'serial' means one route; 'parallel' allows several")
    two = copy.deepcopy(PAYLOAD)
    two["routes"].append({
        "id": "R2", "cargo": "notebook.n.01_1",
        "nodes": [{"id": "D", "goal": ["ontop", "notebook.n.01_1", "bed.n.01_2"]}],
    })
    goal2 = GOAL + [["ontop", "notebook.n.01_1", "bed.n.01_2"]]
    rejects(two, OBJECTS, goal2, INIT, "'serial' means one route")
    two["policy"] = "parallel"
    spec = parse_route_spec(two, OBJECTS, goal2, INIT, activity="fake")
    assert [r.id for r in spec.routes] == ["R", "R2"] and spec.required_nodes == 4
    rejects(dict(PAYLOAD, policy="chained"), OBJECTS, GOAL, INIT, "must be one of")
    ok("the H* shape parses; an unknown policy does not")

    print("test 10: consecutive nodes with one goal warn, because the second is free")
    dup = copy.deepcopy(PAYLOAD)
    dup["routes"][0]["nodes"].insert(1, {"id": "C1b", "goal": ["ontop", "die.n.01_1", "shelf.n.01_1"]})
    spec = parse_route_spec(dup, OBJECTS, GOAL, INIT, activity="fake")
    assert len(spec.warnings) == 1 and "satisfied the instant" in spec.warnings[0], spec.warnings
    ok("warned, not refused -- a route may return to a support, and a floor collapses")

    print("test 11: the lift table names declared synsets and known roles")
    assert set(LIFT_ROLES) == {"arm", "drone", "carrier"}
    rejects(dict(PAYLOAD, lift={"apple.n.01": ["arm"]}), OBJECTS, GOAL, INIT, "not the synset of any declared object")
    rejects(dict(PAYLOAD, lift={"die.n.01": ["forklift"]}), OBJECTS, GOAL, INIT, "unknown roles")
    rejects(dict(PAYLOAD, lift={"die.n.01": []}), OBJECTS, GOAL, INIT, "non-empty list")
    spec = parse_route_spec(PAYLOAD, OBJECTS, GOAL, INIT, activity="fake")
    # Where the cargo starts, read off :init -- the robots may spawn rooms away.
    assert spec.routes[0].start_support == "floor.n.01_1", spec.routes[0]
    assert spec.routes[0].start_room == "childs_room", spec.routes[0]
    assert spec.may_lift("die.n.01", "drone") and spec.may_lift("die.n.01", "arm")
    assert not spec.may_lift("die.n.01", "carrier")
    # A synset the table leaves out: anyone with a hand.
    assert spec.may_lift("notebook.n.01", "arm") and spec.may_lift("notebook.n.01", "drone")
    assert not spec.may_lift("notebook.n.01", "carrier"), "a carrier never lifts"
    ok("lift rules parse and answer; left out means anyone with a hand")

    print("test 12: the file must be about the activity it sits beside")
    rejects(dict(PAYLOAD, activity="other"), OBJECTS, GOAL, INIT, "sits beside")
    rejects(dict(PAYLOAD, bonus=1), OBJECTS, GOAL, INIT, "unknown top-level fields")
    ok("a misplaced or over-specified file is refused")

    print("test 13: the route file on disk round-trips through json unchanged")
    with open(path) as handle:
        raw = json.load(handle)
    assert raw["activity"] == "v4_s1_v4_ll" and len(raw["routes"][0]["nodes"]) == 10
    ok("the transcription is what the loader read")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
