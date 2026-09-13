"""CPU-only test for RouteTracker: sub-goals judged on state, in order.

No Isaac, no GPU. The world is a stub answering the three questions the tracker
asks -- `relation_holds`, `held_objects`, `entity_id_of_name` -- and the spec is
built through the real loader from a fabricated BDDL, so the tracker is driven
exactly as it will be in coop_env.

Run:
    python feasibility_verify/test_route_tracker_stubbed.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.route_spec import parse_route_spec  # noqa: E402
from coop2.behavior_env.route_tracker import RouteTracker  # noqa: E402


def ok(message: str) -> None:
    print(f"  ok: {message}")


class StubWorld:
    """Facts are set by the test; the tracker reads them the way it reads
    BehaviorWorldState. Every `relation_holds` call is counted so the test can
    assert the evaluation budget, not just the answers."""

    def __init__(self, names=None):
        self.facts = set()            # (token, source_id, target_id)
        self.held = {}                # object *name* -> robot name
        self.names = dict(names or {})  # object name -> entity id
        self.asked = []

    def relation_holds(self, token, source_id, target_id):
        self.asked.append((token, source_id, target_id))
        return (token, source_id, target_id) in self.facts

    def held_objects(self):
        return dict(self.held)

    def entity_id_of_name(self, name):
        return self.names.get(name)

    # -- test helpers
    def place(self, cargo, support):
        self.facts = {f for f in self.facts if not (f[0] == "ontop" and f[1] == cargo)}
        self.facts.add(("ontop", cargo, support))

    def lift(self, cargo):
        self.facts = {f for f in self.facts if not (f[0] == "ontop" and f[1] == cargo)}


OBJECTS = {
    "die.n.01": ["die.n.01_1"],
    "notebook.n.01": ["notebook.n.01_1"],
    "floor.n.01": ["floor.n.01_1"],
    "cabinet.n.01": ["cabinet.n.01_1"],
    "shelf.n.01": ["shelf.n.01_1"],
    "bed.n.01": ["bed.n.01_1", "bed.n.01_2"],
    "agent.n.01": ["agent.n.01_1"],
}
INIT = [["inroom", "cabinet.n.01_1", "childs_room"], ["inroom", "shelf.n.01_1", "kitchen"],
        ["inroom", "bed.n.01_2", "bedroom"], ["inroom", "bed.n.01_1", "childs_room"]]


def serial_spec():
    payload = {
        "activity": "fake", "policy": "serial",
        "routes": [{"id": "R", "cargo": "die.n.01_1", "nodes": [
            {"id": "C1", "goal": ["ontop", "die.n.01_1", "cabinet.n.01_1"]},
            {"id": "C2", "goal": ["ontop", "die.n.01_1", "shelf.n.01_1"]},
            {"id": "C3", "goal": ["ontop", "die.n.01_1", "bed.n.01_2"]},
            {"id": "D", "goal": ["ontop", "die.n.01_1", "bed.n.01_1"]},
        ]}],
    }
    return parse_route_spec(payload, OBJECTS, [["ontop", "die.n.01_1", "bed.n.01_1"]], INIT, activity="fake")


def parallel_spec():
    payload = {
        "activity": "fake", "policy": "parallel",
        "routes": [
            {"id": "A", "cargo": "die.n.01_1", "nodes": [
                {"id": "C1", "goal": ["ontop", "die.n.01_1", "cabinet.n.01_1"]},
                {"id": "D", "goal": ["ontop", "die.n.01_1", "bed.n.01_1"]}]},
            {"id": "B", "cargo": "notebook.n.01_1", "nodes": [
                {"id": "C1", "goal": ["ontop", "notebook.n.01_1", "shelf.n.01_1"]},
                {"id": "D", "goal": ["ontop", "notebook.n.01_1", "bed.n.01_2"]}]},
        ],
    }
    goal = [["ontop", "die.n.01_1", "bed.n.01_1"], ["ontop", "notebook.n.01_1", "bed.n.01_2"]]
    return parse_route_spec(payload, OBJECTS, goal, INIT, activity="fake")


def statuses(events):
    return [(e.node_id, e.status) for e in events]


def main() -> int:
    print("test 1: a route completes in order, one node per placement")
    world = StubWorld()
    tracker = RouteTracker(world, serial_spec())
    assert tracker.step(0) == [] and tracker.next_node("R").id == "C1"
    world.place("die.n.01_1", "cabinet.n.01_1")
    assert statuses(tracker.step(100)) == [("C1", "completed")]
    assert tracker.next_node("R").id == "C2"
    world.place("die.n.01_1", "shelf.n.01_1")
    assert statuses(tracker.step(200)) == [("C2", "completed")]
    world.place("die.n.01_1", "bed.n.01_2")
    assert statuses(tracker.step(300)) == [("C3", "completed")]
    assert not tracker.complete()
    world.place("die.n.01_1", "bed.n.01_1")
    assert statuses(tracker.step(400)) == [("D", "completed")]
    assert tracker.complete() and tracker.next_node("R") is None
    assert tracker.summary()["Y_task"] == 1.0 and tracker.summary()["completed_nodes"] == 4
    ok("C1 -> C2 -> C3 -> D, complete() true only on D")

    print("test 2: a box still sitting where it scored is not scored again")
    world = StubWorld()
    tracker = RouteTracker(world, serial_spec())
    world.place("die.n.01_1", "cabinet.n.01_1")
    assert statuses(tracker.step(100)) == [("C1", "completed")]
    assert tracker.step(101) == [] and tracker.step(102) == []
    assert tracker.completed_count() == 1
    # And earlier nodes are never asked about again -- state detection cannot
    # tell "still there" from "visited twice", so it does not try.
    world.asked.clear()
    tracker.step(103)
    assert ("ontop", "die.n.01_1", "cabinet.n.01_1") not in world.asked, world.asked
    ok("no duplicate events, and a scored node is never re-evaluated")

    print("test 3: out of order is recorded, credited to nobody, and progress waits")
    world = StubWorld()
    tracker = RouteTracker(world, serial_spec())
    world.place("die.n.01_1", "bed.n.01_2")           # C3, while C1 is next
    events = tracker.step(100)
    assert statuses(events) == [("C3", "out_of_order")], statuses(events)
    assert events[0].expected == "C1" and events[0].by is None
    assert tracker.next_node("R").id == "C1", "nothing advanced"
    assert tracker.completed_count() == 0
    assert tracker.summary()["out_of_order_visits"] == 1
    # Still there next step: recorded again -- it is still out of order -- but
    # the route does not move.
    assert statuses(tracker.step(101)) == [("C3", "out_of_order")]
    assert tracker.completed_count() == 0
    ok("a skipped support does not count, and is not punished either")

    print("test 4: progress resumes the instant the expected node holds")
    world.place("die.n.01_1", "cabinet.n.01_1")      # back to C1
    assert statuses(tracker.step(200)) == [("C1", "completed")]
    world.place("die.n.01_1", "shelf.n.01_1")
    assert statuses(tracker.step(300)) == [("C2", "completed")]
    world.place("die.n.01_1", "bed.n.01_2")          # C3 again, now in order
    assert statuses(tracker.step(400)) == [("C3", "completed")]
    progress = tracker.progress()["R"]
    assert progress.completed == ["C1", "C2", "C3"] and progress.next == "D"
    assert progress.remaining == [] and progress.out_of_order == 2
    ok("the same support that was out of order scores when its turn comes")

    print("test 5: a placement that satisfies the next two nodes scores both")
    # Two consecutive nodes on one support -- the loader warns about it, and the
    # tracker must not stall on the second.
    payload = {"activity": "fake", "policy": "serial",
               "routes": [{"id": "R", "cargo": "die.n.01_1", "nodes": [
                   {"id": "C1", "goal": ["ontop", "die.n.01_1", "shelf.n.01_1"]},
                   {"id": "C1b", "goal": ["ontop", "die.n.01_1", "shelf.n.01_1"]},
                   {"id": "D", "goal": ["ontop", "die.n.01_1", "bed.n.01_1"]}]}]}
    spec = parse_route_spec(payload, OBJECTS, [["ontop", "die.n.01_1", "bed.n.01_1"]], INIT, activity="fake")
    world = StubWorld()
    tracker = RouteTracker(world, spec)
    world.place("die.n.01_1", "shelf.n.01_1")
    assert statuses(tracker.step(1)) == [("C1", "completed"), ("C1b", "completed")]
    ok("the loop after a completion re-checks, so a shared support is not a stall")

    print("test 6: two parallel routes advance independently")
    world = StubWorld()
    tracker = RouteTracker(world, parallel_spec())
    world.place("notebook.n.01_1", "shelf.n.01_1")     # B.C1 only
    events = tracker.step(10)
    assert [(e.route_id, e.node_id, e.status) for e in events] == [("B", "C1", "completed")]
    assert tracker.next_node("A").id == "C1" and tracker.next_node("B").id == "D"
    world.place("die.n.01_1", "bed.n.01_1")            # A's destination while A.C1 is next
    events = tracker.step(20)
    assert [(e.route_id, e.node_id, e.status) for e in events] == [("A", "D", "out_of_order")]
    world.place("die.n.01_1", "cabinet.n.01_1")
    world.place("notebook.n.01_1", "bed.n.01_2")
    events = tracker.step(30)
    assert sorted((e.route_id, e.node_id) for e in events) == [("A", "C1"), ("B", "D")]
    assert not tracker.complete(), "A still has D to do"
    world.place("die.n.01_1", "bed.n.01_1")
    tracker.step(40)
    assert tracker.complete()
    summary = tracker.summary()
    assert summary["required_nodes"] == 4 and summary["completed_nodes"] == 4
    assert summary["routes"]["A"]["out_of_order"] == 1 and summary["routes"]["B"]["out_of_order"] == 0
    ok("routes are independent; the task completes when the last of them does")

    print("test 7: credit goes to the one robot that let go, if it is the one that acted")
    world = StubWorld(names={"dice_154": "die.n.01_1"})
    tracker = RouteTracker(world, serial_spec())
    world.held = {"dice_154": "agent_0"}
    tracker.step(0, acting={})                        # baseline: agent_0 holds the die
    world.held = {}
    world.place("die.n.01_1", "cabinet.n.01_1")
    events = tracker.step(100, acting={"agent_0": None})
    assert statuses(events) == [("C1", "completed")] and events[0].by == "agent_0"
    ok("held last step, not now, and its primitive terminated -> by=agent_0")

    print("test 8: credit is None whenever it would be a guess")
    # (a) the previous holder is not among the agents whose primitive ended
    world = StubWorld(names={"dice_154": "die.n.01_1"})
    tracker = RouteTracker(world, serial_spec())
    world.held = {"dice_154": "agent_0"}
    tracker.step(0)
    world.held = {}
    world.place("die.n.01_1", "cabinet.n.01_1")
    assert tracker.step(100, acting={"agent_1": None})[0].by is None
    # (b) nobody was holding it before -- a drone's release seen late, say
    world = StubWorld(names={"dice_154": "die.n.01_1"})
    tracker = RouteTracker(world, serial_spec())
    tracker.step(0)
    world.place("die.n.01_1", "cabinet.n.01_1")
    assert tracker.step(100, acting={"agent_0": None})[0].by is None
    # (c) the holder is still holding it (the predicate held for another reason)
    world = StubWorld(names={"dice_154": "die.n.01_1"})
    tracker = RouteTracker(world, serial_spec())
    world.held = {"dice_154": "agent_0"}
    tracker.step(0)
    world.place("die.n.01_1", "cabinet.n.01_1")       # still held
    assert tracker.step(100, acting={"agent_0": None})[0].by is None
    ok("wrong actor, no prior holder, or still held -> unattributed, not guessed")

    print("test 9: a world with no inventories is fine; attribution is just None")
    class Bare:
        def __init__(self): self.facts = set()
        def relation_holds(self, token, a, b): return (token, a, b) in self.facts
    world = Bare()
    tracker = RouteTracker(world, serial_spec())
    world.facts.add(("ontop", "die.n.01_1", "cabinet.n.01_1"))
    events = tracker.step(1, acting={"agent_0": None})
    assert statuses(events) == [("C1", "completed")] and events[0].by is None
    ok("the tracker needs relation_holds and nothing more")

    print("test 10: the evaluation budget is routes x remaining nodes, never more")
    world = StubWorld()
    tracker = RouteTracker(world, serial_spec())
    world.asked.clear()
    tracker.step(1)
    assert len(world.asked) == 4, world.asked                    # C1 expected + 3 later
    world.place("die.n.01_1", "cabinet.n.01_1")
    world.asked.clear()
    tracker.step(2)
    # C1 holds (1), then C2 asked and false (1), then later nodes C3, D (2).
    assert len(world.asked) == 4, world.asked
    world.asked.clear()
    tracker.step(3)
    assert len(world.asked) == 3, world.asked                    # C2 expected + C3, D
    ok("ten predicates a macro-step for any V4 task, and no fact-set build")

    print("test 11: events serialise, and history keeps every one in order")
    world = StubWorld()
    tracker = RouteTracker(world, serial_spec())
    world.place("die.n.01_1", "shelf.n.01_1")
    tracker.step(5)
    world.place("die.n.01_1", "cabinet.n.01_1")
    tracker.step(6)
    history = [e.to_dict() for e in tracker.history()]
    assert [(h["node"], h["status"], h["expected"], h["env_step"]) for h in history] == [
        ("C2", "out_of_order", "C1", 5), ("C1", "completed", "C1", 6)]
    assert history[0]["goal"] == ["ontop", "die.n.01_1", "shelf.n.01_1"]
    ok("route_progress.json will carry the same record the run printed")

    print("test 12: the prompt block shows the whole route, marked")
    from coop2.behavior_env.symbolic_view import render_route_block
    world = StubWorld()
    tracker = RouteTracker(world, serial_spec())
    text = render_route_block(tracker.spec, tracker.progress())
    rows = [l for l in text.splitlines() if l.startswith("  ") and not l.startswith("  it starts")]
    assert len(rows) == 4, text                      # every node, from step 0
    assert rows[0].startswith("  NEXT  C1") and "[childs_room]" in rows[0], rows[0]
    assert "done" not in text
    if tracker.spec.routes[0].start_support:
        assert "it starts ontop(die.n.01_1" in text, text
    world.place("die.n.01_1", "cabinet.n.01_1"); tracker.step(1)
    text = render_route_block(tracker.spec, tracker.progress())
    assert "it starts" not in text, text             # moved: the done marks say where it is
    rows = [l for l in text.splitlines() if l.startswith("  ")]
    assert rows[0].startswith("  done  C1") and rows[1].startswith("  NEXT  C2"), text
    assert rows[2].startswith("        C3") and "[bedroom]" in rows[2], rows[2]
    assert "does not count" in text and "NEXT" in text.splitlines()[-1]
    for node in ("C1", "C2", "C3", "D"):
        assert node in text
    ok("all nodes listed, done/NEXT marks move, each support's room stated")

    print("test 13: a completed route is one line; parallel routes are one block each")
    world = StubWorld()
    tracker = RouteTracker(world, parallel_spec())
    world.place("notebook.n.01_1", "shelf.n.01_1"); tracker.step(1)
    world.place("notebook.n.01_1", "bed.n.01_2"); tracker.step(2)
    text = render_route_block(tracker.spec, tracker.progress())
    assert "Route B: complete" in text, text
    assert "Route A, carry die.n.01_1" in text and "  NEXT  C1" in text
    ok("finished routes collapse, unfinished ones keep their marks")

    print("test 14: the saver writes what compute_metrics reads, and Y_task counts nodes")
    import json, tempfile
    from coop2.cognitive.plan.plan_log_saver import save_route_progress
    from coop2.cognitive.compute_metrics import compute_task_success_metrics, load_logs
    world = StubWorld(names={"dice_154": "die.n.01_1"})
    tracker = RouteTracker(world, serial_spec())
    world.held = {"dice_154": "agent_0"}; tracker.step(0)
    world.held = {}; world.place("die.n.01_1", "cabinet.n.01_1")
    tracker.step(100, acting={"agent_0": None})              # C1 by agent_0
    world.place("die.n.01_1", "bed.n.01_2"); tracker.step(150)  # C3 out of order
    world.place("die.n.01_1", "shelf.n.01_1"); tracker.step(200)  # C2, unattributed
    with tempfile.TemporaryDirectory() as run_dir:
        save_route_progress(tracker, os.path.join(run_dir, "route_progress.json"))
        with open(os.path.join(run_dir, "team_timeline.json"), "w") as handle:
            json.dump({"teams": {"alpha": ["agent_0", "agent_1"]}, "spans": {}}, handle)
        logs = load_logs(run_dir)
        assert "route_progress" in logs and "team_timeline" in logs
        # No plan_logs.json here: an unrouted run returned {} and still does; a
        # routed one must still report its nodes.
        metrics = compute_task_success_metrics(logs)
    assert metrics["required_nodes"] == 4 and metrics["completed_nodes"] == 2
    assert abs(metrics["Y_task"] - 0.5) < 1e-9, metrics
    assert metrics["out_of_order_visits"] == 1 and metrics["route_complete"] is False
    assert metrics["S_team"] == {"alpha": 1, "unattributed": 1}, metrics["S_team"]
    assert compute_task_success_metrics({}) == {}, "an unrouted run is unchanged"
    ok("route_progress.json -> Y_task 2/4, S_team by team with unattributed kept apart")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
