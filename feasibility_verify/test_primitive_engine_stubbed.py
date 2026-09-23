"""CPU-only regression test for the assign/tick control flow.

Stubs out OmniGibson entirely, so it runs anywhere -- no Isaac, no GPU, no
scene load. It cannot tell you whether a grasp works; it tells you whether the
*scheduling* is right, which is the part that must stay isomorphic to COOP2's
``PlanningEnvWrapper.step``:

    * agents advance through their own primitive sequences independently
      (no alignment at primitive boundaries);
    * a failed primitive kills the rest of that agent's plan;
    * every robot gets an action key on every tick;
    * abort / retract, target validation, and the COOP2 record shape.

Run:
    python feasibility_verify/test_primitive_engine_stubbed.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Stub the OmniGibson surface the engine imports
# ---------------------------------------------------------------------------


class FakeReason:
    """Mimics an ``ActionPrimitiveError.Reason`` member (IntEnum -> has .name)."""

    def __init__(self, name: str):
        self.name = name


class FakeActionPrimitiveError(ValueError):
    def __init__(self, reason, message, metadata=None):
        self.reason = reason
        self.metadata = metadata or {}
        super().__init__(f"{reason.name}: {message}")


class FakeActionPrimitiveErrorGroup(ValueError):
    def __init__(self, exceptions):
        self._exceptions = list(exceptions)

    @property
    def exceptions(self):
        return self._exceptions


class FakeRobotBase:
    pass


class FakePrimitiveSet:
    class _Member:
        def __init__(self, name):
            self.name = name

    NAVIGATE_TO = _Member("NAVIGATE_TO")
    GRASP = _Member("GRASP")
    PLACE_ON_TOP = _Member("PLACE_ON_TOP")
    RELEASE = _Member("RELEASE")


def _install_stubs():
    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    stub("torch", Tensor=object)
    stub("omnigibson")
    stub("omnigibson.action_primitives")
    stub(
        "omnigibson.action_primitives.action_primitive_set_base",
        ActionPrimitiveError=FakeActionPrimitiveError,
        ActionPrimitiveErrorGroup=FakeActionPrimitiveErrorGroup,
    )
    stub(
        "omnigibson.action_primitives.starter_semantic_action_primitives",
        StarterSemanticActionPrimitives=object,
        StarterSemanticActionPrimitiveSet=FakePrimitiveSet,
    )
    stub("omnigibson.robots", Robot=FakeRobotBase)


def _register_real(*stems):
    """Put real ``coop2.behavior_env.<stem>`` modules in sys.modules.

    The module under test imports these by their package path. Both are pure
    stdlib -- ``geometry_cache`` is a dict of world AABBs valid for one tick,
    ``carrier`` is a handful of getattr predicates (``_level_drones`` asks it
    which robots are drones) -- so the real files are loaded rather than
    stubbed; only the package namespace around them is faked, the way this
    file fakes omnigibson.
    """
    import types

    for package_name in ("coop2", "coop2.behavior_env"):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
    for stem in stems:
        name = f"coop2.behavior_env.{stem}"
        if name in sys.modules:
            continue
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "coop2", "behavior_env", f"{stem}.py",
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def _load_engine_module():
    _register_real("geometry_cache", "carrier")
    """Import primitive_engine.py directly, after the stubs are in place."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coop2",
        "behavior_env",
        "primitive_engine.py",
    )
    spec = importlib.util.spec_from_file_location("coop2_primitive_engine_stubbed", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module from sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeNamed:
    def __init__(self, name):
        self.name = name


class FakeRobot:
    def __init__(self, name):
        self.name = name
        self.model = "r1"
        self.action_dim = 1

    def q_to_action(self, q):
        return f"idle:{self.name}"

    def get_joint_positions(self):
        return 0


class FakeController:
    """Scripted controller.

    ``script`` maps a primitive name to either an int (number of ticks, then
    success) or ``("fail", ticks, reason_name)``.
    """

    def __init__(self, script):
        self.script = script
        self.held = None

    def navigate_to_room(self, room):
        self.rooms_visited = getattr(self, "rooms_visited", []) + [room]
        yield f"room:{room}"

    def navigate_to_robot(self, robot):
        self.robots_visited = getattr(self, "robots_visited", []) + [robot.name]
        yield f"robot:{robot.name}"

    def apply_ref(self, primitive, *args, attempts=1):
        plan = self.script[primitive.name]
        if isinstance(plan, tuple):
            _, ticks, reason = plan
            for i in range(ticks):
                yield f"act{i}"
            raise FakeActionPrimitiveError(FakeReason(reason), "scripted failure")
        for i in range(plan):
            yield f"act{i}"
        if primitive.name == "GRASP" and args:
            self.held = args[0]

    def _get_obj_in_hand(self):
        return self.held

    def _reset_robot(self):
        for _ in range(3):
            yield "retract"


class FakeScene:
    def __init__(self, objects):
        self._objects = objects

    def object_registry(self, key, name):
        return self._objects.get(name)


class FakeEnv:
    def __init__(self, robots, objects):
        self.robots = robots
        self.scene = FakeScene(objects)
        self.steps = 0
        self.last_action = None

    def step(self, action):
        self.steps += 1
        self.last_action = dict(action)
        return {}, {}, {}, {}, {}


AGENT_IDS = ["agent_0", "agent_1"]


def make_engine(engine_module, motion_mode, scripts):
    objects = {"apple_0": FakeNamed("apple_0"), "apple_1": FakeNamed("apple_1")}
    robots = [FakeRobot(agent_id) for agent_id in AGENT_IDS]
    env = FakeEnv(robots, objects)
    engine = engine_module.MultiAgentPrimitiveEngine(
        env,
        agent_ids=AGENT_IDS,
        motion_mode=motion_mode,
        progress_every=0,
        verbose=False,
        controllers={agent_id: FakeController(scripts[agent_id]) for agent_id in AGENT_IDS},
    )
    return engine, env


def run_plans(engine, plans: Dict[str, List[Tuple[object, str]]], max_ticks: int = 100000):
    """Same loop as the demo's run_plans; kept duplicated so this file stands alone."""
    cursors = {agent_id: 0 for agent_id in plans}
    results = []
    while True:
        for agent_id, steps in plans.items():
            if engine.has_active(agent_id) or cursors[agent_id] >= len(steps):
                continue
            primitive, target = steps[cursors[agent_id]]
            cursors[agent_id] += 1
            immediate = engine.assign(agent_id, primitive, target)
            if immediate is not None:
                results.append(immediate)
                cursors[agent_id] = len(steps)
        if not engine.active_agents() or engine.env_step >= max_ticks:
            break
        for agent_id, outcome in engine.tick().items():
            results.append(outcome)
            if not outcome.ok:
                cursors[agent_id] = len(plans[agent_id])
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def main() -> int:
    _install_stubs()
    engine_module = _load_engine_module()
    MotionMode = engine_module.MotionMode
    P = FakePrimitiveSet

    def ok(message):
        print(f"  PASS {message}")

    print("test 1: asymmetric plans, concurrent -- no primitive-boundary alignment")
    engine, _ = make_engine(
        engine_module,
        MotionMode.CONCURRENT,
        {
            "agent_0": {"NAVIGATE_TO": 100, "GRASP": 50},
            "agent_1": {"NAVIGATE_TO": 30, "GRASP": 50},
        },
    )
    results = run_plans(
        engine,
        {
            "agent_0": [(P.NAVIGATE_TO, "apple_0"), (P.GRASP, "apple_0")],
            "agent_1": [(P.NAVIGATE_TO, "apple_1")],
        },
    )
    by = {(o.agent_id, o.primitive): o for o in results}
    nav1 = by[("agent_1", "NAVIGATE_TO")]
    nav0 = by[("agent_0", "NAVIGATE_TO")]
    grasp0 = by[("agent_0", "GRASP")]
    print(
        f"    agent_1 NAV ended @{nav1.ended_env_step}, agent_0 NAV ended @{nav0.ended_env_step}, "
        f"agent_0 GRASP {grasp0.started_env_step}->{grasp0.ended_env_step}, total={engine.env_step}"
    )
    assert all(o.ok for o in results), results
    assert nav1.ended_env_step == 30, nav1.ended_env_step
    assert nav0.ended_env_step == 100, nav0.ended_env_step
    # agent_0's GRASP starts one tick after its OWN navigate ended (the ending
    # tick still consumes an env.step) and, crucially, not after agent_1's.
    assert grasp0.started_env_step == nav0.ended_env_step + 1, grasp0.started_env_step
    assert engine.env_step == 152, engine.env_step
    assert engine.decision_count == 3, engine.decision_count
    ok("agent_1 finished @30 mid-flight; agent_0's GRASP started @101 at its own boundary; 3 primitives")

    print("test 2: exclusive -- execution serialized")
    engine, _ = make_engine(
        engine_module,
        MotionMode.EXCLUSIVE,
        {
            "agent_0": {"NAVIGATE_TO": 100, "GRASP": 50},
            "agent_1": {"NAVIGATE_TO": 30, "GRASP": 50},
        },
    )
    results = run_plans(
        engine,
        {"agent_0": [(P.NAVIGATE_TO, "apple_0")], "agent_1": [(P.NAVIGATE_TO, "apple_1")]},
    )
    print(f"    serialized total={engine.env_step}")
    assert all(o.ok for o in results)
    assert engine.env_step == 132, engine.env_step  # 100 + 30 + one ending tick each
    ok("serialized to 132 ticks (100 + 30 + 2 ending ticks); concurrent would be ~101")

    print("test 3: a failed primitive kills the rest of that agent's plan")
    engine, _ = make_engine(
        engine_module,
        MotionMode.CONCURRENT,
        {
            "agent_0": {"NAVIGATE_TO": ("fail", 20, "PLANNING_ERROR"), "GRASP": 50},
            "agent_1": {"NAVIGATE_TO": 40, "GRASP": 10},
        },
    )
    results = run_plans(
        engine,
        {
            "agent_0": [(P.NAVIGATE_TO, "apple_0"), (P.GRASP, "apple_0")],
            "agent_1": [(P.NAVIGATE_TO, "apple_1"), (P.GRASP, "apple_1")],
        },
    )
    assert sum(1 for o in results if o.agent_id == "agent_0") == 1
    assert sum(1 for o in results if o.agent_id == "agent_1") == 2
    failure = [o for o in results if not o.ok][0]
    assert failure.reason_code == "PLANNING", failure.reason_code
    assert failure.terminate_plan is True
    ok("agent_0 stopped after one failure; agent_1 completed both; reason_code=PLANNING, terminate_plan=True")

    print("test 4: invalid target fails without moving")
    engine, _ = make_engine(
        engine_module, MotionMode.CONCURRENT, {"agent_0": {"GRASP": 10}, "agent_1": {"GRASP": 10}}
    )
    outcome = engine.assign("agent_0", P.GRASP, "nonexistent")
    assert outcome is not None and outcome.reason_code == "INVALID_TARGET" and outcome.ticks == 0
    assert engine.env_step == 0 and not engine.has_active("agent_0")
    record = engine.get_action_records("agent_0")["agent_0"][-1]
    for key in ("status", "failure_reason", "primitive_action", "primitive_action_history", "outcome"):
        assert key in record, key
    ok("INVALID_TARGET at 0 ticks / 0 env steps, recorded with COOP2's record keys")

    print("test 5: abort + retract")
    engine, _ = make_engine(
        engine_module,
        MotionMode.CONCURRENT,
        {"agent_0": {"NAVIGATE_TO": 1000}, "agent_1": {"NAVIGATE_TO": 5}},
    )
    engine.assign("agent_0", P.NAVIGATE_TO, "apple_0")
    for _ in range(10):
        engine.tick()
    aborted = engine.abort("agent_0", retract=True)
    assert aborted.reason_code == "ABORTED" and aborted.ticks == 10, (aborted.reason_code, aborted.ticks)
    assert engine.has_active("agent_0") and engine.active_primitive("agent_0") == "RETRACT"
    engine.run_until_idle()
    assert not engine.has_active("agent_0")
    records = engine.get_action_records("agent_0")["agent_0"]
    assert len(records) == 1 and records[0]["reason_code"] == "ABORTED"
    ok("aborted at 10 ticks; RETRACT queued and drained; the cleanup run is not recorded")

    print("test 6: idle padding -- every robot keyed on every tick")
    engine, env = make_engine(
        engine_module, MotionMode.CONCURRENT, {"agent_0": {"GRASP": 3}, "agent_1": {"GRASP": 3}}
    )
    engine.assign("agent_0", P.GRASP, "apple_0")
    engine.tick()
    assert set(env.last_action) == {"agent_0", "agent_1"}, env.last_action
    assert env.last_action["agent_1"] == "idle:agent_1"
    assert env.last_action["agent_0"] == "act0"
    ok("action dict carries every robot; the non-acting one holds its joint configuration")

    print("test 7: macro_step convenience wrapper")
    engine, _ = make_engine(
        engine_module, MotionMode.CONCURRENT, {"agent_0": {"GRASP": 7}, "agent_1": {"GRASP": 4}}
    )
    outcomes = engine.macro_step({"agent_0": (P.GRASP, "apple_0"), "agent_1": (P.GRASP, "apple_1")})
    print(f"    macro_step total={engine.env_step}")
    assert set(outcomes) == {"agent_0", "agent_1"} and all(o.ok for o in outcomes.values())
    assert engine.env_step == 8, engine.env_step
    ok("macro_step aligns at the primitive boundary (8 ticks = slower primitive + ending tick)")

    print("test 8: held object surfaces in metadata")
    assert outcomes["agent_0"].metadata["held_object"] == "apple_0", outcomes["agent_0"].metadata
    assert engine.held_objects() == {"agent_0": "apple_0", "agent_1": "apple_1"}
    ok("held_object recorded per outcome and via held_objects()")

    print("test 9: last_advanced -- the simultaneity measurement itself")
    for mode, expect_overlap in ((MotionMode.CONCURRENT, True), (MotionMode.EXCLUSIVE, False)):
        engine, _ = make_engine(
            engine_module,
            mode,
            {"agent_0": {"NAVIGATE_TO": 50}, "agent_1": {"NAVIGATE_TO": 50}},
        )
        histogram: Dict[int, int] = {}
        engine.on_tick = lambda _step, e=engine, h=histogram: h.__setitem__(
            len(e.last_advanced), h.get(len(e.last_advanced), 0) + 1
        )
        run_plans(
            engine,
            {"agent_0": [(P.NAVIGATE_TO, "apple_0")], "agent_1": [(P.NAVIGATE_TO, "apple_1")]},
        )
        busy = sum(count for n, count in histogram.items() if n >= 1)
        overlap = sum(count for n, count in histogram.items() if n >= 2)
        ratio = overlap / busy if busy else 0.0
        print(f"    {mode}: histogram={dict(sorted(histogram.items()))} overlap_ratio={ratio:.2f}")
        if expect_overlap:
            # Both primitives are 50 ticks and start together, so every busy
            # tick should have both agents moving.
            assert ratio == 1.0, ratio
            assert histogram.get(2, 0) == 50, histogram
        else:
            assert ratio == 0.0, ratio
            assert histogram.get(2, 0) == 0, histogram
    ok("concurrent -> overlap_ratio 1.00 (50 ticks with both moving); exclusive -> 0.00")

    print("test 8b: a robot with no arm holds nothing, rather than raising")
    # A heterogeneous team can contain a carrier base -- it navigates and does
    # nothing else -- and `_ag_obj_in_hand` does not exist on a robot that is
    # not a manipulator. Asking anyway took a whole episode down with an
    # AttributeError at the end of that robot's first navigate.
    class Armless:
        name = "carrier"
        is_manipulation = False

    class Exploding:
        def _get_obj_in_hand(self):
            raise AssertionError("a robot with no arm must not be asked")

    engine.robots_by_id["carrier"] = Armless()
    engine.controllers["carrier"] = Exploding()
    assert engine.held_object("carrier") is None
    ok("an armless robot answers None without reaching for a hand it lacks")

    print("\ntest: agent ids are matched to robots by name, not by position")
    # scene.robots is sorted by name; a layout is not. Before the fix the zip
    # gave ridgeback_1 the robot called drone_1, and the first env.step died
    # on the action dimension.
    layout_order = ["ridgeback_1", "jackal_1", "drone_1"]
    sorted_robots = [FakeRobot(n) for n in sorted(layout_order)]  # drone_1, jackal_1, ridgeback_1
    env = FakeEnv(sorted_robots, {})
    engine = engine_module.MultiAgentPrimitiveEngine(
        env, agent_ids=layout_order, progress_every=0, verbose=False,
        controllers={a: FakeController([]) for a in layout_order})
    for agent_id in layout_order:
        assert engine.robots_by_id[agent_id].name == agent_id, (agent_id, engine.robots_by_id[agent_id].name)
    assert [r.name for r in engine.robots] == layout_order
    # Ids that are not robot names keep the positional contract.
    env2 = FakeEnv([FakeRobot("robot_a"), FakeRobot("robot_b")], {})
    engine2 = engine_module.MultiAgentPrimitiveEngine(
        env2, agent_ids=["agent_0", "agent_1"], progress_every=0, verbose=False,
        controllers={"agent_0": FakeController([]), "agent_1": FakeController([])})
    assert engine2.robots_by_id["agent_1"].name == "robot_b"
    print("  ok: by name when names match, positional otherwise")

    print("\ntest: a room name is a NAVIGATE_TO target, dispatched to navigate_to_room")
    class RoomSegMap:
        room_ins_name_to_ins_id = {"kitchen_0": 1, "hall_0": 2}
    room_env = FakeEnv([FakeRobot("agent_0")], {"apple_0": FakeNamed("apple_0")})
    room_env.scene.seg_map = RoomSegMap()
    room_ctrl = FakeController({"NAVIGATE_TO": 3})
    room_engine = engine_module.MultiAgentPrimitiveEngine(
        room_env, agent_ids=["agent_0"], progress_every=0, verbose=False, controllers={"agent_0": room_ctrl})
    assert room_engine.is_room("kitchen_0") and not room_engine.is_room("apple_0")
    assert room_engine.assign("agent_0", P.NAVIGATE_TO, "kitchen_0") is None
    room_engine.tick()
    assert room_ctrl.rooms_visited == ["kitchen_0"], room_ctrl.rooms_visited
    assert room_env.last_action["agent_0"] == "room:kitchen_0", room_env.last_action
    # An object target still goes through the registry; an unknown name is still INVALID_TARGET.
    room_engine.tick()  # StopIteration -> the room run finishes
    assert not room_engine.has_active("agent_0")
    bad = room_engine.assign("agent_0", P.NAVIGATE_TO, "attic_0")
    assert bad is not None and bad.reason_code == "INVALID_TARGET", bad
    print("  ok: kitchen_0 -> navigate_to_room; attic_0 -> INVALID_TARGET")

    print("\ntest: a robot is a NAVIGATE_TO target, and still not a GRASP target")
    mate = FakeRobotBase(); mate.name = "jackal_9"
    nav_env = FakeEnv([FakeRobot("agent_0")], {"apple_0": FakeNamed("apple_0"), "jackal_9": mate})
    nav_ctrl = FakeController({"NAVIGATE_TO": 2, "GRASP": 2})
    nav_engine = engine_module.MultiAgentPrimitiveEngine(
        nav_env, agent_ids=["agent_0"], progress_every=0, verbose=False, controllers={"agent_0": nav_ctrl})
    assert nav_engine.assign("agent_0", P.NAVIGATE_TO, "jackal_9") is None, "driving to a teammate is allowed"
    while nav_engine.has_active("agent_0"):
        nav_engine.tick()
    assert nav_ctrl.robots_visited == ["jackal_9"], "dispatched to navigate_to_robot, not apply_ref"
    bad = nav_engine.assign("agent_0", P.GRASP, "jackal_9")
    assert bad is not None and bad.reason_code == "INVALID_TARGET" and "is a robot" in bad.failure_reason, bad
    print("  ok: navigate_to(robot) accepted; grasp(robot) refused")

    print("\ntest: a failure an agent can read -- whole, and in the ids it was shown")
    readable = engine_module.readable_failure
    ids = {"notebook_154": "notebook.n.01_1", "armchair_qplklw_2": "armchair.n.01_1"}

    # The metadata dict duplicates what the outcome already carries, and it
    # was the half the prompt's 200-char cut was eating, leaving lines ending
    # on "{'target object': 'jackal_3', 'distance':" (user, 2026-09-14).
    got = readable(
        "PRE_CONDITION_ERROR: You are 2.77 m from jackal_3, too far to load onto it "
        "(you must be within 1.74 m). Navigate to it first.. Additional info: "
        "{'target object': 'jackal_3', 'distance': 2.77, 'reason_code': 'TOO_FAR'}")
    assert got.endswith("Navigate to it first."), got
    assert "Additional info" not in got, got
    assert "Additional info: None" not in readable("PLANNING_ERROR: x.. Additional info: None")

    # One attempt: the group wrapper is 52 characters that say nothing.
    one = readable("An error occurred during each attempt of this action.\n\n"
                   "Attempt 0: PRE_CONDITION_ERROR: You have no arm.")
    assert one == "PRE_CONDITION_ERROR: You have no arm.", one
    # Two: which attempt failed how is the whole point, so it is kept.
    two = readable("An error occurred during each attempt of this action.\n\n"
                   "Attempt 0: A.\n\nAttempt 1: B.")
    assert "Attempt 0" in two and "Attempt 1" in two, two

    # A primitive raises with the scene name because that is all it has; the
    # agent is told to use only the ids in its listing and never invent one,
    # so "Cannot reach armchair_qplklw_2" is a failure it cannot act on.
    named = readable(
        "PLANNING_ERROR: Cannot reach armchair_qplklw_2: notebook_154 rests on it.. "
        "Additional info: {'object': 'armchair_qplklw_2'}", ids.get)
    assert named == "PLANNING_ERROR: Cannot reach armchair.n.01_1: notebook.n.01_1 rests on it.", named
    # A name the world does not know is left alone, and so is a robot (a robot
    # is its own id, so the map returns it unchanged).
    assert "jackal_3" in readable("PRE_CONDITION_ERROR: 2.8 m from jackal_3.", ids.get)
    assert "widget_7" in readable("PRE_CONDITION_ERROR: widget_7 is stuck.", ids.get)
    # No resolver -- the engine before its world exists -- still cleans up.
    assert readable("A.. Additional info: {}", None) == "A."
    print("  ok: dict and single-attempt wrapper dropped, scene names resolved, unknowns kept")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
