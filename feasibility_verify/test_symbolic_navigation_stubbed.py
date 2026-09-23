"""CPU-only regression test for the cuRobo-free symbolic NAVIGATE_TO.

Stubs OmniGibson (and torch) entirely, so it runs anywhere -- no Isaac, no GPU,
no scene load. It checks the sampler's geometry and, most importantly, the two
upstream defects this override exists to work around:

    * the inherited sampler dereferences ``self._motion_generator``, which the
      symbolic constructor never builds (``AttributeError: 'NoneType' object
      has no attribute 'update_obstacles'``);
    * ``_navigate_to_obj`` forwards ``skip_obstacle_update`` into the symbolic
      ``_navigate_to_pose(self, pose_2d)``, which does not accept it
      (``TypeError``).

Run:
    python feasibility_verify/test_symbolic_navigation_stubbed.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import sys
import types


# ---------------------------------------------------------------------------
# Stubs: a minimal torch, then the OmniGibson surface the module imports
# ---------------------------------------------------------------------------


class FakeVector(list):
    """Enough of a torch tensor for the sampler: indexing, slicing, item()."""

    def __getitem__(self, index):
        value = list.__getitem__(self, index)
        return FakeVector(value) if isinstance(index, slice) else value

    def item(self):
        assert len(self) == 1, self
        return self[0]

    def __float__(self):
        assert len(self) == 1, self
        return float(self[0])


def _install_torch_stub():
    module = types.ModuleType("torch")
    module.tensor = lambda data, dtype=None: FakeVector(float(x) for x in data)
    module.rand = lambda n: FakeVector(random.random() for _ in range(n))
    module.mean = lambda vector: FakeVector([sum(vector) / len(vector)])
    module.norm = lambda vector: FakeVector([math.sqrt(sum(float(v) ** 2 for v in vector))])
    # Identity on purpose: FakeVector(x) iterates x, and FakeTravMap answers
    # __getitem__ for any index, so the legacy iteration protocol would spin
    # forever building an infinite list.
    module.clone = lambda x: x
    module.float32 = "float32"
    module.Tensor = FakeVector
    sys.modules["torch"] = module


class FakeActionPrimitiveError(ValueError):
    class Reason:
        PLANNING_ERROR = type("_Member", (), {"name": "PLANNING_ERROR"})()

    def __init__(self, reason, message, metadata=None):
        self.reason = reason
        self.metadata = metadata or {}
        super().__init__(f"{reason.name}: {message}")


class FakeSymbolicPrimitives:
    """Stands in for SymbolicSemanticActionPrimitives.

    Note the ``_navigate_to_pose`` signature: it takes **no**
    ``skip_obstacle_update``, faithfully reproducing the upstream override that
    the inherited ``_navigate_to_obj`` breaks on.
    """

    def __init__(self, env, robot, **kwargs):
        self.env = env
        self.robot = robot
        self.arm = "left"
        self.teleported_to = None
        # The whole point: the symbolic constructor skips cuRobo.
        self._motion_generator = None

    def _navigate_to_pose(self, pose_2d):
        self.teleported_to = pose_2d
        for _ in range(3):
            yield "settle"

    def _get_robot_pose_from_2d_pose(self, pose_2d):
        # Upstream's non-holonomic branch: z forced to 0.0.
        return FakeVector([pose_2d[0], pose_2d[1], 0.0]), "upstream-orn"


def _install_omnigibson_stubs():
    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    stub("omnigibson")
    stub("omnigibson.action_primitives")
    stub(
        "omnigibson.action_primitives.action_primitive_set_base",
        ActionPrimitiveError=FakeActionPrimitiveError,
    )
    stub(
        "omnigibson.action_primitives.symbolic_semantic_action_primitives",
        SymbolicSemanticActionPrimitives=FakeSymbolicPrimitives,
    )


def _register_real(*names):
    """Put real ``coop2.behavior_env.<name>`` modules in sys.modules.

    The module under test imports these by their package path. Both are pure
    stdlib -- ``geometry_cache`` is a dict of world AABBs valid for one tick,
    ``carrier`` is a handful of getattr predicates -- so the real files are
    loaded rather than stubbed; only the package namespace around them is
    faked, the way this file fakes omnigibson.
    """
    import types

    for package_name in ("coop2", "coop2.behavior_env"):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
    for stem in names:
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


def _load_module():
    _register_real("geometry_cache", "carrier")
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coop2",
        "behavior_env",
        "symbolic_navigation.py",
    )
    spec = importlib.util.spec_from_file_location("coop2_symbolic_navigation_stubbed", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSegMap:
    """'kitchen_0' is the disc of radius 3 around the origin; else 'hall_0'."""

    room_ins_name_to_ins_id = {"kitchen_0": 1, "hall_0": 2}

    def get_room_instance_by_point(self, xy):
        radius = math.hypot(float(xy[0]), float(xy[1]))
        return "kitchen_0" if radius <= 3.0 else "hall_0"

    def get_random_point_by_room_instance(self, room):
        import random
        if room not in self.room_ins_name_to_ins_id:
            return None, None
        r = random.uniform(0.5, 2.5) if room == "kitchen_0" else random.uniform(4.0, 6.0)
        a = random.uniform(-math.pi, math.pi)
        return 0, FakeVector([r * math.cos(a), r * math.sin(a), 0.0])


class FakeTravMap:
    """Everything inside |x|,|y| < 4 m is standable; outside is not.

    world_to_map/_erode_trav_map/floor_map mirror the real TraversableMap
    surface the sampler touches.
    """

    def __init__(self, half_extent=4.0):
        self.half_extent = half_extent
        self.floor_map = [self]
        self.shape = (100, 100)

    def _erode_trav_map(self, floor_map, robot=None):
        return self

    def world_to_map(self, xy):
        # Centre the 100x100 grid on the origin at 0.1 m per cell.
        return FakeVector([float(xy[0]) / 0.1 + 50, float(xy[1]) / 0.1 + 50])

    def __getitem__(self, row):
        return _TravRow(self, row)


class _TravRow:
    def __init__(self, trav_map, row):
        self.trav_map = trav_map
        self.row = row

    def __getitem__(self, col):
        x = (self.row - 50) * 0.1
        y = (col - 50) * 0.1
        inside = abs(x) < self.trav_map.half_extent and abs(y) < self.trav_map.half_extent
        return 255 if inside else 0


class FakeScene:
    def __init__(self, seg_map=None, trav_map=None):
        self.seg_map = seg_map
        self.trav_map = trav_map
        self.robots = []
        # name -> object, as OmniGibson's is (membership by value, not by name).
        self.fixed_objects = {}


class FakeRobot:
    def __init__(self, scene, name="agent_0", position=(0.0, 0.0, 0.0)):
        self.scene = scene
        self.name = name
        self.position = FakeVector(position)
        self.arm_workspace_range = {"left": FakeVector([0.5, 1.0])}
        # norm([0.8, 0.8]) / 2 == 0.5657 -> a tidy circumscribed radius.
        self.reset_joint_pos_aabb_extent = FakeVector([0.8, 0.8, 1.2])
        scene.robots.append(self)

    def get_position_orientation(self):
        return self.position, None


class FakeObject:
    def __init__(self, name, position, in_rooms=None, aabb_extent=(0.05, 0.05, 0.05)):
        self.name = name
        self._position = FakeVector(position)
        self.in_rooms = list(in_rooms or [])
        self.aabb_extent = FakeVector(aabb_extent)

    def get_position_orientation(self):
        return self._position, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def main() -> int:
    _install_torch_stub()
    _install_omnigibson_stubs()
    module = _load_module()
    Navigable = module.NavigableSymbolicActionPrimitives

    def ok(message):
        print(f"  PASS {message}")

    scene = FakeScene(FakeSegMap())
    robot = FakeRobot(scene)
    controller = Navigable(None, robot, require_traversable=False)
    apple = FakeObject("apple_0", [1.0, 0.5, 0.4])  # no in_rooms -> point query

    print("test 1: samples inside the target's room, in the band around its footprint")
    # Measured from the footprint, not from the centre: a target that is not
    # walked on is approached at its edge (2026-09-14), so the bound that holds
    # is the offset out of `_edge_candidate`. The ring's `hi` still bounds it
    # from the centre, because edge distance never exceeds the half-diagonal --
    # which is what keeps every band pose inside the reach gate.
    lo, hi = controller.sampling_range_for(apple)
    band_lo = controller.body_offset() + controller._nav_clearance_margin
    band_hi = band_lo + controller._nav_reach
    for _ in range(200):
        pose = controller._sample_pose_near_object(apple)
        assert pose is not None
        edge = controller.edge_distance_to(apple, pose[:2])
        assert band_lo - 1e-6 <= edge <= band_hi + 1e-6, (edge, band_lo, band_hi)
        assert math.hypot(float(pose[0]) - 1.0, float(pose[1]) - 0.5) <= hi + 1e-6
        assert scene.seg_map.get_room_instance_by_point(pose[:2]) == "kitchen_0"
    ok(f"200 samples in kitchen_0, all {band_lo:.2f}-{band_hi:.2f} m off the footprint")

    print("test 1b: the lower bound keeps the robot off the target")
    # The old range started at 0.0, so the robot could stand ON a floor object.
    # Measured consequence: symbolic grasp then teleported the apple into a
    # robot standing on it and the settle shook it loose (POST_CONDITION).
    # The offset is the chassis half-width -- a base pose faces its target, so
    # what points at the object is a face, not the 45-degree diagonal.
    assert band_lo >= controller.body_offset(), (band_lo, controller.body_offset())
    assert lo >= controller.body_offset(), (lo, controller.body_offset())
    table = FakeObject("table_0", [1.0, 0.5, 0.4], aabb_extent=(1.6, 0.9, 0.7))
    table_lo, table_hi = controller.sampling_range_for(table)
    assert table_lo > lo, (table_lo, lo)
    assert table_hi > hi, (table_hi, hi)
    ok(f"apple lo={lo:.2f} m, table lo={table_lo:.2f} m -- clearance scales with the object")

    print("test 2: yaw faces the target (upstream formula: yaw + pi - mean(workspace))")
    # The band aims at the footprint point it was pushed out from, not at the
    # centre, so the two differ by however much the object subtends from there
    # -- 4 degrees for a 5 cm apple half a metre away. What must hold is that
    # the robot is facing the object at all.
    for _ in range(50):
        pose = controller._sample_pose_near_object(apple)
        dx, dy = float(pose[0]) - 1.0, float(pose[1]) - 0.5
        expected = math.atan2(dy, dx) + math.pi - 0.75
        delta = ((float(pose[2]) - expected + math.pi) % (2 * math.pi)) - math.pi
        subtends = math.atan2(controller.clearance_for(apple) - controller.body_offset()
                              - controller._nav_clearance_margin, math.hypot(dx, dy))
        assert abs(delta) <= subtends + 1e-4, (float(pose[2]), expected, subtends)
    ok("yaw == sampling_yaw + pi - mean(arm_workspace_range)")

    print("test 3: a FIXED object's in_rooms wins over the live point query")
    far = FakeObject("far_shelf", [10.0, 0.0, 0.4], in_rooms=["hall_0"])
    scene.fixed_objects[far.name] = far
    pose = controller._sample_pose_near_object(far)
    assert pose is not None
    assert scene.seg_map.get_room_instance_by_point(pose[:2]) == "hall_0"
    ok("fixed: in_rooms=['hall_0'] respected")

    print("test 3b: a MOVABLE object is sampled where it stands, not where its label says")
    # The S1 LL failure of 2026-09-13: a die labelled childs_room_0 at load,
    # carried to a bed in bedroom_0; every candidate around it was rejected
    # {'room': 200, 'trav': 0, 'robots': 0} because the sampler filtered for
    # the label. Here: labelled hall_0, standing in the kitchen.
    carried = FakeObject("carried_die", [1.0, 0.5, 0.4], in_rooms=["hall_0"])
    assert carried.name not in scene.fixed_objects
    pose = controller._sample_pose_near_object(carried)
    assert pose is not None, "a movable object in a reachable room must be reachable"
    assert scene.seg_map.get_room_instance_by_point(pose[:2]) == "kitchen_0", pose
    ok("movable: stale in_rooms=['hall_0'] ignored, sampled in kitchen_0 where it is")

    print("test 4: unreachable room -> None -> PLANNING_ERROR, not a crash")
    # Fixed on purpose: a fixed object's label is trusted, so a shelf labelled
    # kitchen_0 that stands at x=10 (hall_0) has no candidate in its room. (A
    # movable object in that state is the stale-label case test 3b covers, and
    # is now reachable.)
    impossible = FakeObject("ghost", [10.0, 0.0, 0.4], in_rooms=["kitchen_0"])
    scene.fixed_objects[impossible.name] = impossible
    assert controller._sample_pose_near_object(impossible) is None
    generator = controller._navigate_to_obj(impossible)
    try:
        next(generator)
        raise AssertionError("expected PLANNING_ERROR")
    except FakeActionPrimitiveError as error:
        assert error.reason.name == "PLANNING_ERROR", error.reason
        assert error.metadata["object"] == "ghost"
    ok("None from the sampler surfaces as PLANNING_ERROR with metadata")

    print("test 5: _navigate_to_obj must NOT forward skip_obstacle_update")
    controller.teleported_to = None
    steps = list(controller._navigate_to_obj(apple))
    assert steps == ["settle"] * 3, steps
    assert controller.teleported_to is not None
    ok("teleport + settle ran; no TypeError from the symbolic _navigate_to_pose signature")

    print("test 6: no cuRobo motion generator is ever touched")
    assert controller._motion_generator is None
    ok("_motion_generator stayed None throughout")

    print("test 7: eef_pose overrides the object position as the sampling centre")
    eef_pose = (FakeVector([-2.0, 0.0, 0.9]), None)
    pose = controller._sample_pose_near_object(apple, eef_pose=eef_pose)
    distance = math.hypot(float(pose[0]) + 2.0, float(pose[1]))
    assert distance <= controller.sampling_range_for(apple)[1] + 1e-6, distance
    ok("sampled around eef_pose, not around the object")

    print("test 8: plain Scene without a seg map -> accepts the first candidate")
    plain = Navigable(None, FakeRobot(FakeScene(seg_map=None)), require_traversable=False)
    assert plain._sample_pose_near_object(apple) is not None
    ok("works without a segmentation map")

    print("test 9: require_same_room=False bypasses the room filter")
    loose = Navigable(None, robot, require_same_room=False, require_traversable=False)
    assert loose._sample_pose_near_object(impossible) is not None
    ok("require_same_room=False returns a pose even for an out-of-room target")

    print("test 10: untraversable poses are rejected, not returned")
    # The whole reason this filter exists: without it the sampler returned the
    # first candidate whatever it was, and on Rs_int 96% of those put the robot
    # somewhere it does not fit -- it teleported into a wall and toppled.
    trav_scene = FakeScene(FakeSegMap(), trav_map=FakeTravMap(half_extent=4.0))
    trav_robot = FakeRobot(trav_scene)
    strict = Navigable(None, trav_robot, require_same_room=False)
    for _ in range(50):
        pose = strict._sample_pose_near_object(apple)
        assert pose is not None
        assert strict._is_traversable(pose[:2]), pose
    # A target sitting outside the traversable region has no standable pose at
    # all, and that must surface as None -> PLANNING_ERROR rather than a bad pose.
    marooned = FakeObject("marooned", [40.0, 40.0, 0.4])
    assert strict._sample_pose_near_object(marooned) is None
    ok("50 samples all traversable; a marooned target yields None")

    print("test 11: candidates too close to another robot are rejected")
    crowd_scene = FakeScene(FakeSegMap())
    mover = FakeRobot(crowd_scene, name="agent_0")
    parked = FakeRobot(crowd_scene, name="agent_1", position=(1.0, 0.5, 0.0))
    crowded = Navigable(None, mover, require_same_room=False, require_traversable=False)
    separation = crowded.robot_separation
    assert abs(separation - 2 * crowded.robot_radius) < 1e-9, separation
    for _ in range(50):
        pose = crowded._sample_pose_near_object(apple)
        if pose is None:
            continue
        gap = math.hypot(float(pose[0]) - 1.0, float(pose[1]) - 0.5)
        assert gap >= separation - 1e-6, (gap, separation)
    ok(f"no sample within {separation:.2f} m of agent_1 (2 x robot radius)")

    print("test 12: concurrent samplers must not all pick the same spot")
    # The TOCTOU race this registry exists for: every agent samples while every
    # other robot is still at its start pose metres away, so a
    # current-position-only check passes for all of them and they teleport into
    # each other. Measured at 0.64 m apart against a 1.24 m requirement, which
    # let one robot's assisted grasp latch onto another robot.
    race_scene = FakeScene(FakeSegMap())
    movers = [FakeRobot(race_scene, name=f"agent_{i}", position=(5.0 * i, 5.0 * i, 0.0)) for i in range(3)]
    registry = module.DestinationRegistry()
    controllers = [
        Navigable(None, robot, require_same_room=False, require_traversable=False, destinations=registry)
        for robot in movers
    ]
    contested = FakeObject("apple_0", [0.0, 0.0, 0.4])
    poses = []
    for controller in controllers:
        pose = controller._sample_pose_near_object(contested)  # nobody has moved yet
        assert pose is not None
        poses.append((float(pose[0]), float(pose[1])))
    separation = controllers[0].robot_separation
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            gap = math.hypot(poses[i][0] - poses[j][0], poses[i][1] - poses[j][1])
            assert gap >= separation - 1e-6, f"agents {i},{j} would overlap at {gap:.2f} m"
    assert len(registry) == 3
    ok(f"3 concurrent samples around one object, all >= {separation:.2f} m apart")

    print("test 13: without the registry the race reproduces")
    # Guards the guard: if this stops finding overlaps, test 12 has stopped
    # testing anything. Measured over many trials rather than one, because a
    # single random draw of three poses around an object sometimes happens to
    # be well separated -- a one-shot version of this assertion is flaky and
    # would eventually be "fixed" by deleting it.
    def closest_pair(use_registry):
        scene = FakeScene(FakeSegMap())
        robots = [FakeRobot(scene, name=f"r{i}", position=(5.0 * i, 5.0 * i, 0.0)) for i in range(3)]
        shared = module.DestinationRegistry() if use_registry else None
        picked = []
        for robot in robots:
            controller = Navigable(
                None, robot, require_same_room=False, require_traversable=False, destinations=shared
            )
            pose = controller._sample_pose_near_object(contested)
            if pose is None:
                return None
            picked.append((float(pose[0]), float(pose[1])))
        return min(
            math.hypot(picked[i][0] - picked[j][0], picked[i][1] - picked[j][1])
            for i in range(3)
            for j in range(i + 1, 3)
        )

    trials = 200
    unguarded = [closest_pair(False) for _ in range(trials)]
    guarded = [closest_pair(True) for _ in range(trials)]
    overlaps_without = sum(1 for d in unguarded if d is not None and d < separation)
    overlaps_with = sum(1 for d in guarded if d is not None and d < separation)
    assert overlaps_without > trials * 0.1, f"only {overlaps_without}/{trials} overlapped; race not reproduced"
    assert overlaps_with == 0, f"{overlaps_with}/{trials} overlapped despite reservations"
    ok(
        f"no registry: {overlaps_without}/{trials} trials overlap; "
        f"with registry: {overlaps_with}/{trials}"
    )

    print("test 14: separation is just-touching, not a planning margin")
    # Symbolic navigation teleports, so there is no path to keep clear -- only
    # bodies to keep from overlapping. Anything larger needlessly shrinks the
    # usable floor, which matters at N=9.
    assert crowded.robot_separation == 2 * crowded.robot_radius
    tight = Navigable(None, mover, robot_separation=0.1, require_traversable=False)
    assert tight.robot_separation == 0.1
    ok("default = 2 x radius, and it is overridable")

    print("test 8: a teleport in the plane keeps the altitude the body has now")
    # Upstream returns the z *joint* (measured from the root anchor at 0.05 m)
    # as a *world* z, so every navigate re-lands a hovering drone 5 cm lower.
    # Measured 1.200 -> 1.150 -> 1.100 -> 1.050 over three hops.
    fake_T = types.ModuleType("omnigibson.utils.transform_utils")
    seen = {}
    # Overwrite, not setdefault: this block calls the method more than once and
    # a first-write-wins spy reports the first call's eulers for every later one.
    fake_T.euler_intrinsic2mat = lambda e: seen.__setitem__("eulers", list(e))
    fake_T.mat2quat = lambda m: "quat-from-eulers"
    sys.modules["omnigibson.utils.transform_utils"] = fake_T
    drone = FakeRobot(scene, name="drone", position=(2.44, 7.23, 1.2))
    drone.is_holonomic_base = True
    drone.base_idx = [0, 1, 2, 3, 4, 5]
    drone.get_joint_positions = lambda: FakeVector([2.39, 7.18, 1.15, 0.0, 0.0, 0.3])
    flyer = Navigable(None, drone, require_traversable=False)
    pos, orn = flyer._get_robot_pose_from_2d_pose(FakeVector([3.0, 4.0, 0.5]))
    assert [float(pos[0]), float(pos[1])] == [3.0, 4.0], list(pos)
    assert abs(float(pos[2]) - 1.2) < 1e-9, f"z came back as {float(pos[2])}, wanted the body's 1.2 not the joint's 1.15"
    assert seen["eulers"] == [0.0, 0.0, 0.5] and orn == "quat-from-eulers", (seen, orn)
    ok("holonomic: z is the body's world z (1.2), yaw comes from the command")
    # ...unless the robot keeps its heading. A drone's suction mount is under
    # its belly, so nothing it does is gated on facing, and the sampler's
    # face-the-target yaw only makes its recorded frames snap to an unrelated
    # angle at the end of every hop -- which is what got reported as the drone
    # spinning. The yaw it already has (the rz joint, 0.3) is used instead of
    # the commanded 0.5; x, y and z are unaffected.
    drone.is_drone = True
    drone.get_joint_positions = lambda: FakeVector([2.39, 7.18, 1.15, 0.11, -0.07, 0.3])
    pos, orn = flyer._get_robot_pose_from_2d_pose(FakeVector([3.0, 4.0, 0.5]))
    assert [float(pos[0]), float(pos[1])] == [3.0, 4.0], list(pos)
    assert abs(float(pos[2]) - 1.2) < 1e-9, float(pos[2])
    # rz kept (0.3, not the commanded 0.5) AND rx/ry levelled: the tilt joints
    # are free on this base, so a body carrying a die arrives tipped, and the
    # world yaw is the composition of all three. Pinning rz alone left the
    # measured yaw walking 0 -> -8 -> -27 -> -19 deg over one route's hops;
    # levelling too holds it inside +-5.7 deg for the whole route.
    assert seen["eulers"] == [0.0, 0.0, 0.3], seen
    ok("a drone arrives level and on the heading it had, not facing its target")
    # And a robot that is not a drone is untouched by the same call, so the
    # arms still arrive facing what they came for.
    arm = FakeRobot(scene, name="ridgeback", position=(1.0, 1.0, 0.05))
    arm.is_holonomic_base = True
    arm.base_idx = [0, 1, 2, 3, 4, 5]
    arm.get_joint_positions = lambda: FakeVector([1.0, 1.0, 0.05, 0.0, 0.0, 0.3])
    Navigable(None, arm, require_traversable=False)._get_robot_pose_from_2d_pose(
        FakeVector([3.0, 4.0, 0.5]))
    assert seen["eulers"] == [0.0, 0.0, 0.5], seen
    ok("an arm still arrives yawed to face its target")
    wheeled = FakeRobot(scene, name="cart", position=(0.0, 0.0, 0.05))
    wheeled.is_holonomic_base = False
    pos, orn = Navigable(None, wheeled, require_traversable=False)._get_robot_pose_from_2d_pose(FakeVector([1.0, 1.0, 0.0]))
    assert float(pos[2]) == 0.0 and orn == "upstream-orn", (list(pos), orn)
    ok("non-holonomic: falls through to upstream unchanged")

    print("test: an object on furniture is approached by way of the furniture")
    shelf_scene = FakeScene(FakeSegMap())
    shelf_scene.objects = []
    shelf_robot = FakeRobot(shelf_scene, name="agent_0")
    shelf_ctrl = Navigable(None, shelf_robot, require_traversable=False)
    bookcase = FakeObject("bookcase_0", [2.0, 0.0, 0.5], aabb_extent=(1.0, 0.4, 1.0))
    die = FakeObject("die_0", [2.1, 0.05, 1.02], aabb_extent=(0.04, 0.04, 0.04))   # on its top (1.0)
    floor = FakeObject("floor_0", [0.0, 0.0, 0.0], aabb_extent=(8.0, 8.0, 0.02)); floor.category = "floors"
    apple_on_floor = FakeObject("apple_0", [-1.0, 0.0, 0.05], aabb_extent=(0.08, 0.08, 0.08))
    shelf_scene.objects += [bookcase, die, floor, apple_on_floor, shelf_robot]
    assert shelf_ctrl.support_of(die) is bookcase
    assert shelf_ctrl.support_of(apple_on_floor) is None, "a floor is walked on, not reached across"
    assert shelf_ctrl.support_of(bookcase) is None
    assert shelf_ctrl.approach_anchor(die) is bookcase, "narrow furniture: stand at it and reach across"
    # A bed: the AABB top is the headboard, 0.5 m above the mattress the die
    # sits on. It IS the support (the die rests within its span, above its
    # base) but it is too wide to reach across, so the die is approached where
    # it is -- a ring around a 2.1 m bed's centre lands in the walls.
    bed = FakeObject("bed_0", [5.0, 0.0, 0.55], aabb_extent=(2.1, 1.7, 1.1))       # z-span 0.0 .. 1.1
    die_on_bed = FakeObject("die_5", [4.4, 0.5, 0.57], aabb_extent=(0.04, 0.04, 0.04))  # bottom 0.55
    shelf_scene.objects += [bed, die_on_bed]
    assert shelf_ctrl.support_of(die_on_bed) is bed, "a die on the mattress rests on the bed, headboard or not"
    assert not shelf_ctrl.reach_across(bed) and shelf_ctrl.reach_across(bookcase)
    assert shelf_ctrl.approach_anchor(die_on_bed) is bed, "a wide support is still where you go -- to its edge"
    for _ in range(40):
        c = shelf_ctrl._edge_candidate(bed, prefer_xy=[4.4, 0.5])
        gap = shelf_ctrl.edge_distance_to(bed, c[:2])
        assert 0.0 < gap <= shelf_ctrl.robot_radius + shelf_ctrl._nav_clearance_margin + shelf_ctrl._nav_reach + 1e-6, gap
    chosen = shelf_ctrl._sample_pose_near_object(bed, prefer_xy=[4.4, 0.5])
    assert chosen is not None
    assert shelf_ctrl.edge_distance_to(bed, chosen[:2]) <= shelf_ctrl.robot_radius + shelf_ctrl._nav_clearance_margin + shelf_ctrl._nav_reach + 1e-6
    assert float(chosen[0]) < 5.0 or float(chosen[1]) > 0.3, f"should stand on the die's side, got {list(chosen)}"
    assert shelf_ctrl.edge_distance_to(bed, [5.0, 0.0]) == 0.0
    assert abs(shelf_ctrl.edge_distance_to(bed, [7.05, 0.0]) - 1.0) < 1e-6
    assert shelf_ctrl.support_of(apple_on_floor) is None and shelf_ctrl.approach_anchor(apple_on_floor) is apple_on_floor
    # A die on the floor under a table is not resting ON the table.
    table = FakeObject("table_0", [7.0, 0.0, 0.4], aabb_extent=(1.2, 0.6, 0.8))        # z-span 0.0 .. 0.8
    die_under_table = FakeObject("die_7", [7.0, 0.0, 0.02], aabb_extent=(0.04, 0.04, 0.04))
    shelf_scene.objects += [table, die_under_table]
    assert shelf_ctrl.support_of(die_under_table) is None, "at the table's base, not on it"
    # A robot rests on nothing; a die under a hovering drone is not its support.
    hover = FakeRobot(shelf_scene, name="drone_9", position=(2.1, 0.05, 1.25))
    hover.reset_joint_pos_aabb_extent = FakeVector([0.24, 0.24, 0.1])
    die_under = FakeObject("die_9", [2.1, 0.05, 1.30], aabb_extent=(0.04, 0.04, 0.04))
    shelf_scene.objects += [die_under]
    assert shelf_ctrl.support_of(hover) is None
    assert shelf_ctrl.support_of(die_under) is None, "the die's footprint holds nothing bigger than itself"
    # The sampled spot is around the bookcase, not around the die sitting on
    # it. The bound that holds for BOTH shapes the sampler alternates between
    # is the ring's top, which is also what the interaction gate allows -- a
    # band pose is never further than that, since edge distance never exceeds
    # the half-diagonal. Pinning the band bound instead makes this flaky: a
    # ring draw is legitimately further from the footprint.
    _, b_hi = shelf_ctrl.sampling_range_for(bookcase)
    _, die_hi = shelf_ctrl.sampling_range_for(die)
    assert die_hi < b_hi, (die_hi, b_hi)
    seen_far = False
    for _ in range(50):
        pose = shelf_ctrl._sample_pose_near_object(bookcase)
        centre = math.hypot(float(pose[0]) - 2.0, float(pose[1]))
        assert centre <= b_hi + 1e-6, list(pose)
        seen_far = seen_far or centre > die_hi
    assert seen_far, "sampled only within the die's range, so not around the bookcase"
    # The fake _navigate_to_pose does not move the robot, so watch what the
    # sampler is asked for: navigate_to(die) must sample around the bookcase.
    asked = []
    real_sampler = shelf_ctrl._sample_pose_near_object
    shelf_ctrl._sample_pose_near_object = lambda obj, **kw: (asked.append(obj.name), real_sampler(obj, **kw))[1]
    list(shelf_ctrl._navigate_to_obj(die))
    assert asked == ["bookcase_0"], asked
    list(shelf_ctrl._navigate_to_obj(apple_on_floor))
    assert asked[-1] == "apple_0", asked
    ok("die on a bookcase: support_of finds the bookcase, navigate_to(die) samples around the bookcase")

    print("test: a hovering drone does not take a ground robot's standing spot, and vice versa")
    layer_scene = FakeScene(FakeSegMap())
    ground = FakeRobot(layer_scene, name="ridgeback_1", position=(0.0, 0.0, 0.013))
    hover = FakeRobot(layer_scene, name="drone_1", position=(1.0, 0.5, 1.2))
    parked_ground = FakeRobot(layer_scene, name="jackal_1", position=(-1.0, 0.5, 0.044))
    walker = Navigable(None, ground, require_traversable=False)
    assert walker._clear_of_other_robots((1.0, 0.5)), "the drone overhead is another layer"
    assert not walker._clear_of_other_robots((-1.0, 0.5)), "the Jackal on the floor still blocks"
    flyer2 = Navigable(None, hover, require_traversable=False)
    assert flyer2._clear_of_other_robots((0.0, 0.0)), "and the ground robot does not block the drone"
    ok("z-separated bodies are different layers; same-layer blocking unchanged")

    print("test: navigate_to a room by name lands inside that room")
    room_scene = FakeScene(FakeSegMap())
    rover = FakeRobot(room_scene, name="agent_0")
    roomer = Navigable(None, rover, require_traversable=False)
    landed = []
    roomer._navigate_to_pose = lambda pose: (landed.append((float(pose[0]), float(pose[1]))), iter(()))[1]
    for _ in range(20):
        list(roomer.navigate_to_room("hall_0"))
    assert landed and all(room_scene.seg_map.get_room_instance_by_point(xy) == "hall_0" for xy in landed), landed
    # A parked robot's spot is not offered; an unknown room is refused by name.
    parked = FakeRobot(room_scene, name="agent_1", position=(5.0, 0.0, 0.0))
    for _ in range(20):
        list(roomer.navigate_to_room("hall_0"))
    assert all(math.hypot(x - 5.0, y) >= roomer.robot_separation for x, y in landed[20:]), landed[20:]
    try:
        list(roomer.navigate_to_room("attic_0")); raise AssertionError("unknown room accepted")
    except Exception as error:  # noqa: BLE001
        assert "No room named 'attic_0'" in str(error) and "kitchen_0" in str(error), error
    ok("navigate_to_room samples inside the room, skips occupied spots, refuses unknown rooms")

    print("test: a settle is at most MAX_SETTLE_TICKS, and stops early when still")
    calm = FakeRobot(FakeScene(FakeSegMap()), name="calm")
    calm.get_linear_velocity = lambda: FakeVector([0.0, 0.0, 0.0])
    calm.q_to_action = lambda q: "hold"
    calm.get_joint_positions = lambda: 0
    spinner = FakeRobot(FakeScene(FakeSegMap()), name="spinner")
    spinner.get_linear_velocity = lambda: FakeVector([2.0, 0.0, 0.0])
    spinner.q_to_action = lambda q: "hold"
    spinner.get_joint_positions = lambda: 0
    for robot, expected in ((calm, 0), (spinner, Navigable.MAX_SETTLE_TICKS)):
        ctrl = Navigable(None, robot, require_traversable=False)
        ctrl._postprocess_action = lambda a: a
        ticks = list(ctrl._settle_robot())
        assert len(ticks) == expected, (robot.name, len(ticks), expected)
    assert Navigable.MAX_SETTLE_TICKS == 10
    ok("still robot: 0 ticks; a robot that never stops: 10, not 550")

    print("test: a spot taken by a robot says which robot, in the failure an agent reads")
    # "rejected_by {'robots': 146}" tells an agent to wait without telling it
    # who it is waiting for. With fifteen robots in a house that is the
    # difference between "wait" and "ask jackal_1 to move" (user, 2026-09-14).
    blame_scene = FakeScene(FakeSegMap())
    me = FakeRobot(blame_scene, name="ridgeback_3", position=(0.0, 0.0, 0.0))
    FakeRobot(blame_scene, name="jackal_1", position=(1.0, 0.0, 0.0))
    blamer = Navigable(None, me, require_traversable=False)
    assert blamer._blocking_robot((1.0, 0.0)) == "jackal_1", blamer._blocking_robot((1.0, 0.0))
    assert blamer._blocking_robot((9.0, 9.0)) is None
    # The bool wrapper the rest of the code uses still agrees with it.
    assert not blamer._clear_of_other_robots((1.0, 0.0))
    assert blamer._clear_of_other_robots((9.0, 9.0))
    # A reservation blames its owner too -- a robot that has only decided to
    # go there is as much in the way as one already standing there.
    blamer._nav_destinations = module.DestinationRegistry()
    blamer._nav_destinations.reserve("drone_2", (-2.0, 0.0))
    assert blamer._blocking_robot((-2.0, 0.0)) == "drone_2"
    assert blamer._nav_destinations.others("ridgeback_3") == [(-2.0, 0.0)], "the xy-only form still works"

    # And the failure an agent actually reads names them, most costly first.
    target = FakeObject("coffee_table_0", [0.0, 0.0, 0.4])
    boxed = Navigable(None, me, require_traversable=False)
    boxed._blocking_robot = lambda xy: "jackal_1" if float(xy[0]) > 0 else "drone_2"
    assert boxed._sample_pose_near_object(target) is None, "every spot is taken"
    assert boxed._nav_last_rejections["robots"] > 0, boxed._nav_last_rejections
    assert set(boxed._nav_last_blockers) == {"jackal_1", "drone_2"}, boxed._nav_last_blockers
    try:
        next(boxed._navigate_to_obj(target))
        raise AssertionError("a target with no free spot must raise")
    except FakeActionPrimitiveError as error:
        text, info = str(error), error.metadata
    assert "Robots occupying the space now:" in text, text
    assert "jackal_1" in text and "drone_2" in text, text
    assert "Message them to leave, or take a different target." in text, text
    # Every blocker by name, never "and N other robot(s)": a robot the agent is
    # not told about is one it cannot ask to move (user, 2026-09-14).
    assert "other robot" not in text, text
    # The annulus is not in the prose: the agent does not choose where it
    # stands, the sampler does, so "0.9-1.5 m away" is nothing it can act on
    # (user, 2026-09-14). It stays in the metadata for the logs.
    assert "m away" not in text and "need a spot" not in text, text
    assert info["sampling_range"], info
    assert set(info["blocked_by"]) == {"jackal_1", "drone_2"}, info
    # Most costly first, so the first name is the one worth addressing.
    assert list(info["blocked_by"]) == sorted(info["blocked_by"],
                                              key=info["blocked_by"].get, reverse=True), info
    ok("a blocked sample names every robot on the spots, and says what to do about them")

    print("test: and when no robot is in the way, it does not say one is")
    # The old text blamed "another agent" and advised waiting whatever had
    # rejected the poses. Over the 189 navigation failures of `S1_full_fast`
    # the dominant filter was the room in 53% and traversability in 33%, and
    # 26 had no robot rejection at all -- five times in six the agent was told
    # to wait for a teammate who was not there (user, 2026-09-14).
    why = module._why_no_space
    # The old text blamed "another agent" and advised waiting whatever had
    # rejected the poses. Over the 189 navigation failures of `S1_full_fast`
    # the dominant filter was the room in 53% and traversability in 33%, and
    # 26 had no robot rejection at all -- five times in six the agent was told
    # to wait for a teammate who was not there (user, 2026-09-14). There is
    # nobody to message then, and an empty list is worse than saying so.
    empty = why({"room": 151, "trav": 49, "robots": 0}, {})
    assert "No robot is in the way" in empty, empty
    assert "Robots occupying" not in empty and "Message" not in empty, empty
    assert "waiting will not free it" in empty, empty
    # One shape for every target and every breakdown: the names decide it, and
    # nothing else does -- an object target and a teammate target read alike.
    named = why({"room": 0, "trav": 78, "robots": 122}, {"jackal_1": 122})
    assert named == why(None, {"jackal_1": 1}), (named, why(None, {"jackal_1": 1}))
    assert named.startswith(" Robots occupying the space now: jackal_1."), named
    # And the room-target failure says it the same way.
    room_ctrl = Navigable(None, me, require_traversable=False)
    room_ctrl._blocking_robot = lambda xy: "drone_7"
    try:
        list(room_ctrl.navigate_to_room("kitchen_0"))
        raise AssertionError("expected PLANNING_ERROR")
    except FakeActionPrimitiveError as error:
        assert "Robots occupying the space now: drone_7." in str(error), str(error)
    ok("one shape everywhere: the robots on the spots, then leave-or-retarget")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
