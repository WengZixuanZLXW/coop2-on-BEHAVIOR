"""A cuRobo-free ``NAVIGATE_TO`` for the symbolic primitive set.

``SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO`` is in the enum and mapped to
a handler, but it cannot execute as shipped. Two independent defects on the
same path:

1. ``SymbolicSemanticActionPrimitives`` overrides only ``_navigate_to_pose``
   (making it a pure teleport). ``_navigate_to_obj`` and
   ``_sample_pose_near_object`` are inherited from
   ``StarterSemanticActionPrimitives``, and the sampler opens with
   ``self._motion_generator.update_obstacles()`` then validates candidates with
   cuRobo IK + collision checks. The symbolic constructor passes
   ``skip_curobo_initilization=True``, so ``self._motion_generator is None``
   and the call raises ``AttributeError: 'NoneType' object has no attribute
   'update_obstacles'``.
2. Even with a motion generator present, ``_navigate_to_obj`` calls
   ``self._navigate_to_pose(pose, skip_obstacle_update=...)`` while the
   symbolic override is declared ``_navigate_to_pose(self, pose_2d)`` -- no
   such keyword -- so it would raise ``TypeError``.

So NAVIGATE_TO is unusable in the symbolic set exactly the way OPEN / CLOSE /
TOGGLE_ON / TOGGLE_OFF are unusable in the physical set (those raise
``NotImplementedError``).

This module supplies the missing piece: sample a base pose around the target,
keep it in the same room, and teleport there. No cuRobo, no collision check,
no reachability check -- which is consistent with the rest of the symbolic set,
where ``_grasp`` teleports the object to the end-effector at any distance and
``_navigate_to_pose`` teleports the base with no collision checking either.

Why bother navigating at all when symbolic grasping ignores distance? Because
COOP2's **spatial** constraint is defined on where the agents are: "how many
agents are near the task". Without a working NAVIGATE_TO there is no way for a
symbolic-mode agent to satisfy it, and the spatial metric degenerates.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch as th

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
)

__all__ = ["DestinationRegistry", "NavigableSymbolicActionPrimitives"]


class DestinationRegistry:
    """Where each agent has *decided* to teleport, before it gets there.

    Checking separation against other robots' current positions is not enough
    under concurrency, and the failure is a textbook time-of-check/time-of-use
    race. All N agents are assigned NAVIGATE_TO on the same tick; each samples
    its destination on its generator's first ``next()``, while every other robot
    is still standing at its start pose metres away. Every check passes. Then
    all of them teleport next to the same object and interpenetrate -- measured
    at 0.64 m apart against a 1.24 m requirement, which was enough for one
    robot's assisted grasp to latch onto **another robot** (``holding=agent_0``).

    So agents reserve the pose they are about to occupy, and everyone else's
    sampler avoids reservations as well as bodies.

    Reservations are overwritten, never explicitly released: a reservation means
    "this agent intends to be here", which stays true until it decides to go
    somewhere else, and once the robot has actually arrived the reservation and
    its body coincide. That makes a dropped or aborted primitive self-correcting
    instead of leaking a permanently blocked spot.
    """

    def __init__(self):
        self._by_agent: dict = {}

    def reserve(self, agent_id: str, xy) -> None:
        self._by_agent[agent_id] = (float(xy[0]), float(xy[1]))

    def release(self, agent_id: str) -> None:
        self._by_agent.pop(agent_id, None)

    def others(self, agent_id: str):
        return [xy for name, xy in self._by_agent.items() if name != agent_id]

    def __len__(self) -> int:
        return len(self._by_agent)


class NavigableSymbolicActionPrimitives(SymbolicSemanticActionPrimitives):
    """Symbolic primitives whose ``NAVIGATE_TO`` actually runs.

    Drop-in replacement for :class:`SymbolicSemanticActionPrimitives`; only the
    two broken navigation methods are overridden.

    Upstream samples the same 0-1.5 m annulus we do, but then runs every
    candidate through ``_validate_poses`` -- cuRobo IK plus a collision query --
    and returns only one that passes. That stage is what keeps the robot out of
    walls, not the distance range. Dropping cuRobo therefore means replacing it,
    and measurement says the replacement has to be *three* checks, because each
    covers a different obstacle class:

    * ``reach``/``clearance`` -- the target itself. cuRobo does not catch this:
      an apple is 5 cm on the floor and the base clears it, so ``check_collisions``
      called every pose from 0.05 m out collision-free (8/8). Symbolic grasp then
      teleports the object into a robot standing on top of it and the settle
      shakes it loose.
    * ``require_traversable`` -- static geometry. The scene's baked trav map,
      eroded by the robot's own radius, is what cuRobo's collision query buys us,
      for free and on the CPU.
    * ``robot_separation`` -- teammates. Static maps do not contain them. Without
      this, two robots teleport into the same spot, PhysX shoves them apart, and
      the physical assisted grasp latches onto the other *robot*.

    Args:
        env, robot: as the base class.
        reach: metres of usable annulus *beyond* the target's clearance radius.
            The sampling range is ``[clearance(obj), clearance(obj) + reach]``,
            so it widens automatically for a table and stays tight for an apple.
            A fixed range cannot do this: for a large object a 1.5 m upper bound
            is inside the object.

            This also sets the **manipulation gate**, because
            ``interaction_radius_for`` is this annulus's upper bound plus a
            margin -- deliberately, so that a pose ``navigate_to`` produced can
            never be rejected as ``TOO_FAR``, which would loop forever. So
            ``reach`` is the one number that decides how far away an agent may
            grasp or place: the slack works out to
            ``reach + 0.35 + clearance_margin`` of clear floor between the
            robot's edge and the object's, the same for every object.

            Was 1.5 (1.9 m of clear floor -- twice R1's arm), then 0.8 (1.2 m).
            **Now 0.6**, which with the manipulation guard gives a 0.70 m gap.
            Measured before changing: shrinking the annulus does not cost standing
            poses, it gains them, because the outer ring of a wide annulus mostly
            falls outside the room or on non-traversable floor (coffee table
            acceptance 48.8 % at 1.5, 60.0 % at 0.6; apples 37.7 %/25.2 % ->
            38.2 %/35.2 %). Below 0.6 the apples do start losing poses, which is
            why this is the floor.
        clearance_margin: slack added to (target half-diagonal + robot radius).
        robot_radius: circumscribed radius. ``None`` derives it from
            ``reset_joint_pos_aabb_extent``, which is what the trav map's own
            erosion uses.
        robot_separation: minimum centre-to-centre distance to another robot.
            ``None`` means ``2 * robot_radius`` -- just-touching. Nothing more is
            needed: symbolic navigation teleports, so there is no path to keep
            clear, only bodies to keep from overlapping.
        require_traversable: reject poses where the eroded trav map says the
            robot does not fit. Silently inert on a scene with no trav map.
        destinations: shared :class:`DestinationRegistry`. Without one,
            separation is checked only against where robots currently stand,
            which under concurrency is where they were *before* they all
            teleported to the same place.
        sampling_attempts: how many candidates to try before giving up.
        require_same_room: reject candidates outside the target's room. Set
            False for scenes without a segmentation map (a plain ``Scene``
            rather than an ``InteractiveTraversableScene``).
        distance_range: absolute ``(lo, hi)`` override. When given it replaces
            the object-relative range entirely -- kept for the stubbed tests and
            for reproducing the old behaviour.
    """

    def _get_robot_pose_from_2d_pose(self, pose_2d):
        """(x, y, yaw) -> world pose, keeping the altitude the robot has *now*.

        Upstream builds the z of a holonomic base from its z *joint*, which is
        measured from the root anchor, and hands that joint value back as a
        *world* z. The anchor sits at the spawn height (0.05 m), so every
        teleport re-applies the pose 5 cm lower than the body actually is.
        A wheeled base never notices -- the floor pushes it back up. A drone
        holding altitude on a driven z joint does: measured 1.200 -> 1.150 ->
        1.100 -> 1.050 over three navigates, a steady 5 cm per hop.

        The right z for a teleport in the plane is the one the body already has.
        Orientation follows upstream (rx, ry from the joints, yaw from the
        command).
        """
        import importlib  # noqa: PLC0415
        import torch as th  # noqa: PLC0415

        # By name, not `import a.b.c as T`: the latter also walks the parent
        # packages, which the CPU stub tests do not provide.
        T = importlib.import_module("omnigibson.utils.transform_utils")

        if not self.robot.is_holonomic_base:
            return super()._get_robot_pose_from_2d_pose(pose_2d)
        world_z = float(self.robot.get_position_orientation()[0][2])
        q = self.robot.get_joint_positions()
        idx = [int(i) for i in self.robot.base_idx]          # x, y, z, rx, ry, rz
        pos = th.tensor([float(pose_2d[0]), float(pose_2d[1]), world_z], dtype=th.float32)
        eulers = th.tensor([float(q[idx[3]]), float(q[idx[4]]), float(pose_2d[2])], dtype=th.float32)
        orn = T.mat2quat(T.euler_intrinsic2mat(eulers))
        return pos, orn

    def __init__(
        self,
        env,
        robot,
        reach: float = 0.6,
        clearance_margin: float = 0.05,
        robot_radius: Optional[float] = None,
        robot_separation: Optional[float] = None,
        require_traversable: bool = True,
        destinations: Optional["DestinationRegistry"] = None,
        sampling_attempts: int = 200,
        require_same_room: bool = True,
        distance_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(env, robot, **kwargs)
        self._nav_reach = float(reach)
        self._nav_clearance_margin = float(clearance_margin)
        self._nav_robot_radius = robot_radius
        self._nav_robot_separation = robot_separation
        self._nav_require_traversable = bool(require_traversable)
        # Shared across every controller in the scene, or separation is only
        # ever checked against stale positions. None = no reservations.
        self._nav_destinations = destinations
        self._nav_sampling_attempts = sampling_attempts
        self._nav_require_same_room = require_same_room
        self._nav_distance_range = distance_range
        self._nav_eroded_map = None

    # -- geometry ---------------------------------------------------------

    @property
    def robot_radius(self) -> float:
        """Circumscribed radius of the robot at its reset pose.

        ``reset_joint_pos_aabb_extent`` is the whole-robot AABB (arms included),
        so half its planar diagonal is the radius of a circle that contains the
        robot at *any* yaw. Conservative for a rectangular chassis; an oriented
        box test could pack tighter at the cost of real code.
        """
        if self._nav_robot_radius is None:
            try:
                extent = self.robot.reset_joint_pos_aabb_extent[:2]
                self._nav_robot_radius = float(th.norm(extent)) / 2.0
            except Exception:  # noqa: BLE001 - robots without the property
                self._nav_robot_radius = 0.6
        return self._nav_robot_radius

    #: Bodies further apart than this in z occupy different layers (a drone
    #: over a ground robot) and do not block each other's standing spots.
    LAYER_SEPARATION_Z = 0.6

    @property
    def robot_separation(self) -> float:
        if self._nav_robot_separation is None:
            self._nav_robot_separation = 2.0 * self.robot_radius
        return self._nav_robot_separation

    def clearance_for(self, obj) -> float:
        """Smallest centre distance at which the robot does not overlap @obj."""
        try:
            extent = obj.aabb_extent[:2]
            half_diagonal = float(th.norm(extent)) / 2.0
        except Exception:  # noqa: BLE001 - objects without an aabb
            half_diagonal = 0.0
        return half_diagonal + self.robot_radius + self._nav_clearance_margin

    @staticmethod
    def is_walkable_surface(obj) -> bool:
        """Is @obj something the robot stands *on* rather than beside?"""
        return str(getattr(obj, "category", "")).lower() in ("floors", "carpet", "rug")

    def sampling_range_for(self, obj) -> Tuple[float, float]:
        """``(lo, hi)`` metres to sample the base position in, for @obj.

        A floor is the exception, and it is not a small one. The distance below
        comes from the target's own half-diagonal, which for a floor spanning a
        room is metres: navigating to one asked the robot to stand 3.3-3.9 m
        from its centre, i.e. on a ring outside the room it is the floor of.
        187 of 200 candidates were then rejected for being in the wrong room and
        the primitive failed NO_SPACE_AROUND_TARGET every time -- which is how
        the only way to say "take this to the bedroom" failed.

        So for a surface the robot stands on, sample *within* it instead. The
        room filter and the traversability check still apply, so what comes back
        is a spot on that floor the robot actually fits at.
        """
        if self._nav_distance_range is not None:
            return tuple(self._nav_distance_range)
        if self.is_walkable_surface(obj):
            try:
                extent = obj.aabb_extent[:2]
                half_diagonal = float(th.norm(extent)) / 2.0
            except Exception:  # noqa: BLE001 - objects without an aabb
                half_diagonal = self._nav_reach
            # Not right to the edge: leave the robot's own radius of margin so a
            # sampled point is not half outside the surface.
            return 0.0, max(self._nav_reach, half_diagonal - self.robot_radius)
        clearance = self.clearance_for(obj)
        return clearance, clearance + self._nav_reach

    # -- traversability ---------------------------------------------------

    def _trav_map(self):
        scene = self.robot.scene
        return getattr(scene, "trav_map", None)

    def _eroded_trav_map(self):
        """The floor map eroded by the robot radius, built once and cached."""
        if self._nav_eroded_map is None:
            trav_map = self._trav_map()
            floor_map = getattr(trav_map, "floor_map", None)
            if not floor_map:
                return None
            self._nav_eroded_map = trav_map._erode_trav_map(th.clone(floor_map[0]), robot=self.robot)
        return self._nav_eroded_map

    def _is_traversable(self, xy) -> bool:
        """Does the robot fit at @xy? True when the scene has no trav map."""
        if not self._nav_require_traversable:
            return True
        eroded = self._eroded_trav_map()
        if eroded is None:
            return True
        row, col = self._trav_map().world_to_map(th.tensor([float(xy[0]), float(xy[1])]))
        row, col = int(row), int(col)
        if not (0 <= row < eroded.shape[0] and 0 <= col < eroded.shape[1]):
            return False
        return bool(eroded[row][col] == 255)

    def _clear_of_other_robots(self, xy) -> bool:
        """Is @xy clear of every other robot's body **and** its destination?"""
        separation = self.robot_separation
        scene = getattr(self.robot, "scene", None)
        robots = getattr(scene, "robots", None)
        if robots is None:
            robots = getattr(self.env, "robots", None) or []
        occupied = []
        my_z = float(self.robot.get_position_orientation()[0][2])
        for other in robots:
            if other is self.robot:
                continue
            other_position = other.get_position_orientation()[0]
            # A hovering drone and a ground robot are not in each other's
            # way: the drone flies at 1.2 m, the Jackal is 0.7 m tall and the
            # Ridgeback 0.57. With nine robots in a 2.7 m2 kitchen (2026-09-13)
            # a drone parked over the bookcase was costing the arm every spot
            # it could have stood at. Bodies more than a robot height apart in
            # z are different layers; the same-layer rule is unchanged.
            if abs(float(other_position[2]) - my_z) > self.LAYER_SEPARATION_Z:
                continue
            occupied.append((float(other_position[0]), float(other_position[1])))
        if self._nav_destinations is not None:
            # The bodies are where everyone *was*; the reservations are where
            # everyone is *going*. Under concurrency only the second set is
            # current, because nobody has teleported yet when the samplers run.
            occupied.extend(self._nav_destinations.others(self.robot.name))
        for other_x, other_y in occupied:
            if math.hypot(float(xy[0]) - other_x, float(xy[1]) - other_y) < separation:
                return False
        return True

    # -- helpers ----------------------------------------------------------

    def _seg_map(self):
        """The scene's segmentation map, or None on a non-traversable scene."""
        scene = self.robot.scene
        return getattr(scene, "seg_map", None) or getattr(scene, "_seg_map", None)

    def _room_of(self, xy) -> Optional[str]:
        seg_map = self._seg_map()
        if seg_map is None:
            return None
        # Note get_room_type_by_point is broken upstream (it indexes a dict
        # with a 0-dim tensor); the instance variant calls .item() and works.
        return seg_map.get_room_instance_by_point(xy)

    def _target_rooms(self, obj, target_xy) -> Sequence[Optional[str]]:
        """Rooms the target counts as being in.

        ``in_rooms`` is static scene metadata assigned at load time and is
        never updated when an object moves, and objects added through the env
        config have none at all -- so fall back to a live point query, exactly
        as the upstream sampler does.
        """
        in_rooms = getattr(obj, "in_rooms", None)
        if in_rooms:
            return list(in_rooms)
        return [self._room_of(target_xy)]

    def _facing_yaw_offset(self) -> float:
        """Yaw correction that leaves the target in front of the arm.

        Verbatim from the upstream sampler: ``yaw + pi - mean(workspace)``.
        """
        try:
            return math.pi - float(th.mean(self.robot.arm_workspace_range[self.arm]))
        except Exception:  # noqa: BLE001 - robots without a workspace range
            return math.pi

    # -- overrides --------------------------------------------------------

    def _sample_pose_near_object(
        self,
        obj,
        eef_pose=None,
        plan_with_open_gripper=False,
        sampling_attempts=None,
        skip_obstacle_update=False,
    ):
        """cuRobo-free replacement for the inherited sampler.

        Keeps the upstream geometry (polar sampling around the target, robot
        yawed to face it, same-room rejection) and drops the two cuRobo stages
        (``update_obstacles`` and ``_validate_poses``).

        Returns:
            th.Tensor or None: ``(x, y, yaw)``, or None if no candidate landed
            in the target's room.
        """
        if eef_pose is not None:
            target_position = eef_pose[0]
        else:
            target_position = obj.get_position_orientation()[0]
        target_xy = target_position[:2]

        target_rooms = self._target_rooms(obj, target_xy)
        yaw_offset = self._facing_yaw_offset()
        distance_lo, distance_hi = self.sampling_range_for(obj)
        attempts = self._nav_sampling_attempts if sampling_attempts is None else sampling_attempts

        # Which filter rejected how many. NO_SPACE_AROUND_TARGET only says
        # "none of the candidates passed", and three separate hypotheses about
        # which filter was responsible were all wrong when measured offline --
        # the annulus accepts 28-48% of candidates in every state that could be
        # reproduced by hand. Reporting the breakdown at the moment of failure
        # is cheaper than guessing which state the episode was in.
        rejected = {"room": 0, "trav": 0, "robots": 0}
        for _ in range(attempts):
            distance = th.rand(1).item() * (distance_hi - distance_lo) + distance_lo
            yaw = th.rand(1).item() * 2.0 * math.pi - math.pi
            candidate = th.tensor(
                [
                    float(target_xy[0]) + distance * math.cos(yaw),
                    float(target_xy[1]) + distance * math.sin(yaw),
                    yaw + yaw_offset,
                ],
                dtype=th.float32,
            )
            # Room first: it is the cheapest of the three and, on a scene
            # where the target has no room, it is a no-op rather than a
            # rejection (target_rooms is then [None] and _room_of returns
            # None for anything off the map).
            if self._nav_require_same_room and self._seg_map() is not None:
                if self._room_of(candidate[:2]) not in target_rooms:
                    rejected["room"] += 1
                    continue
            if not self._is_traversable(candidate[:2]):
                rejected["trav"] += 1
                continue
            if not self._clear_of_other_robots(candidate[:2]):
                rejected["robots"] += 1
                continue
            if self._nav_destinations is not None:
                self._nav_destinations.reserve(self.robot.name, candidate[:2])
            return candidate

        self._nav_last_rejections = dict(rejected)
        # No candidate satisfied all of the filters. Returning None makes
        # _navigate_to_obj raise PLANNING_ERROR -- the honest signal that this
        # target has no standable pose around it. Measured on Rs_int that is
        # 46% of targets, which is a property of the scene, not of this code:
        # see feasibility_verify/measure_teleport_risk.py. Previously the
        # sampler returned the first candidate regardless, so the robot
        # teleported into a wall and toppled instead of reporting failure.
        return None

    # -- what an object rests on -------------------------------------------

    @staticmethod
    def _aabb_of(obj):
        """``(lo, hi)`` world AABB of @obj, from ``aabb`` when the object has
        one, else from its position and ``aabb_extent`` (the CPU fakes)."""
        aabb = getattr(obj, "aabb", None)
        if aabb is not None and not callable(aabb):
            lo, hi = aabb
            return [float(v) for v in lo], [float(v) for v in hi]
        position = obj.get_position_orientation()[0]
        extent = getattr(obj, "aabb_extent", None) or (0.0, 0.0, 0.0)
        lo = [float(position[i]) - float(extent[i]) / 2.0 for i in range(3)]
        hi = [float(position[i]) + float(extent[i]) / 2.0 for i in range(3)]
        return lo, hi

    def support_of(self, obj):
        """The piece of furniture @obj is resting on, or None.

        A die on a bookcase is 0.9 m from nowhere a robot can stand: the
        sampler wants a spot 0.3-0.9 m from the die, and every such spot is
        inside the bookcase's footprint or in the wall behind it. In the first
        routed LL episode (2026-09-13) nine robots spent 2000 steps on that,
        alternating NO_SPACE_AROUND_TARGET on the die with TOO_FAR after
        navigating to the bookcase, because the gate was the die's own. What
        a person does is stand at the bookcase and reach across it, so both
        the sampler (:meth:`_navigate_to_obj`) and the reach gate
        (``interaction_radius_for``) use the support when there is one.

        Geometric, not a physics query: the highest non-walkable, non-robot
        object whose footprint contains @obj's xy and whose top is at @obj's
        bottom (within 0.15 m). Floors are excluded -- a die on the floor is
        approached as before -- and so is anything with no room to stand on
        that is smaller than the object itself.
        """
        scene = getattr(self.robot, "scene", None)
        objects = getattr(scene, "objects", None)
        if not objects:
            return None
        try:
            lo, _ = self._aabb_of(obj)
            x, y = self._object_xy_of(obj)
        except Exception:  # noqa: BLE001
            return None
        bottom = lo[2]
        best, best_top = None, None
        for other in objects:
            if other is obj or getattr(other, "category", None) == "agent":
                continue
            if self.is_walkable_surface(other) or hasattr(other, "action_dim"):
                continue
            try:
                o_lo, o_hi = self._aabb_of(other)
            except Exception:  # noqa: BLE001
                continue
            margin = 0.05
            if not (o_lo[0] - margin <= x <= o_hi[0] + margin and o_lo[1] - margin <= y <= o_hi[1] + margin):
                continue
            top = o_hi[2]
            if top > bottom + 0.15 or top < bottom - 0.15:
                continue
            if best_top is None or top > best_top:
                best, best_top = other, top
        return best

    @staticmethod
    def _object_xy_of(obj):
        position = obj.get_position_orientation()[0]
        return float(position[0]), float(position[1])

    def navigate_to_room(self, room: str):
        """Drive to a free spot inside @room (a room instance, ``kitchen_0``).

        The room listing is strictly local, so before this the only way into
        another room was to name an object there -- which an agent cannot do
        for a room whose objects it has never seen. The prompt now lists every
        room in the house and says navigate_to takes a room (user,
        2026-09-13). Same filters as an object target: traversable, clear of
        other robots' bodies and reservations. A generator like every other
        primitive; all checks run on the first ``next()``.
        """
        seg_map = self._seg_map()
        names = getattr(seg_map, "room_ins_name_to_ins_id", None) or {}
        if room not in names:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                f"No room named {room!r} in this house. Rooms: {', '.join(sorted(names)) or 'unknown'}.",
                {"room": room, "reason_code": "INVALID_TARGET"},
            )
        import random  # noqa: PLC0415

        attempts = self._nav_sampling_attempts or 200
        rejected = {"room": 0, "trav": 0, "robots": 0}
        for _ in range(attempts):
            _, point = seg_map.get_random_point_by_room_instance(room)
            if point is None:
                break
            xy = (float(point[0]), float(point[1]))
            if self._room_of(xy) not in (room, None):
                rejected["room"] += 1
                continue
            if not self._is_traversable(xy):
                rejected["trav"] += 1
                continue
            if not self._clear_of_other_robots(xy):
                rejected["robots"] += 1
                continue
            if self._nav_destinations is not None:
                self._nav_destinations.reserve(self.robot.name, xy)
            pose = th.tensor([xy[0], xy[1], random.uniform(-math.pi, math.pi)], dtype=th.float32)
            yield from self._navigate_to_pose(pose)
            return
        self._nav_last_rejections = dict(rejected)
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PLANNING_ERROR,
            f"Cannot find a free spot to stand in {room}: every sampled point was blocked by "
            "furniture, walls or other robots. Wait for them to move, or go elsewhere.",
            {"room": room, "rejected_by": dict(rejected), "reason_code": "NO_SPACE_IN_ROOM"},
        )

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        """Same as upstream, minus the keyword the symbolic override rejects.

        The inherited version forwards ``skip_obstacle_update`` into
        ``_navigate_to_pose``, but the symbolic override's signature is
        ``_navigate_to_pose(self, pose_2d)``, so that call is a ``TypeError``.

        An object resting on furniture is approached by way of the furniture
        (see :meth:`support_of`): the spot is sampled around the support, and
        ``interaction_radius_for`` accepts anything that produces.
        """
        support = self.support_of(obj) if eef_pose is None else None
        anchor = support if support is not None else obj
        pose = self._sample_pose_near_object(anchor, eef_pose=eef_pose)
        if pose is None:
            lo, hi = self.sampling_range_for(anchor)
            if support is not None:
                obj_label = f"{obj.name} (it rests on {support.name}, which is what you stand at)"
            else:
                obj_label = obj.name
            # This text goes into the prompt verbatim, so it has to say what an
            # agent can act on. "Could not find a valid base pose" describes the
            # sampler's internals; what the agent needs to know is that the
            # space around the target is taken -- by furniture, by walls, or by
            # teammates who reserved it first -- and that waiting or retargeting
            # is the move, not retrying.
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                f"Cannot reach {obj_label}: there is no free floor space around it to stand on "
                f"(need a spot {lo:.1f}-{hi:.1f} m away, clear of walls, furniture and other agents). "
                "Another agent may already be standing there. Try a different target, or wait for "
                "them to move.",
                {"object": obj.name, "sampling_range": [round(lo, 2), round(hi, 2)],
                 "rejected_by": getattr(self, "_nav_last_rejections", None),
                 "target_xy": [round(float(v), 2) for v in obj.get_position_orientation()[0][:2]],
                 "reason_code": "NO_SPACE_AROUND_TARGET"},
            )
        yield from self._navigate_to_pose(pose)
