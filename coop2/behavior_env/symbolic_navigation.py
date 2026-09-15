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
from collections import Counter
from typing import Mapping, Optional, Sequence, Tuple

import torch as th

from coop2.behavior_env.geometry_cache import aabb_cache, invalidate_aabb_cache
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
)

__all__ = [
    "DEFAULT_SAMPLING_ATTEMPTS",
    "DestinationRegistry",
    "NavigableSymbolicActionPrimitives",
    "invalidate_aabb_cache",
]

#: Candidates tried before a target is called unreachable. **800 since
#: 2026-09-14** (user); 200 before that.
#:
#: The number matters only where the valid fraction of the annulus is tiny, and
#: on `v4_s1_v4_lh` it is: enumerating every cell of the ring on Merom_1_int
#: (2026-09-14) the Ridgeback has **1 standable cell of 608** at the breakfast
#: table and the Jackal **0 of 648**, against 236/520 and 210/568 at the coffee
#: table. At p = 1/608 a target that *is* reachable is found 28% of the time in
#: 200 draws and 73% in 800 -- so C5 of that route was failing on the sampler's
#: budget, not on the scene. Zero stays zero: no budget finds a spot that does
#: not exist, which is why the failure text now says so (`_why_no_space`).
#:
#: It is not a cost on the success path: the sampler returns on its first valid
#: candidate, which at the coffee table is the third or fourth draw. The whole
#: budget is spent only on a failure, which was already paying for a replan.
DEFAULT_SAMPLING_ATTEMPTS = 800

def _named(blockers: Mapping[str, int]) -> str:
    """``jackal_1, drone_2, ridgeback_4`` -- the one costing most candidates first.

    Every one of them, never "and N other robot(s)" (user, 2026-09-14): the
    point of the name is that the agent can address that robot, and a robot it
    is not told about is one it cannot ask to move. Fifteen is the most any
    layout has, so the whole list is a line.
    """
    return ", ".join(blockers)


def _why_no_space(rejected: Optional[Mapping[str, int]],
                  blockers: Optional[Mapping[str, int]]) -> str:
    """What is on the spots, as one clause the agent can act on.

    One shape for every target, object or teammate (user, 2026-09-14): name the
    robots standing on the annulus, then the two moves that exist -- ask them
    to leave, or go somewhere else. It replaces a fixed sentence that blamed
    "another agent" whatever had actually rejected the poses, and then a
    three-way branch on the filter breakdown.

    The no-robot case is the common one and still has to be honest about it:
    over the 189 navigation failures of the `S1_full_fast` sweep the filter
    that rejected the most candidates was the room in 53%, traversability in
    33% and a robot in only 14%, and 26 had no robot rejection at all. There
    is nobody to message then, and no amount of waiting frees a wall, so that
    branch says so in one line instead of naming an empty list.
    """
    blockers = dict(blockers or {})
    if blockers:
        return (f" Robots occupying the space now: {_named(blockers)}. "
                "Message them to leave, or take a different target.")
    return (" No robot is in the way -- walls or the room itself take the space, "
            "so waiting will not free it. Take a different target.")


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

    def others_named(self, agent_id: str):
        """``(agent_id, xy)`` for every reservation but @agent_id's own, so a
        sampler can say *which* agent took the spot, not only that one did."""
        return [(name, xy) for name, xy in self._by_agent.items() if name != agent_id]

    def others(self, agent_id: str):
        return [xy for _, xy in self.others_named(agent_id)]

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
        sampling_attempts: how many candidates to try before giving up; see
            :data:`DEFAULT_SAMPLING_ATTEMPTS` for why it is 800.
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
        sampling_attempts: int = DEFAULT_SAMPLING_ATTEMPTS,
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

    def body_offset(self) -> float:
        """How far the robot's own body needs, approaching a target.

        Its chassis **half-width**, not the circumscribed radius the rest of
        this class uses. The radius is the 45-degree worst case; a base pose
        here is yawed to face its target (`_facing_yaw_offset`), so what points
        at the object is a face, and the face is the half-width. On the
        Ridgeback (0.5 x 0.5 chassis) that is 0.25 m against a 0.354 m radius,
        on the Jackal (0.7 x 0.7) 0.35 against 0.495.

        The 0.10 m and 0.15 m this returns are not cosmetic. Merom_1_int's
        dining room, 2026-09-14: every standable cell around
        `breakfast_table_skczfi_0` lies 0.2-0.6 m from its footprint and there
        is **none at all past 0.6 m**, so a minimum offset of 0.40 m (radius +
        margin) left the Ridgeback one cell and the Jackal none. At half-width
        the floor is reachable and the poses are still clear: measured minimum
        clearance over 400 accepted poses per target is 0.30 m for the
        Ridgeback and 0.40 m for the Jackal, both above their half-width.

        Separation *between robots* keeps the circumscribed radius
        (`robot_separation`): two robots have no agreed facing, so there the
        45-degree case is the real one.
        """
        try:
            extent = self.robot.reset_joint_pos_aabb_extent[:2]
            return max(float(extent[0]), float(extent[1])) / 2.0
        except Exception:  # noqa: BLE001 - robots without a chassis extent
            return self.robot_radius

    def clearance_for(self, obj) -> float:
        """Smallest centre distance at which the robot does not overlap @obj."""
        try:
            extent = obj.aabb_extent[:2]
            half_diagonal = float(th.norm(extent)) / 2.0
        except Exception:  # noqa: BLE001 - objects without an aabb
            half_diagonal = 0.0
        return half_diagonal + self.body_offset() + self._nav_clearance_margin

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
        return self._blocking_robot(xy) is None

    def _blocking_robot(self, xy) -> Optional[str]:
        """The name of the robot occupying @xy, or ``None`` if it is clear.

        The name, not a count: "200/200 rejected by robots" tells an agent to
        wait without telling it *who* it is waiting for, and with fifteen
        robots in a house that is the difference between "wait" and "ask
        jackal_1 to move". The first blocker found wins; the sampler tallies
        names across its attempts.
        """
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
            occupied.append((getattr(other, "name", None) or str(other),
                             float(other_position[0]), float(other_position[1])))
        if self._nav_destinations is not None:
            # The bodies are where everyone *was*; the reservations are where
            # everyone is *going*. Under concurrency only the second set is
            # current, because nobody has teleported yet when the samplers run.
            occupied.extend((name, xy_other[0], xy_other[1])
                            for name, xy_other in self._nav_destinations.others_named(self.robot.name))
        for name, other_x, other_y in occupied:
            if math.hypot(float(xy[0]) - other_x, float(xy[1]) - other_y) < separation:
                return name
        return None

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
        """Rooms the target counts as being in -- the world model's rule.

        Fixed furniture keeps its static ``in_rooms`` annotation; anything that
        can move is point-queried where it stands now, falling back to the
        annotation only when the point lands on no room (a boundary cell).
        This mirrors ``world_state.rooms_of`` on purpose: the prompt and the
        sampler must agree on where a thing is.

        They did not, and it cost every S1 LL run of 2026-09-13 its last route
        node. This used to prefer ``in_rooms`` whenever present, and ``in_rooms``
        is written once at load -- the die was sampled in childs_room_0 and
        carried nine stations to a bed in bedroom_0. Every navigate to it after
        that rejected all 200 candidate poses with ``{'room': 200, 'trav': 0,
        'robots': 0}``: the poses were fine, they were just in the room the die
        was actually in rather than the one its label said. The world model
        meanwhile told the agent, correctly, that the die was in the bedroom.
        Furniture never showed it because furniture never moves; the first
        object anyone had to walk up to *after* it changed rooms was that die.
        """
        in_rooms = list(getattr(obj, "in_rooms", None) or [])
        if in_rooms and self._is_fixed(obj):
            return in_rooms
        live = self._room_of(target_xy)
        return [live] if live else (in_rooms or [None])

    def _is_fixed(self, obj) -> bool:
        """Whether @obj is one of the scene's immovable objects.

        ``scene.fixed_objects`` is a name -> object dict (so membership must be
        tested against its values, not the dict), and the CPU fakes may carry a
        plain ``fixed_base`` flag instead.
        """
        scene = getattr(self.robot, "scene", None)
        fixed = getattr(scene, "fixed_objects", None)
        if fixed:
            values = fixed.values() if hasattr(fixed, "values") else fixed
            if obj in set(values):
                return True
        return bool(getattr(obj, "fixed_base", False))

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
        prefer_xy=None,
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
        # And, when it was a robot, which one. A count says "wait"; a name says
        # who to wait for, or ask to move.
        blockers: Counter = Counter()
        # Two shapes, alternating, half the budget each -- not one chosen up
        # front. A band round the footprint (`_edge_candidate`) and a ring round
        # the centre find different floor, and which of them finds any depends
        # on the target's own proportions, measured 2026-09-14 over every route
        # support of S1 (Merom_1_int) and S2 (Beechwood_0_int):
        #
        #   coffee_table 1.03 x 2.05   ring 16%  band 52%   <- long: the band
        #   bookcase     1.83 x 0.47   ring 19%  band  1%
        #   chair        0.47 x 0.44   ring 13%  band  0%   <- small: the ring
        #   footstool    0.72 x 0.69   ring  1%  band  0%
        #
        # (Jackal.) The band hugs the object in a thin shell, which is the
        # whole point for something long and the wrong move for something
        # small, where the ring's wider annulus is likelier to hold free floor.
        # Choosing per shape-heuristic was tried and is not good enough --
        # bookcase is long and still loses. Alternating costs nothing: the
        # sampler returns on its first valid candidate, so only a failure ever
        # spends the budget, and then it has tried both.
        #
        # With a preferred point -- the object resting on the support -- the
        # first several valid spots of EITHER shape are collected and the one
        # nearest that point wins, so the robot stands where the die is. It
        # used to pool only band spots, which alternation would have broken: a
        # ring spot returns on the spot and the side preference never applies.
        edge_band = eef_pose is None and not self.is_walkable_surface(obj)
        pool = []
        for index in range(attempts):
            from_band = edge_band and index % 2 == 0
            if from_band:
                candidate = self._edge_candidate(obj, prefer_xy)
            else:
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
            blocker = self._blocking_robot(candidate[:2])
            if blocker is not None:
                rejected["robots"] += 1
                blockers[blocker] += 1
                continue
            if prefer_xy is not None:
                pool.append((math.hypot(float(candidate[0]) - float(prefer_xy[0]),
                                        float(candidate[1]) - float(prefer_xy[1])), candidate))
                if len(pool) < 8:
                    continue
                candidate = min(pool, key=lambda item: item[0])[1]
            if self._nav_destinations is not None:
                self._nav_destinations.reserve(self.robot.name, candidate[:2])
            return candidate
        if pool:
            candidate = min(pool, key=lambda item: item[0])[1]
            if self._nav_destinations is not None:
                self._nav_destinations.reserve(self.robot.name, candidate[:2])
            return candidate

        self._nav_last_rejections = dict(rejected)
        self._nav_last_blockers = dict(blockers.most_common())
        # No candidate satisfied all of the filters. Returning None makes
        # _navigate_to_obj raise PLANNING_ERROR -- the honest signal that this
        # target has no standable pose around it. Measured on Rs_int that is
        # 46% of targets, which is a property of the scene, not of this code:
        # see feasibility_verify/measure_teleport_risk.py. Previously the
        # sampler returned the first candidate regardless, so the robot
        # teleported into a wall and toppled instead of reporting failure.
        return None

    #: Ticks a settle may take, at most. Upstream's ``_settle_robot`` is 50
    #: unconditional ticks and then up to MAX_STEPS_FOR_SETTLING (500) more
    #: until the base is still. Every primitive ends in one, several begin
    #: with one, and a failed attempt pays another via ``_reset_robot``. The
    #: symbolic robots teleport and are still at once; the one exception, a
    #: drone that spins in the air once it holds something, never stops and
    #: so paid the full 550 every time (grasp = 501 ticks in the routed
    #: episodes). 10 for everything (user, 2026-09-13).
    MAX_SETTLE_TICKS = 10

    def _settle_robot(self):
        """Hold still for at most :attr:`MAX_SETTLE_TICKS`, stopping early
        once the base is still. Replaces upstream's 50 + 500."""
        if not hasattr(self.robot, "get_linear_velocity"):
            # The CPU fakes: their own scripted settle, which the tests count.
            yield from super()._settle_robot()
            return
        for _ in range(self.MAX_SETTLE_TICKS):
            try:
                if float(th.norm(self.robot.get_linear_velocity())) < 0.01:
                    break
            except Exception:  # noqa: BLE001 - a stub without velocities
                pass
            yield self._postprocess_action(self.robot.q_to_action(self.robot.get_joint_positions()))

    # -- what an object rests on -------------------------------------------

    @staticmethod
    def _aabb_of(obj):
        """``(lo, hi)`` world AABB of @obj, from ``aabb`` when the object has
        one, else from its position and ``aabb_extent`` (the CPU fakes).

        Memoised for the current tick (`geometry_cache`): the scan in
        ``support_of`` asks for the same objects over and over within one
        observation, and recomputing them was most of what an observation
        cost.
        """
        cache = aabb_cache()
        key = getattr(obj, "name", None) or id(obj)
        hit = cache.get(key)
        if hit is not None:
            return hit
        aabb = getattr(obj, "aabb", None)
        if aabb is not None and not callable(aabb):
            lo, hi = aabb
            box = ([float(v) for v in lo], [float(v) for v in hi])
            cache[key] = box
            return box
        position = obj.get_position_orientation()[0]
        extent = getattr(obj, "aabb_extent", None) or (0.0, 0.0, 0.0)
        lo = [float(position[i]) - float(extent[i]) / 2.0 for i in range(3)]
        hi = [float(position[i]) + float(extent[i]) / 2.0 for i in range(3)]
        cache[key] = (lo, hi)
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
        robots = list(getattr(scene, "robots", None) or [])
        if hasattr(obj, "action_dim") or any(obj is r for r in robots):
            # A robot rests on nothing we would stand at -- and a drone with
            # a die welded under it was reported as "resting on dice_154".
            return None
        objects = getattr(scene, "objects", None)
        if not objects:
            return None
        try:
            lo, hi = self._aabb_of(obj)
            x, y = self._object_xy_of(obj)
        except Exception:  # noqa: BLE001
            return None
        bottom = lo[2]
        obj_dx, obj_dy = hi[0] - lo[0], hi[1] - lo[1]
        best, best_top = None, None
        for other in objects:
            if other is obj or getattr(other, "category", None) == "agent":
                continue
            if self.is_walkable_surface(other) or hasattr(other, "action_dim") or any(other is r for r in robots):
                continue
            try:
                o_lo, o_hi = self._aabb_of(other)
            except Exception:  # noqa: BLE001
                continue
            margin = 0.05
            if not (o_lo[0] - margin <= x <= o_hi[0] + margin and o_lo[1] - margin <= y <= o_hi[1] + margin):
                continue
            # A support is at least as big as what rests on it; a die under a
            # drone is not the drone's support.
            if (o_hi[0] - o_lo[0]) < obj_dx or (o_hi[1] - o_lo[1]) < obj_dy:
                continue
            top = o_hi[2]
            # Resting on it: the object's bottom lies within the support's
            # vertical span, and above its base. "Top within 0.15 m of the
            # bottom" was the old test, and a bed fails it -- a bed's AABB top is
            # its headboard, 0.5 m above the mattress the die sits on -- so the
            # die on the C9 bed had no support and fell back to being its own
            # approach target. The base margin keeps a die on the floor from
            # being credited to the table standing over it.
            if not (o_lo[2] + 0.10 <= bottom <= top + 0.15):
                continue
            # Prefer the surface the object actually sits on: a flat top at the
            # object's bottom beats a taller enclosing span; among the rest, the
            # nearest top above.
            gap = top - bottom
            rank = (0, 0.0) if abs(gap) <= 0.15 else (1, gap if gap > 0 else 10.0 + abs(gap))
            if best_top is None or rank < best_top:
                best, best_top = other, rank
        return best

    def reach_across(self, support) -> bool:
        """Can a robot stand at @support's edge and reach anything on it?

        True for a cabinet, a bookcase, a fridge top, a table: their narrower
        half-extent is within arm reach, so the middle is reachable from the
        edge. False for a bed -- 2.1 x 1.7 m, half-extent 0.85 against a 0.6 m
        reach -- and treating it like a cabinet is worse than not recognising
        it: navigating "to the bed" samples a 1.9-2.5 m ring around its centre,
        which in Merom_1_int's bedroom lands in walls and next door (4 of 4
        attempts failed that way on 2026-09-13, room 124 / trav 67 / robots 9).
        An object on a wide support is approached where it is, and is out of
        reach from the floor if it sits mid-mattress -- which is true.
        """
        try:
            extent = support.aabb_extent
            half_min = min(float(extent[0]), float(extent[1])) / 2.0
        except Exception:  # noqa: BLE001
            return True
        return half_min <= self._nav_reach

    def approach_anchor(self, obj):
        """What navigate_to(@obj) samples around and the reach gate measures to.

        The support when the object rests on furniture a robot can reach
        across; otherwise the object itself. One rule for the sampler and for
        ``interaction_radius_for``, so anything NAVIGATE_TO produces is in range.
        """
        support = self.support_of(obj)
        return support if support is not None else obj

    def edge_distance_to(self, obj, xy) -> float:
        """Distance from @xy to @obj's footprint (its xy AABB); 0 inside it."""
        lo, hi = self._aabb_of(obj)
        dx = max(lo[0] - float(xy[0]), 0.0, float(xy[0]) - hi[0])
        dy = max(lo[1] - float(xy[1]), 0.0, float(xy[1]) - hi[1])
        return math.hypot(dx, dy)

    def _edge_candidate(self, obj, prefer_xy=None):
        """One (x, y, yaw) just outside @obj's footprint, facing it.

        A ring around the centre is the wrong shape for anything that is not a
        point: its inner radius is the object's half-DIAGONAL, so for a 2.1 m
        bed it is 1.9-2.5 m out and lands in the walls of any bedroom that fits
        the bed, and for a 1.53 x 0.94 table it is 1.30 m from the centre, out
        past the ends. What a person does is walk up to the side. So: a point
        on the footprint's perimeter, pushed out by the robot's own half-width
        plus up to one arm's reach.

        This is now the only path for a target that is not walked on. It used
        to be reserved for supports too wide to reach across (`reach_across`),
        and measured 2026-09-14 on Merom_1_int that reservation was costing
        real floor: with the band and `body_offset` together, the standable
        fraction goes 0 -> 5% at the breakfast table and 4 -> 16% at the
        armchair for the Ridgeback, 34 -> 44% at the bookcase and 30 -> 41% at
        the bed, with no pose closer than the chassis half-width. Either change
        alone measured nothing -- the band still started at the circumscribed
        radius, and the half-width was still on a ring.

        The band is always inside what the reach gate allows, whatever the
        shape: edge distance never exceeds the half-diagonal, so a pose at
        ``edge + body + margin + reach`` is within
        ``interaction_radius_for``'s ``half_diagonal + body + margin + reach``. With @prefer_xy (the
        thing resting on it) the perimeter point is mirrored onto the object's
        half of the footprint three times in four, so the robot ends up on the
        side the object is on when that side is standable, and elsewhere
        otherwise. Floors never come here: they are walked on, not stood at.
        """
        lo, hi = self._aabb_of(obj)
        w, d = max(hi[0] - lo[0], 1e-3), max(hi[1] - lo[1], 1e-3)
        u = th.rand(1).item() * 2.0 * (w + d)
        if u < w:
            bx, by, nx, ny = lo[0] + u, lo[1], 0.0, -1.0
        elif u < w + d:
            bx, by, nx, ny = hi[0], lo[1] + (u - w), 1.0, 0.0
        elif u < 2 * w + d:
            bx, by, nx, ny = hi[0] - (u - w - d), hi[1], 0.0, 1.0
        else:
            bx, by, nx, ny = lo[0], hi[1] - (u - 2 * w - d), -1.0, 0.0
        if prefer_xy is not None and th.rand(1).item() < 0.75:
            cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
            if nx != 0.0 and (bx - cx) * (float(prefer_xy[0]) - cx) < 0:
                bx, nx = 2 * cx - bx, -nx
            if ny != 0.0 and (by - cy) * (float(prefer_xy[1]) - cy) < 0:
                by, ny = 2 * cy - by, -ny
        out = self.body_offset() + self._nav_clearance_margin + th.rand(1).item() * self._nav_reach
        x, y = bx + nx * out, by + ny * out
        # Same convention as the ring: the angle from the TARGET to the robot,
        # which `_facing_yaw_offset` (pi - mean workspace) then turns to face
        # back at it. Written the other way round -- atan2(by - y, bx - x), the
        # angle from the robot to the target -- this is pi out and the robot
        # stands with its back to the thing it came for. It was wrong here from
        # the start and only the bed reached this path, so nothing caught it
        # until the band became the only path (2026-09-14).
        yaw = math.atan2(y - by, x - bx) + self._facing_yaw_offset()
        return th.tensor([x, y, yaw], dtype=th.float32)

    @staticmethod
    def _object_xy_of(obj):
        position = obj.get_position_orientation()[0]
        return float(position[0]), float(position[1])

    def navigate_to_robot(self, robot):
        """Drive to a spot beside @robot -- a teammate, usually the carrier.

        Upstream's ``apply_ref`` refuses a robot argument outright ("Cannot
        call a symbolic semantic action primitive with a robot as an
        argument"), so the engine dispatches here instead. The same sampler
        as for an object: a robot has an AABB, so the annulus and the reach
        gate come out of ``sampling_range_for`` like anything else, and its
        own body is skipped by ``_clear_of_other_robots`` because the annulus
        starts outside it.
        """
        yield from self._navigate_to_obj(robot)

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

        attempts = self._nav_sampling_attempts or DEFAULT_SAMPLING_ATTEMPTS
        rejected = {"room": 0, "trav": 0, "robots": 0}
        blockers: Counter = Counter()
        for _ in range(attempts):
            xy = self._random_point_in_room(seg_map, room, names[room])
            if xy is None:
                break
            if self._room_of(xy) not in (room, None):
                rejected["room"] += 1
                continue
            if not self._is_traversable(xy):
                rejected["trav"] += 1
                continue
            blocker = self._blocking_robot(xy)
            if blocker is not None:
                rejected["robots"] += 1
                blockers[blocker] += 1
                continue
            if self._nav_destinations is not None:
                self._nav_destinations.reserve(self.robot.name, xy)
            pose = th.tensor([xy[0], xy[1], random.uniform(-math.pi, math.pi)], dtype=th.float32)
            yield from self._navigate_to_pose(pose)
            return
        self._nav_last_rejections = dict(rejected)
        self._nav_last_blockers = dict(blockers.most_common())
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PLANNING_ERROR,
            f"Cannot find a free spot to stand in {room}: there is no free floor space in it "
            "to stand on."
            + _why_no_space(rejected, self._nav_last_blockers),
            {"room": room, "rejected_by": dict(rejected),
             "blocked_by": dict(self._nav_last_blockers), "reason_code": "NO_SPACE_IN_ROOM"},
        )

    @staticmethod
    def _random_point_in_room(seg_map, room: str, ins_id) -> Optional[Tuple[float, float]]:
        """A uniformly random cell of @room, in world metres, or None.

        Not ``seg_map.get_random_point_by_room_instance``: upstream's is
        broken -- ``th.randint(n)`` without a size raises ``TypeError: only
        integer tensors of a single element can be converted to an index``,
        which crashed every room navigation in the first acceptance launch
        (2026-09-13). Same idea, done with ``nonzero`` and Python's RNG; the
        upstream call is kept only as the fallback for maps without a cell
        grid (the CPU fakes).
        """
        import random  # noqa: PLC0415

        cells = getattr(seg_map, "room_ins_map", None)
        if cells is not None:
            try:
                idx = th.nonzero(cells == ins_id)
                if idx.shape[0] == 0:
                    return None
                pick = idx[random.randrange(int(idx.shape[0]))]
                world = seg_map.map_to_world(pick)
                return float(world[0]), float(world[1])
            except Exception:  # noqa: BLE001 - fall through to upstream
                pass
        _, point = seg_map.get_random_point_by_room_instance(room)
        if point is None:
            return None
        return float(point[0]), float(point[1])

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        """Same as upstream, minus the keyword the symbolic override rejects.

        The inherited version forwards ``skip_obstacle_update`` into
        ``_navigate_to_pose``, but the symbolic override's signature is
        ``_navigate_to_pose(self, pose_2d)``, so that call is a ``TypeError``.

        An object resting on furniture is approached by way of the furniture
        (see :meth:`support_of`): the spot is sampled around the support, and
        ``interaction_radius_for`` accepts anything that produces.
        """
        anchor = self.approach_anchor(obj) if eef_pose is None else obj
        support = anchor if anchor is not obj else None
        prefer = obj.get_position_orientation()[0][:2] if support is not None else None
        pose = self._sample_pose_near_object(anchor, eef_pose=eef_pose, prefer_xy=prefer)
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
            # teammates who reserved it first -- and what to do instead, which
            # `_why_no_space` says and which depends on which of the three it
            # was. The annulus itself is not in the text: the agent cannot act
            # on "0.9-1.5 m away" (it does not choose where it stands, the
            # sampler does), and it stays in `sampling_range` for the logs.
            blocked = getattr(self, "_nav_last_blockers", None) or {}
            counts = getattr(self, "_nav_last_rejections", None)
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                f"Cannot reach {obj_label}: there is no free floor space around it to stand on."
                + _why_no_space(counts, blocked),
                {"object": obj.name, "sampling_range": [round(lo, 2), round(hi, 2)],
                 "rejected_by": counts,
                 "blocked_by": dict(blocked),
                 "target_xy": [round(float(v), 2) for v in obj.get_position_orientation()[0][:2]],
                 "reason_code": "NO_SPACE_AROUND_TARGET"},
            )
        yield from self._navigate_to_pose(pose)
