"""Resource contention for the symbolic primitive set.

The symbolic primitives are distance-blind and holder-blind. Every
``_navigate_if_needed`` call in
``SymbolicSemanticActionPrimitives`` is commented out, so:

* ``_grasp`` teleports the object to the end-effector **at any distance**, and
  its only precondition is ``self._get_obj_in_hand() is None`` -- which reads
  ``self.robot._ag_obj_in_hand[arm]``, i.e. *this* robot's own hand. There is
  no cross-robot check anywhere. ``_establish_grasp`` then creates its joint at
  ``{self.eef_links[arm].prim_path}/ag_constraint``, a path scoped to the
  *grasping* robot, so a second robot grasping the same object does not
  collide with the first one's joint: the object is yanked out of A's hand to
  B's end-effector, ends up carrying **two** FixedJoints, and both robots'
  ``_ag_obj_in_hand`` claim to hold it. Both post-condition checks pass. No
  exception, no log -- silent state corruption plus an over-constrained body.
* ``_place_with_predicate`` / ``_open_or_close`` / ``_toggle`` are the same
  story minus the joint: they set the state from across the room.

Consequence: symbolic mode has essentially no resource contention, which is
what COOP2's cooperation pressure is made of. This module puts it back with
three coupled rules:

1. **Interaction radius.** Acting on an object requires the robot base to be
   within ``interaction_radius`` metres of it, else ``TOO_FAR``.
2. **Travel time.** ``NAVIGATE_TO`` yields hold-position actions proportional
   to the distance actually travelled *before* teleporting, so being far away
   costs ticks.
3. **Claims.** Acting on an object another robot is holding fails with
   ``OBJECT_CLAIMED`` instead of corrupting the scene.

Together these make "who gets there first" depend on where the agents started,
which is exactly the allocation problem the centralized leader and the
broadcast chain are supposed to solve -- and going after a teammate's target is
a measurable net loss (travel ticks burned, then a failure).

Deliberately *not* a lock in the engine: the loser still burns a full navigate
and still issues its GRASP, so contention stays visible to the cognitive layer
as a wasted decision. A pre-assignment lock would hide the very signal the
topology layer is being measured on. See PORTING_PLAN.md 3.4 / 4.2 and the
"Deliberately not implemented" section of coop2/CLAUDE.md.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError

from coop2.behavior_env.symbolic_navigation import NavigableSymbolicActionPrimitives

__all__ = [
    "ContentiousSymbolicActionPrimitives",
    "DEFAULT_GATED_PRIMITIVES",
    "GATE_GRASP",
    "GATE_OPEN_CLOSE",
    "GATE_PLACE",
    "GATE_CARRY",
    "GATE_TOGGLE",
]

#: Gate labels, so callers can switch individual rules off without subclassing.
GATE_GRASP = "grasp"
GATE_PLACE = "place"
GATE_OPEN_CLOSE = "open_close"
GATE_TOGGLE = "toggle"
#: Loading onto and unloading from a carrier robot gate on reach like any
#: other manipulation.
GATE_CARRY = "carry"

DEFAULT_GATED_PRIMITIVES = frozenset(
    {GATE_GRASP, GATE_PLACE, GATE_OPEN_CLOSE, GATE_TOGGLE, GATE_CARRY}
)

#: Headroom between the navigation sampler's upper bound and the interaction
#: radius. Without it a pose sampled exactly at the far edge of the annulus
#: would be borderline TOO_FAR, and the agent would loop
#: navigate -> TOO_FAR -> navigate on a float comparison.
#:
#: This is a **guard, not a policy**: it covers the float comparison and the
#: millimetres a base drifts during the settle after a teleport, nothing more.
#: It was 0.35, which quietly added a third of a metre to how far away every
#: agent could manipulate; the distance an agent is allowed to reach across is
#: ``reach`` alone, so that one number governs it (user, 2026-09-09). Every
#: non-navigation verb -- grasp, place, open/close, toggle -- goes through
#: ``interaction_radius_for`` and therefore shares it.
DEFAULT_RADIUS_MARGIN = 0.05

#: Ticks of travel per metre. One env step is 1/``action_frequency`` seconds
#: (30 Hz by default), so 30 ticks/m is 1 s/m, i.e. a 1.0 m/s base. Tune it as
#: an experiment variable: this number sets how expensive distance is relative
#: to a decision.
#:
#: **30, chosen for speed with the cost known** (user, 2026-09-11). navigate_to
#: is 77 % of all primitive work in the nine-apple hall, and this constant, not
#: the engine, is what sets how long a run takes. The price is fidelity: 1.0 m/s
#: is about twice a real R1, and 60 ticks/m (0.5 m/s) is what the physical
#: primitives actually achieve. It was set to 30 on 2026-09-10, reverted to 60
#: the same day for that reason, and set back here -- the experiments are the
#: thing being bought, and they were too slow to run.
#:
#: Two measurements worth keeping either way:
#:
#: * The total navigate_to does **not** halve. Each one also pays a settle floor
#:   of ~100 ticks: 213/165/243/173 ticks for 2.6-3.7 m hops at 30/m.
#: * A shorter episode is better bought by making the task's objects less
#:   scattered than by making the robot faster than it is. The nine-apple hall
#:   puts chairs 4.7-43 m from the table; that, not the speed, is why a run
#:   needs tens of thousands of steps.
#:
#: Any metric recorded at 60 ticks/m is **not comparable** with one recorded
#: here -- travel is most of the tick budget, so every step count moves.
DEFAULT_TRAVEL_TICKS_PER_METER = 30.0

#: Ticks a ``wait`` holds for when the agent does not say. Long enough that a
#: teammate's NAVIGATE_TO (300-500 ticks here) makes real progress during it.
DEFAULT_WAIT_TICKS = 200

#: Cap on a single wait. An agent that yields the floor for the rest of the
#: episode cannot react to the thing it was waiting for.
MAX_WAIT_TICKS = 600



#: Mirrors ``omnigibson.controllers.IsGraspingState``: TRUE = 1, UNKNOWN = 0,
#: FALSE = -1. Spelled out as an int rather than imported because the CPU test
#: suites load this module under a stubbed ``omnigibson``, and a check that
#: cannot run in the stubs is a check no test can cover.
GRASPING_TRUE = 1


def is_definitely_grasping(state) -> bool:
    """``True`` only for ``IsGraspingState.TRUE``.

    ``Robot.is_grasping`` returns an IntEnum with ``FALSE = -1``, so a bare
    ``if state:`` is true for a definite **no** and false for "don't know". And
    FALSE is not a rare answer: the gripper controller reports TRUE whenever it
    is closed on something, then downgrades to FALSE if the fingers are not in
    contact with the object you asked about (``robots/robot.py``). A robot
    holding one apple therefore answered "yes, I hold it" about every other
    object in the scene, and the second apple went permanently OBJECT_CLAIMED
    the instant the first was picked up -- while ``place_on_top`` failed
    PRE_CONDITION, because nothing was really in the hand.

    Accepts the enum, a plain int, or a bool (``True`` is ``1``). Anything
    unreadable counts as *not* grasping: this gates a claim that blocks another
    agent, so an answer we cannot interpret must not create one.
    """
    if state is None or isinstance(state, str):
        return False
    try:
        return int(state) == GRASPING_TRUE
    except (TypeError, ValueError):
        return False

class ContentiousSymbolicActionPrimitives(NavigableSymbolicActionPrimitives):
    """Symbolic primitives with proximity, travel cost, and cross-robot claims.

    Drop-in replacement for :class:`NavigableSymbolicActionPrimitives`.

    Args:
        env, robot: as the base class.
        interaction_radius: fixed metres within which an object may be
            grasped / placed on / opened / toggled. ``None`` (the default and
            the sane choice) derives it **per object** from the navigation
            sampler's own range -- see :meth:`interaction_radius_for`.
        travel_ticks_per_meter: hold-position ticks emitted per metre of
            straight-line base travel before the teleport. 0 disables the
            travel cost.
        travel_ticks_range: ``(min, max)`` clamp on the emitted travel ticks.
            The max keeps a cross-scene navigate from eating the episode
            budget; the min gives every navigate a floor cost so that
            "navigate to where I already am" is not free.
        gated_primitives: which proximity gates to enforce. See
            :data:`DEFAULT_GATED_PRIMITIVES`.
        enforce_claims: reject actions on an object another robot holds.
        distance_range, sampling_attempts, require_same_room: passed to
            :class:`NavigableSymbolicActionPrimitives`.
    """

    def __init__(
        self,
        env,
        robot,
        interaction_radius: Optional[float] = None,
        travel_ticks_per_meter: float = DEFAULT_TRAVEL_TICKS_PER_METER,
        travel_ticks_range: Tuple[int, int] = (0, 3000),
        gated_primitives: Iterable[str] = DEFAULT_GATED_PRIMITIVES,
        enforce_claims: bool = True,
        **kwargs,
    ):
        super().__init__(env, robot, **kwargs)

        travel_min, travel_max = travel_ticks_range
        if travel_min < 0 or travel_max < travel_min:
            raise ValueError(f"travel_ticks_range must be a non-negative (min, max), got {travel_ticks_range!r}")

        self._interaction_radius = None if interaction_radius is None else float(interaction_radius)
        self.travel_ticks_per_meter = float(travel_ticks_per_meter)
        self.travel_ticks_range = (int(travel_min), int(travel_max))
        self.gated_primitives = frozenset(gated_primitives)
        self.enforce_claims = bool(enforce_claims)

    def interaction_radius_for(self, obj) -> float:
        """How close the base must be to act on @obj. One rule for every verb.

        Every non-navigation primitive -- grasp, place, open/close, toggle --
        gates on this, so "how far can an agent reach" is a single number:
        ``reach``, plus a small guard (:data:`DEFAULT_RADIUS_MARGIN`) for the
        float comparison and settle drift.

        What is uniform is the **clear floor between the robot's edge and the
        object's**, not the centre-to-centre distance, and it cannot be the
        latter: the coffee table's own clearance is 1.48 m, so a uniform
        centre-distance small enough to be meaningful for an apple would make
        the table unreachable at any pose. So the radius is
        ``clearance(obj) + reach + guard`` and the *gap* is what stays constant
        across objects.

        Derived from the navigation sampler's range for this very object, so the
        two can never disagree: anything NAVIGATE_TO can produce is in range. An
        agent that navigated successfully must never then be told TOO_FAR, or it
        loops navigate -> TOO_FAR -> navigate forever.
        """
        if self._interaction_radius is not None:
            return self._interaction_radius
        support = self.support_of(obj)
        if support is not None:
            # Standing at the support, reaching across it: navigate_to(obj)
            # samples around the support, so the robot ends within the
            # support's own radius of the support's centre, and the object is
            # at most the support's half-diagonal beyond that. The gate is the
            # sum, measured to the object, so it accepts every spot the
            # sampler can produce -- the same contract as the plain case.
            try:
                extent = support.aabb_extent
                half_diagonal = math.hypot(float(extent[0]), float(extent[1])) / 2.0
            except Exception:  # noqa: BLE001
                half_diagonal = 0.0
            return self.sampling_range_for(support)[1] + half_diagonal + DEFAULT_RADIUS_MARGIN
        return self.sampling_range_for(obj)[1] + DEFAULT_RADIUS_MARGIN

    # -- world queries ----------------------------------------------------

    def _peer_robots(self) -> Sequence[Any]:
        """Every robot in the scene, this one included.

        Prefers ``robot.scene.robots`` over ``env.robots`` so the class still
        works when driven without an Environment (tests, direct scene use).
        """
        scene = getattr(self.robot, "scene", None)
        robots = getattr(scene, "robots", None)
        if robots is None:
            robots = getattr(self.env, "robots", None)
        return list(robots or [])

    def holder_of(self, obj) -> Optional[Any]:
        """The robot currently holding ``obj``, or None.

        Asks every robot's public ``is_grasping(arm, candidate_obj)``. This is
        the cross-robot view ``_get_obj_in_hand`` does not give: that one is
        indexed by ``self.robot`` and ``self.arm`` only. Going through the
        public API also picks up the ``grasping_mode == "physical"`` branch that
        reading ``_ag_obj_in_hand`` directly would skip.

        **Compare against TRUE explicitly.** ``is_grasping`` returns an
        ``IsGraspingState``, an IntEnum whose members are ``TRUE = 1``,
        ``UNKNOWN = 0`` and ``FALSE = -1`` -- so a bare truthiness test is true
        for a definite *no* and false for "don't know". And FALSE is exactly what
        comes back in the case that matters: the controller reports TRUE whenever
        the gripper is closed on something, then downgrades to FALSE if the
        fingers are not in contact with ``candidate_obj``
        (``robots/robot.py``). So a robot holding apple 1 answered "yes, I hold
        apple 2" for every other object in the scene, and the second apple became
        permanently OBJECT_CLAIMED the instant the first was picked up -- with
        ``place_on_top`` then failing PRE_CONDITION because nothing was really in
        the hand. Latent until the robots came up with a working controller
        stack; before that ``is_grasping`` raised and the ``break`` hid it.
        """
        for robot in self._peer_robots():
            for arm in getattr(robot, "arm_names", []):
                try:
                    if is_definitely_grasping(robot.is_grasping(arm=arm, candidate_obj=obj)):
                        return robot
                except Exception:  # noqa: BLE001 - non-manipulation robots
                    break
        return self._carrier_of(obj)

    def _carrier_of(self, obj):
        """The carrier @obj is riding on, or None.

        A hand is not the only way to have an object. Cargo is welded to a
        carrier's base, so no gripper reports it and every gate that asks
        "is anyone holding this" would answer no -- and then a grasp welds a
        second joint to an object that already has one, which is the silent
        corruption `_grasp`'s claim check exists to stop. Taking cargo back is
        `unload_from`, and it has to be the only way.
        """
        try:
            from coop2.behavior_env.carrier import carrier_holding  # noqa: PLC0415

            return carrier_holding(obj, self._peer_robots())
        except Exception:  # noqa: BLE001 - a scene with no carrier support
            return None

    def _base_xy(self) -> Tuple[float, float]:
        position = self.robot.get_position_orientation()[0]
        return float(position[0]), float(position[1])

    def _object_xy(self, obj) -> Tuple[float, float]:
        position = obj.get_position_orientation()[0]
        return float(position[0]), float(position[1])

    def distance_to(self, obj) -> float:
        """Planar base-to-object distance.

        Planar and base-relative on purpose: NAVIGATE_TO moves the *base* in
        the xy plane, so gating on anything else (end-effector, 3D distance)
        would make a successful navigate a non-guarantee of reachability.
        """
        base_x, base_y = self._base_xy()
        obj_x, obj_y = self._object_xy(obj)
        return math.hypot(obj_x - base_x, obj_y - base_y)

    # -- gates ------------------------------------------------------------

    def _error(self, reason_code: str, message: str, metadata: dict) -> ActionPrimitiveError:
        """A PRE_CONDITION error tagged with a COOP2 reason code.

        ``ReasonCode.from_primitive_error`` reads ``metadata['reason_code']``
        in preference to the ``ActionPrimitiveError.Reason`` enum, which only
        has five members and cannot express these two.
        """
        return ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            message,
            dict(metadata, reason_code=reason_code),
        )

    def _require_near(self, obj, verb: str, gate: str) -> None:
        if gate not in self.gated_primitives:
            return
        distance = self.distance_to(obj)
        radius = self.interaction_radius_for(obj)
        if distance <= radius:
            return
        raise self._error(
            "TOO_FAR",
            f"You are {distance:.2f} m from {obj.name}, too far to {verb} it "
            f"(you must be within {radius:.2f} m). Navigate to it first.",
            {"target object": obj.name, "distance": round(distance, 3), "radius": round(radius, 3)},
        )

    def _require_not_held_by_self(self, obj) -> None:
        """Refuse to grasp what is already in this robot's own gripper.

        Upstream re-grasps happily: the primitive settles for fifty ticks and
        reports success without changing anything. An agent that had achieved
        its goal therefore kept proposing ``holding(apple)`` and kept being
        told it had succeeded -- nineteen times in one episode -- because the
        environment never contradicted it. A no-op that scores as a success
        both wastes the agent's decisions and inflates Y_plan, so it fails, and
        the failure sends the agent back to reasoning where it can see that it
        is already holding the thing and pick something else to do.
        """
        if self.holder_of(obj) is self.robot:
            raise self._error(
                "ALREADY_HELD",
                f"You are already holding {obj.name}; grasping it again does nothing. "
                "If this was your goal, it is met -- choose a different task.",
                {"target object": obj.name},
            )

    def _require_unclaimed(self, obj, verb: str) -> None:
        if not self.enforce_claims:
            return
        holder = self.holder_of(obj)
        if holder is None or holder is self.robot:
            return
        # The holder's name is the agent id the cognitive layer knows it by,
        # and this prose goes straight into the prompt -- it is the main way a
        # decentralized agent finds out it needs to negotiate.
        #
        # Cargo says something different, because the remedy is different: a
        # box on a carrier's back is not going to be released, it is going to
        # be unloaded, and by the robot that wants it.
        if self._carrier_of(obj) is holder:
            raise self._error(
                "OBJECT_CLAIMED",
                f"{obj.name} is riding on {holder.name}'s back, so you cannot "
                f"{verb} it. Take it with unload_from({holder.name}) instead -- "
                "you have to be within reach of the carrier.",
                {"target object": obj.name, "carried by": holder.name},
            )
        raise self._error(
            "OBJECT_CLAIMED",
            f"{obj.name} is currently held by {holder.name}, so you cannot {verb} it. "
            "Pick a different target, or ask them to release it.",
            {"target object": obj.name, "held by": holder.name},
        )

    def _hold_action(self):
        """One "keep the current joint configuration" action.

        Identical to what ``_settle_robot`` yields, so the travel padding is
        indistinguishable from settling as far as the controller is concerned.
        """
        return self._postprocess_action(self.robot.q_to_action(self.robot.get_joint_positions()))

    def wait(self, ticks: Optional[int] = None):
        """Hold position for @ticks, then finish. A real primitive on purpose.

        ``wait`` used to be a "communication action": no primitive, completed
        the instant it was issued. That made it worse than useless. The plan
        loop freezes physics whenever any agent is not ready -- it returns
        without stepping the env -- so an instant wait put the agent straight
        back into reasoning, which stopped the world, which meant the teammate
        it was waiting for advanced by exactly nothing while the wait burned an
        LLM call. Yielding the floor is only meaningful if time passes, and time
        only passes while some agent holds an active primitive.

        Args:
            ticks: how long to hold. Defaults to DEFAULT_WAIT_TICKS, clamped to
                MAX_WAIT_TICKS.
        """
        held = DEFAULT_WAIT_TICKS if ticks is None else int(ticks)
        held = max(1, min(held, MAX_WAIT_TICKS))
        for _ in range(held):
            yield self._hold_action()

    # -- carrying ----------------------------------------------------------

    def _require_base_free_to_move(self) -> None:
        """Refuse to drive while holding, for a robot whose base locks.

        A suction arm on a mobile base cannot do both at once: V4 states this
        outright (`v4:base_locked_while_holding`), and it is what makes its
        tasks need more than one robot. Without it the robot that picks the box
        up also delivers it, and the team is one worker and some spectators.

        Only robots that declare the constraint are affected, so nothing that
        worked before changes.
        """
        if not getattr(self.robot, "base_locked_while_holding", False):
            return
        held = self._get_obj_in_hand()
        if held is None:
            return
        raise self._error(
            "BASE_LOCKED",
            f"You cannot drive while holding {held.name}: your base is locked "
            f"whenever your arm is loaded. Put it down, or load it onto a "
            f"carrier robot with load_onto, and let the carrier drive.",
            {"held object": held.name},
        )

    def _require_arm(self, verb: str) -> None:
        """Refuse a hand verb to a robot that has no hand.

        A carrier is defined by not being able to manipulate -- that is what
        makes it a carrier rather than a second arm. Without this the refusal
        still happens but somewhere useless: `load_onto` reports "you are not
        holding anything to load" (true, and not the reason), and `_grasp`
        fails inside the parent on an arm that is not there. The prompt does not
        offer these verbs to a carrier; this is the half that makes that promise
        safe to rely on.
        """
        try:
            from coop2.behavior_env.carrier import is_carrier  # noqa: PLC0415

            armless = is_carrier(self.robot)
        except Exception:  # noqa: BLE001 - a scene with no carrier support
            armless = False
        if not armless:
            return
        raise self._error(
            "NO_ARM",
            f"You have no arm, so you cannot {verb}. You can navigate_to and "
            f"wait. Cargo is put on your back and taken off it by a robot that "
            f"does have one -- load_onto({self.robot.name}) and "
            f"unload_from({self.robot.name}), performed by that robot.",
            {},
        )

    def _require_may_lift(self, obj, verb: str) -> None:
        """Refuse to take @obj into the hand when this robot's role may not.

        The route file's `lift` table (route_spec) is keyed by cargo synset and
        lists the roles allowed; `coop_env` hands it to every controller as
        `lift_rules`. COOHAVIOR's contract is the reason it exists: the drone
        may carry the 8 g box and is "not a load-bearing role for the 20 g
        box", which is what makes the heavy tasks need an arm. A synset the
        table does not name may be lifted by anyone with a hand, so a run
        without a route file is unchanged. Refused with CANNOT_LIFT so the
        prompt can say why, the way TOO_FAR and BASE_LOCKED do.
        """
        rules = getattr(self, "lift_rules", None) or {}
        if not rules:
            return
        synset = self._synset_of(obj)
        allowed = rules.get(synset)
        if allowed is None:
            return
        try:
            from coop2.behavior_env.carrier import lift_role  # noqa: PLC0415

            role = lift_role(self.robot)
        except Exception:  # noqa: BLE001
            role = "arm"
        if role in allowed:
            return
        raise self._error(
            "CANNOT_LIFT",
            f"{obj.name} is too heavy for you: a {role} may not {verb} a {synset}; "
            f"only {', '.join(allowed)} may. Leave it to a robot that can.",
            {"target object": obj.name, "synset": synset, "role": role,
             "allowed": list(allowed)},
        )

    @staticmethod
    def _synset_of(obj) -> str:
        """The BDDL synset of @obj's category (``notebook`` -> ``notebook.n.01``),
        or the category itself when the taxonomy does not know it."""
        category = getattr(obj, "category", None) or ""
        try:
            from bddl.object_taxonomy import ObjectTaxonomy  # noqa: PLC0415

            return ObjectTaxonomy().get_synset_from_category(category) or category
        except Exception:  # noqa: BLE001
            return category

    def _require_carrier(self, carrier, verb: str):
        """@carrier must be a robot that carries cargo, and within reach."""
        from coop2.behavior_env.carrier import is_carrier  # noqa: PLC0415

        if not getattr(carrier, "is_robot", False) and carrier not in self._peer_robots():
            raise self._error(
                "INVALID_TARGET",
                f"{getattr(carrier, 'name', carrier)} is not a robot, so there is "
                f"nothing to {verb}.",
                {"target object": getattr(carrier, "name", str(carrier))},
            )
        if not is_carrier(carrier):
            raise self._error(
                "INVALID_TARGET",
                f"{carrier.name} has an arm of its own and carries things in it, "
                f"not on its back.",
                {"target object": carrier.name},
            )
        if carrier is self.robot:
            raise self._error(
                "INVALID_TARGET",
                "You cannot load onto yourself.",
                {"target object": carrier.name},
            )
        self._require_near(carrier, verb, GATE_CARRY)

    def load_onto(self, carrier):
        """Put what this robot is holding onto @carrier's back."""
        from coop2.behavior_env.carrier import carried_by, load_onto  # noqa: PLC0415

        self._require_arm("load onto a carrier")
        self._require_carrier(carrier, "load onto")
        held = self._get_obj_in_hand()
        if held is None:
            raise self._error(
                "PRE_CONDITION_ERROR",
                "You are not holding anything to load.",
                {"target object": carrier.name},
            )
        riding = carried_by(carrier)
        if riding is not None:
            raise self._error(
                "PRE_CONDITION_ERROR",
                f"{carrier.name} is already carrying {riding.name}.",
                {"target object": carrier.name, "carrying": riding.name},
            )
        for arm in self.robot.arm_names:
            self.robot.release_grasp_immediately(arm=arm)
        load_onto(carrier, held)
        yield from self._settle_robot()

    def unload_from(self, carrier):
        """Take what is riding on @carrier into this robot's hand."""
        from coop2.behavior_env.carrier import carried_by, unload_from  # noqa: PLC0415

        self._require_arm("unload from a carrier")
        self._require_carrier(carrier, "unload from")
        if self._get_obj_in_hand() is not None:
            raise self._error(
                "PRE_CONDITION_ERROR",
                "Your gripper is already full.",
                {"target object": carrier.name},
            )
        riding = carried_by(carrier)
        if riding is None:
            raise self._error(
                "PRE_CONDITION_ERROR",
                f"{carrier.name} is not carrying anything.",
                {"target object": carrier.name},
            )
        # Unloading puts the cargo in this robot's hand, so it is a lift too.
        self._require_may_lift(riding, "unload")
        unload_from(carrier)
        # The same weld a grasp makes, now on this robot's own end effector
        # (under it, for a drone -- see _hold_position).
        hold = self._hold_position(riding)
        riding.set_position_orientation(position=hold)
        try:
            riding.keep_still()
        except Exception:  # noqa: BLE001
            pass
        self.robot._establish_grasp(riding, riding.root_link_name, self.arm, hold, "FixedJoint")
        yield from self._settle_robot()

    def travel_ticks(self, distance: float) -> int:
        """Hold-position ticks charged for travelling ``distance`` metres."""
        ticks = int(round(max(0.0, distance) * self.travel_ticks_per_meter))
        low, high = self.travel_ticks_range
        return max(low, min(high, ticks))

    # -- overrides --------------------------------------------------------

    def _navigate_to_pose(self, pose_2d):
        """Charge travel time, then teleport.

        The padding must come **before** ``super()._navigate_to_pose``, which
        teleports on its first ``next()``: pad afterwards and the robot arrives
        instantly and then idles, so a teammate inspecting the world during
        those ticks sees it already at the destination and distance stops being
        a scarce resource. Padding first models "in transit".

        Measuring here rather than in the engine also means the distance is
        read before the teleport for free.
        """
        self._require_base_free_to_move()
        start_x, start_y = self._base_xy()
        distance = math.hypot(float(pose_2d[0]) - start_x, float(pose_2d[1]) - start_y)
        ticks = self.travel_ticks(distance)
        # One line per navigate -- there are tens per episode, not thousands.
        # Without it a 1435-tick NAVIGATE_TO to a target 3.7 m away is only
        # visible after the run, as a number with no way to attribute it
        # between travel padding and the settle loop.
        print(f"[nav] {getattr(self.robot, 'name', '?')} -> "
              f"({float(pose_2d[0]):.2f}, {float(pose_2d[1]):.2f}) "
              f"{distance:.1f} m, {ticks} travel ticks")
        for _ in range(ticks):
            yield self._hold_action()
        yield from self._teleport_with_attachments(pose_2d)

    def _attachments(self):
        """Bodies welded to this robot right now: what is in its hand, and
        what rides on its back if it is a carrier."""
        out = []
        if getattr(self.robot, "arm_names", None) and hasattr(self.robot, "_ag_obj_in_hand"):
            # A carrier has no hand to ask about.
            held = self._get_obj_in_hand()
            if held is not None:
                out.append(held)
        try:
            from coop2.behavior_env.carrier import carried_by  # noqa: PLC0415

            cargo = carried_by(self.robot)
            if cargo is not None:
                out.append(cargo)
        except Exception:  # noqa: BLE001
            pass
        return out

    def _teleport_with_attachments(self, pose_2d):
        """Upstream's ``_navigate_to_pose`` -- set the pose, settle -- plus one
        thing it never needed: the welded bodies move with the robot.

        A FixedJoint drags its child after a teleported parent only through the
        solver, over ticks. With the 550-tick settle that always finished; with
        the 10-tick cap (2026-09-13) a Jackal's die was measured 0.76 m behind
        the Jackal when the primitive returned. So the cargo and the held object
        are moved by the same rigid transform as the robot, and stilled.
        """
        attachments = [(obj, obj.get_position_orientation()) for obj in self._attachments()]
        before_pos, before_orn = self.robot.get_position_orientation()
        # Upstream teleports on the first next() and then settles; take that
        # first step, move the attachments, then hand the rest through. This
        # keeps the teleport itself upstream's (and the CPU fakes').
        teleport = super()._navigate_to_pose(pose_2d)
        first = next(teleport, None)
        if attachments:
            after_pos, after_orn = self.robot.get_position_orientation()
            try:
                import importlib  # noqa: PLC0415

                T = importlib.import_module("omnigibson.utils.transform_utils")
            except Exception:  # noqa: BLE001 - the CPU stubs: translate only
                T = None
            for obj, (obj_pos, obj_orn) in attachments:
                try:
                    if T is not None:
                        rel_pos, rel_orn = T.relative_pose_transform(obj_pos, obj_orn, before_pos, before_orn)
                        new_pos, new_orn = T.pose_transform(after_pos, after_orn, rel_pos, rel_orn)
                        obj.set_position_orientation(position=new_pos, orientation=new_orn)
                    else:
                        shift = [float(after_pos[i]) - float(before_pos[i]) for i in range(3)]
                        obj.set_position_orientation(
                            position=type(obj_pos)([float(obj_pos[i]) + shift[i] for i in range(3)]))
                    obj.keep_still()
                except Exception as error:  # noqa: BLE001 - never let bookkeeping kill a navigate
                    print(f"[nav] could not move {getattr(obj, 'name', obj)} with {self.robot.name}: {error}")
        if first is not None:
            yield first
        yield from teleport

    # Each override below is a generator function, so the body -- and the gate
    # -- runs on the first next(), which is before the parent has touched the
    # scene. A rejected primitive therefore costs 0 ticks and moves nothing.

    def _hold_position(self, obj):
        """Where @obj sits once this robot holds it.

        Upstream puts the object's centre AT the end-effector link's origin.
        For a gripper that is between the fingers; for the Crazyflie's bottom
        suction mount it is inside the mount's collision shape, and the two
        bodies -- welded together and interpenetrating -- push on each other
        every tick. Measured 2026-09-13: angular velocity 0.0 before the grasp,
        3.6 rad/s one tick after, 4.4 rad/s for ever after, the die orbiting
        at 0.9 m/s. That spin is what the videos showed as a drone flinging
        its die about, and why a drone's settle never converged. A drone
        holds its load *under* the mount: half the mount's height plus half
        the object's, plus 5 mm of daylight. Everyone else is unchanged.
        """
        eef = self.robot.get_eef_position(self.arm)
        try:
            from coop2.behavior_env.carrier import lift_role  # noqa: PLC0415

            if lift_role(self.robot) != "drone":
                return eef
        except Exception:  # noqa: BLE001
            return eef
        try:
            mount_half = float(self.robot.eef_links[self.arm].aabb_extent[2]) / 2.0
        except Exception:  # noqa: BLE001
            mount_half = 0.02
        try:
            obj_half = float(obj.aabb_extent[2]) / 2.0
        except Exception:  # noqa: BLE001
            obj_half = 0.02
        drop = mount_half + obj_half + 0.005
        try:
            import torch as th  # noqa: PLC0415

            return th.tensor([float(eef[0]), float(eef[1]), float(eef[2]) - drop], dtype=th.float32)
        except Exception:  # noqa: BLE001 - the CPU stubs have no torch
            return type(eef)([float(eef[0]), float(eef[1]), float(eef[2]) - drop])

    def _grasp(self, obj):
        self._require_arm("grasp")
        self._require_may_lift(obj, "grasp")
        self._require_not_held_by_self(obj)
        self._require_unclaimed(obj, "grasp")
        self._require_near(obj, "grasp", GATE_GRASP)
        if not hasattr(self.robot, "get_eef_position"):
            yield from super()._grasp(obj)
            return
        # Upstream's body, with the object at _hold_position instead of the
        # eef origin; the joint is anchored at the same point.
        hold = self._hold_position(obj)
        obj.set_position_orientation(position=hold)
        try:
            obj.keep_still()
        except Exception:  # noqa: BLE001
            pass
        self.robot._establish_grasp(obj, obj.root_link_name, self.arm, hold, "FixedJoint")
        yield from self._settle_robot()
        if self._get_obj_in_hand() is None:
            raise self._error("EXECUTION_ERROR", f"Grasp of {obj.name} did not take.",
                              {"target object": obj.name})

    def _place_with_predicate(self, obj, predicate, near_poses=None, near_poses_threshold=None):
        """Upstream's placement, with the object's velocity zeroed on arrival.

        obj is the *reference* here (the table), not the thing in hand.

        Reimplemented rather than delegated, for two changes to the order.

        Upstream does ``_release()`` -> ``set_position_orientation`` -> settle,
        and ``_release()`` ends in a settle of its own. So the object spends
        MAX_STEPS_FOR_SETTLING ticks falling out of the gripper before it is
        moved: on screen it drops to the floor and only then jumps onto the
        table. Here the release is split -- detach, teleport, settle once -- so
        the object goes straight to where it was placed, and a whole settle
        comes off the cost of every placement.

        And ``set_position_orientation`` moves a body without touching its
        velocity, so the object used to arrive carrying that fall and the settle
        integrated it: one recorded episode had a task apple leave the house
        entirely, logged at 12 m, then 27 m, then 36 m, then 38 m from the living
        room, after which every navigate_to it failed with
        NO_SPACE_AROUND_TARGET (200/200 candidates rejected by the same-room
        filter) -- which reads as a crowding problem and is really a missing
        object. ``keep_still()`` is upstream's own helper for exactly this.
        """
        self._require_arm("place onto")
        self._require_unclaimed(obj, "place onto")
        self._require_near(obj, "place onto", GATE_PLACE)

        from omnigibson.action_primitives.action_primitive_set_base import (  # noqa: PLC0415
            ActionPrimitiveError as _Error,
        )

        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise _Error(
                _Error.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping an object first to place it somewhere.",
            )

        obj_pose = self._sample_pose_with_object_and_predicate(
            predicate, obj_in_hand, obj,
            near_poses=near_poses, near_poses_threshold=near_poses_threshold,
        )

        # Detach, then move, then settle -- once. Upstream calls ``_release()``,
        # which is ``release_grasp_immediately`` followed by its own
        # ``_settle_robot()``, and only teleports afterwards. Those settle steps
        # are the object in free fall from gripper height: it visibly drops to
        # the floor and *then* jumps onto the table. Splitting the release lets
        # the object go straight there, and drops a whole settle
        # (MAX_STEPS_FOR_SETTLING) from every placement.
        for arm in self.robot.arm_names:
            self.robot.release_grasp_immediately(arm=arm)

        obj_in_hand.set_position_orientation(*obj_pose)
        # Teleporting does not zero velocity, and the object carries whatever it
        # picked up in the gripper; the settle below would otherwise integrate it.
        obj_in_hand.keep_still()
        yield from self._settle_robot()

        if not obj_in_hand.states[predicate].get_value(obj):
            raise _Error(
                _Error.Reason.EXECUTION_ERROR,
                f"Failed to place {obj_in_hand.name} onto {obj.name}: it did not come to rest "
                "there. It has been released, so grasp it again before retrying.",
                {"dropped object": obj_in_hand.name, "target object": obj.name},
            )

    def _open_or_close(self, obj, should_open):
        verb = "open" if should_open else "close"
        self._require_arm(verb)
        self._require_unclaimed(obj, verb)
        self._require_near(obj, verb, GATE_OPEN_CLOSE)
        yield from super()._open_or_close(obj, should_open)

    def _toggle(self, obj, value):
        verb = "toggle on" if value else "toggle off"
        self._require_arm(verb)
        self._require_unclaimed(obj, verb)
        self._require_near(obj, verb, GATE_TOGGLE)
        yield from super()._toggle(obj, value)
