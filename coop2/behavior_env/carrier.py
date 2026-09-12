"""Carrying an object *on* another robot, rather than in a hand.

The V4 tasks are built around a division of labour our action vocabulary could
not express. A Ridgeback's suction arm can pick a box up but its base is locked
while it holds one, so it cannot also carry it anywhere; a Jackal has no arm at
all but has a cargo plate and can drive. Getting a box across the flat therefore
takes both of them, and a handoff in each direction:

    ridgeback   grasp(box) -> load_onto(jackal)
    jackal      navigate_to(destination)
    ridgeback   unload_from(jackal) -> place_on_top(floor)

Without this the "team" is one robot doing the whole job while the others watch,
which is what the first solved run of v4_s1_v4_ll actually was.

Mechanically a load is the same thing a grasp is -- a FixedJoint between a link
and the object -- which is why it survives the carrier driving off. The
difference is only which link: the eef for a grasp, the carrier's base for a
load. The joint is created the way `Robot._create_assisted_grasp_joint` creates
its own, and removed the same way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch as th

__all__ = [
    "CARGO_JOINT_NAME",
    "carried_by",
    "carrier_holding",
    "clear_cargo",
    "is_carrier",
    "load_onto",
    "unload_from",
]

#: Prim name for the joint that holds cargo down, under the carrier's base link.
CARGO_JOINT_NAME = "cargo_constraint"

#: ``{carrier robot name: (joint prim path, carried object)}``.
#:
#: Module level because this is a fact about the scene, not about any one
#: robot: the robot that loads a box is usually not the one that unloads it, and
#: neither is the carrier. Keyed by name so a stale handle cannot resurrect a
#: joint that a reset removed.
_CARGO: Dict[str, Tuple[str, Any]] = {}


def is_carrier(robot) -> bool:
    """Can @robot carry cargo on its back?

    True for a robot that cannot manipulate: it has no other way to move an
    object, and having a flat top is what makes it useful to a team. A robot
    with an arm carries things in it.
    """
    if robot is None:
        return False
    if getattr(robot, "is_carrier", None) is True:
        return True
    return getattr(robot, "is_manipulation", True) is False


def carried_by(carrier) -> Optional[Any]:
    """The object riding on @carrier, or None."""
    name = getattr(carrier, "name", None)
    entry = _CARGO.get(name)
    return entry[1] if entry else None


def carrier_holding(obj, robots) -> Optional[Any]:
    """The carrier @obj is riding on, or None. The cross-robot view."""
    for robot in robots or []:
        if carried_by(robot) is obj:
            return robot
    return None


def clear_cargo(scene=None) -> None:
    """Forget every load. For a reset, which removes the joints with the prims."""
    _CARGO.clear()


def _cargo_pose(carrier, obj):
    """Where on @carrier's back @obj should sit."""
    carrier_pos = carrier.get_position_orientation()[0]
    try:
        carrier_top = float(carrier.aabb[1][2])
    except Exception:  # noqa: BLE001 - a robot without an aabb
        carrier_top = float(carrier_pos[2])
    try:
        obj_half_height = float(obj.aabb_extent[2]) / 2.0
    except Exception:  # noqa: BLE001
        obj_half_height = 0.05
    return th.tensor(
        [float(carrier_pos[0]), float(carrier_pos[1]), carrier_top + obj_half_height],
        dtype=th.float32,
    )


def _base_link(robot):
    """The link cargo is welded to: the carrier's own base."""
    name = getattr(robot, "base_footprint_link_name", None)
    links = getattr(robot, "links", {}) or {}
    if name and name in links:
        return links[name]
    # Fall back to the root link, which is what a robot without a declared
    # footprint link has.
    root = getattr(robot, "root_link", None)
    if root is not None:
        return root
    return next(iter(links.values())) if links else None


def load_onto(carrier, obj) -> str:
    """Weld @obj to @carrier's back. Returns the joint's prim path.

    The caller has already released it; this only fixes it in place.
    """
    import omnigibson as og  # noqa: PLC0415
    from omnigibson.utils.usd_utils import create_joint  # noqa: PLC0415

    link = _base_link(carrier)
    if link is None:
        raise ValueError(f"{getattr(carrier, 'name', carrier)} has no link to carry on")

    obj.set_position_orientation(position=_cargo_pose(carrier, obj))
    try:
        obj.keep_still()
    except Exception:  # noqa: BLE001 - not every object exposes it
        pass

    # Frames relative to each body, exactly as an assisted grasp computes them,
    # so the joint holds the object where it was put rather than snapping it.
    import omnigibson.utils.transform_utils as T  # noqa: PLC0415

    contact = obj.get_position_orientation()[0]
    identity = th.tensor([0.0, 0.0, 0.0, 1.0])
    link_pos, link_orn = link.get_position_orientation()
    parent_pos, parent_orn = T.relative_pose_transform(contact, identity, link_pos, link_orn)
    child_link = obj.root_link
    child_pos_world, child_orn_world = child_link.get_position_orientation()
    child_pos, child_orn = T.relative_pose_transform(
        contact, identity, child_pos_world, child_orn_world
    )

    joint_path = f"{link.prim_path}/{CARGO_JOINT_NAME}"
    create_joint(
        prim_path=joint_path,
        joint_type="FixedJoint",
        body0=link.prim_path,
        body1=child_link.prim_path,
        enabled=True,
        exclude_from_articulation=True,
        joint_frame_in_parent_frame_pos=parent_pos / carrier.scale,
        joint_frame_in_parent_frame_quat=parent_orn,
        joint_frame_in_child_frame_pos=child_pos / obj.scale,
        joint_frame_in_child_frame_quat=child_orn,
    )
    # Same reason as the removal below: creating a joint under a robot's link
    # changes the articulation, and the views have to be rebuilt before anyone
    # reads a joint position again.
    og.sim.update_handles()
    _CARGO[carrier.name] = (joint_path, obj)
    return joint_path


def unload_from(carrier) -> Optional[Any]:
    """Remove the cargo joint on @carrier. Returns what was riding on it.

    Both steps are what `Robot._release_grasp` does for its own joint, and both
    are required. `stage.RemovePrim` alone leaves every articulation view that
    contained the joint stale, and the next `apply_action` on that robot dies in
    `get_joint_positions` with `'NoneType' object has no attribute 'view'` --
    seen 950 steps into a run, on the tick after the unload.
    """
    import omnigibson as og  # noqa: PLC0415
    from omnigibson.utils.usd_utils import delete_or_deactivate_prim  # noqa: PLC0415

    entry = _CARGO.pop(getattr(carrier, "name", None), None)
    if entry is None:
        return None
    joint_path, obj = entry
    delete_or_deactivate_prim(joint_path)
    og.sim.update_handles()
    return obj
