"""Rebuild the traversability map from the scene that is actually loaded.

The dataset bakes two maps per scene: ``floor_trav_0.png`` with every object of
the shipped layout in it, and ``floor_trav_no_obj_0.png`` with the building
shell alone. Both are wrong for a task instance, and in opposite directions.

Measured on Merom_1_int, the ring a robot must stand in to reach
``armchair_qplklw_2`` -- node C4 of the S1 route, after COOHAVIOR's filter has
removed 84 objects and moved this one into a corner:

    map                      ridgeback     jackal
    with objects (shipped)        0.0 %      0.0 %     <- every candidate refused
    no objects                   26.4 %     25.7 %     <- but 52-55 % of those
                                                          spots are inside the
                                                          breakfast table, which
                                                          really is there

So the with-objects map blocks floor that was cleared, and the no-objects map
opens floor that is occupied. Neither describes the scene being simulated, and
the first one is what made every LH/HL/HH episode stop at C4: 200 of 200 base
poses rejected, 75 of them for traversability alone.

This takes the no-object map as the shell and paints back the footprint of
every object the scene really holds. Nothing here samples or plans; it replaces
``scene.trav_map.floor_map[0]``, which is the one tensor the navigation
sampler, ``place_robots`` and ``scene.get_random_point`` all read.

What is painted, and what is not:

* objects whose top is above ``MIN_OBSTACLE_HEIGHT`` -- a die or a notebook
  lying on the floor is 2-3 cm tall and is not an obstacle, and painting the
  cargo would block the very pose a robot needs to pick it up;
* not floors, carpets or rugs (they are what you walk on);
* not walls, ceilings, windows or doors -- the shell is already in the base
  map, and COOHAVIOR deletes the doors anyway;
* not robots: they move, and ``_clear_of_other_robots`` already keeps poses off
  them and off each other's reservations.

Footprints come from each object's live world AABB, so an object that has been
moved -- which is most of a V4 port -- is painted where it now stands.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

#: An object shorter than this is driven past, not around.
MIN_OBSTACLE_HEIGHT = 0.10

#: Already in the no-object base map, or deliberately absent from it.
STRUCTURE_CATEGORIES = frozenset({"walls", "ceilings", "window", "openable_window", "door", "doors"})

#: What a robot stands on.
WALKABLE_CATEGORIES = frozenset({"floors", "carpet", "rug"})


def paint_footprints(base, boxes, to_pixel, size):
    """Return @base with each footprint in @boxes set to 0 (blocked).

    Pure and sim-free so a CPU test can exercise it. @boxes is a sequence of
    ``(name, x_lo, x_hi, y_lo, y_hi)`` in world metres; @to_pixel maps one
    ``(x, y)`` to ``(row, col)``.
    """
    painted = []
    for name, x_lo, x_hi, y_lo, y_hi in boxes:
        r0, c0 = to_pixel((x_lo, y_lo))
        r1, c1 = to_pixel((x_hi, y_hi))
        r0, r1 = sorted((int(r0), int(r1)))
        c0, c1 = sorted((int(c0), int(c1)))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, size - 1), min(c1, size - 1)
        if r0 > r1 or c0 > c1:
            continue
        base[r0:r1 + 1, c0:c1 + 1] = 0
        painted.append(name)
    return base, painted


def obstacle_boxes(scene, min_height: float = MIN_OBSTACLE_HEIGHT) -> Sequence:
    """``(name, x_lo, x_hi, y_lo, y_hi)`` for every object that blocks a robot."""
    robots = {id(r) for r in (getattr(scene, "robots", None) or [])}
    boxes = []
    for obj in (getattr(scene, "objects", None) or []):
        if id(obj) in robots or hasattr(obj, "action_dim"):
            continue
        category = str(getattr(obj, "category", "") or "").lower()
        if category in WALKABLE_CATEGORIES or category in STRUCTURE_CATEGORIES or category == "agent":
            continue
        try:
            lo, hi = obj.aabb
            lo = [float(v) for v in lo]
            hi = [float(v) for v in hi]
        except Exception:  # noqa: BLE001 - an object without a live AABB blocks nothing
            continue
        if hi[2] <= min_height:
            continue
        boxes.append((getattr(obj, "name", "?"), lo[0], hi[0], lo[1], hi[1]))
    return boxes


def rebuild_trav_map(scene, min_height: float = MIN_OBSTACLE_HEIGHT, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Replace the scene's floor map with shell + what is actually in the scene.

    The shell comes from OmniGibson's own loader rather than a second reader
    here: flip ``trav_map_with_objects`` and call ``load_map`` again, so the
    resize, the 255-threshold and the ``world_to_map`` convention are upstream's
    and cannot drift from them. Only the painting is ours -- there is no
    upstream API that derives a floor map from a loaded scene (``TraversableMap``
    reads baked PNGs; ``ScanSensor.occupancy_grid`` is a robot-local 5 m lidar
    grid, not a scene map).

    Returns a summary, or None when the scene has no traversability map (a
    plain ``Scene``, or a dataset without the baked PNGs).
    """
    import torch as th  # noqa: PLC0415

    trav_map = getattr(scene, "trav_map", None)
    floor_map = getattr(trav_map, "floor_map", None) if trav_map is not None else None
    if not floor_map:
        return None
    scene_dir = getattr(scene, "scene_dir", None)
    if not scene_dir:
        return None
    layout_dir = os.path.join(scene_dir, "layout")
    if not os.path.exists(os.path.join(layout_dir, "floor_trav_no_obj_0.png")):
        if verbose:
            print(f"[trav] no shell map in {layout_dir}; keeping the loaded one")
        return None

    with_objects = trav_map.trav_map_with_objects
    try:
        trav_map.trav_map_with_objects = False
        # `_load_map` computes map_size only when trav_map_original_size is
        # None, and leaves it None otherwise -- reloading without clearing it
        # hands cv2.resize a (None, None) and raises.
        trav_map.trav_map_original_size = None
        trav_map.load_map(layout_dir, floor_heights=trav_map.floor_heights)
    finally:
        trav_map.trav_map_with_objects = with_objects

    # Re-read the list: `_load_map` assigns `self.floor_map = []` and appends,
    # so the list captured before the reload is a discarded object. Painting
    # into that one leaves the scene on the bare shell -- every pose inside the
    # furniture that is really there would be accepted.
    floor_map = trav_map.floor_map
    shell = floor_map[0].cpu().numpy().copy()
    size = int(shell.shape[0])
    shell_free = int((shell > 0).sum())
    resolution = float(trav_map.map_resolution)

    def to_pixel(xy):
        # Upstream's world_to_map: x indexes the column, y the row.
        return (xy[1] / resolution + size / 2.0, xy[0] / resolution + size / 2.0)

    boxes = obstacle_boxes(scene, min_height=min_height)
    shell, painted = paint_footprints(shell, boxes, to_pixel, size)
    floor_map[0] = th.tensor(shell)
    assert floor_map is trav_map.floor_map and int((trav_map.floor_map[0] > 0).sum()) == int((shell > 0).sum()), (
        "the painted map is not the one the scene reads")

    summary = {
        "objects_painted": len(painted),
        "shell_free_m2": round(shell_free * resolution * resolution, 1),
        "free_m2": round(int((shell > 0).sum()) * resolution * resolution, 1),
    }
    if verbose:
        print(f"[trav] floor map rebuilt: shell {summary['shell_free_m2']} m2 minus "
              f"{summary['objects_painted']} objects actually in the scene -> "
              f"{summary['free_m2']} m2 standable")
    return summary
