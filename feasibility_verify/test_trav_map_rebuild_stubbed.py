"""CPU-only: the floor map is rebuilt from the scene that is actually loaded.

The dataset ships two baked maps and a task instance matches neither. Measured
on Merom_1_int, the ring for node C4's armchair: 0.0% standable on the shipped
with-objects map (200 of 200 base poses refused, which stopped every LH/HL/HH
episode there), 26% on the no-object map -- of which half is inside the
breakfast table, which really is in the scene.

This checks the painting rule on a synthetic 20x20 m room: the shell is open,
and only what the scene holds blocks.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/zixuanwe/Desktop/BEHAVIOR-1K")
import numpy as np

from coop2.behavior_env.trav_map_rebuild import MIN_OBSTACLE_HEIGHT, obstacle_boxes, paint_footprints


class FakeObject:
    def __init__(self, name, category, lo, hi):
        self.name, self.category, self.aabb = name, category, (lo, hi)


class FakeRobot(FakeObject):
    action_dim = 3


class FakeScene:
    def __init__(self, objects, robots):
        self.objects, self.robots = objects, robots


def main() -> int:
    RES, SIZE = 0.1, 200                       # 20 x 20 m at 0.1 m
    to_pixel = lambda xy: (xy[1] / RES + SIZE / 2.0, xy[0] / RES + SIZE / 2.0)
    free = lambda grid, x, y: grid[int(y / RES + SIZE / 2), int(x / RES + SIZE / 2)] > 0

    table = FakeObject("breakfast_table_0", "breakfast_table", (1.0, 1.0, 0.0), (2.0, 2.0, 0.8))
    die = FakeObject("dice_0", "dice", (-1.0, -1.0, 0.0), (-0.96, -0.96, 0.036))
    rug = FakeObject("rug_0", "carpet", (3.0, 3.0, 0.0), (5.0, 5.0, 0.02))
    wall = FakeObject("walls_0", "walls", (-5.0, -5.0, 0.0), (-4.0, 5.0, 2.4))
    window = FakeObject("openable_window_0", "openable_window", (4.0, -4.0, 0.5), (4.2, -2.0, 1.8))
    robot = FakeRobot("jackal_1", "agent", (0.0, 3.0, 0.0), (0.7, 3.7, 0.7))
    scene = FakeScene([table, die, rug, wall, window, robot], [robot])

    print("1. what counts as an obstacle")
    names = {b[0] for b in obstacle_boxes(scene)}
    assert names == {"breakfast_table_0"}, names
    print(f"   painted {names}; skipped the die (2 cm), the rug, the wall, the window and the robot")

    print("\n2. painting blocks the footprint and nothing else")
    shell = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    grid, painted = paint_footprints(shell.copy(), obstacle_boxes(scene), to_pixel, SIZE)
    assert painted == ["breakfast_table_0"]
    assert not free(grid, 1.5, 1.5), "inside the table must be blocked"
    assert free(grid, 2.5, 1.5) and free(grid, 1.5, 2.5), "beside the table stays open"
    assert free(grid, -1.0, -1.0), "a die on the floor is not an obstacle"
    assert free(grid, 4.0, 4.0), "a rug is walked on"
    blocked = int((grid == 0).sum()) * RES * RES
    assert abs(blocked - 1.0) < 0.25, f"one 1x1 m table, got {blocked:.2f} m2"
    print(f"   {blocked:.2f} m2 blocked, the table's own footprint")

    print("\n3. a taller object is painted, a shorter one is not")
    low = FakeObject("threshold_0", "misc", (6.0, 6.0, 0.0), (6.5, 6.5, MIN_OBSTACLE_HEIGHT - 0.01))
    high = FakeObject("cabinet_0", "bottom_cabinet", (7.0, 7.0, 0.0), (7.5, 7.5, MIN_OBSTACLE_HEIGHT + 0.01))
    scene.objects += [low, high]
    names = {b[0] for b in obstacle_boxes(scene)}
    assert "cabinet_0" in names and "threshold_0" not in names, names
    print(f"   > {MIN_OBSTACLE_HEIGHT} m blocks, below it does not")

    print("\n4. out-of-bounds footprints are clipped, not crashed")
    scene.objects.append(FakeObject("far_0", "misc", (-999.0, -999.0, 0.0), (-990.0, -990.0, 1.0)))
    grid, painted = paint_footprints(np.full((SIZE, SIZE), 255, dtype=np.uint8), obstacle_boxes(scene), to_pixel, SIZE)
    assert "far_0" not in painted and free(grid, 0.0, 0.0)
    print("   an object outside the map paints nothing")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
