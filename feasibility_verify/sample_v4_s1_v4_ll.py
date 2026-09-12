"""Sample and cache the ``v4_s1_v4_ll`` instance, with COOHAVIOR's own layout.

Ported from COOHAVIOR's S1-V4-LL: move a packing box from the child's room
floor to the bedroom floor, in Merom_1_int.

Unlike ``sample_nine_apples_hall.py`` this does **not** accept the sampler's
layout. The point of porting the task is to reproduce it, so the box is pinned
to the coordinate its V4 staging USD places it at, and the sampler is used only
to bind the BDDL scope (which floor is ``floor.n.01_1``, which is
``floor.n.01_2``) and to build a loadable instance around it.

That is sound because the two coordinate frames are the same one. COOHAVIOR's
staging USD edits BEHAVIOR's objects by their own names, and of the eight
objects checked, five sit at *identical* coordinates in
``Merom_1_int_best.json``; the three that differ are the moves the V4 layout
makes deliberately (armchair 1.14 m, fridge 2.11 m).

Run:
    python feasibility_verify/sample_v4_s1_v4_ll.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm, macros

from feasibility_verify.sample_coop_task_instance import strip_robots_from_template

ACTIVITY = "v4_s1_v4_ll"
SCENE_MODEL = "Merom_1_int"
N_ROBOTS = 1

#: Straight out of S1-V4-LL.usda, `V4_Task_Staging/task_boxes/box_s1_m5`. Its
#: spawn marker carries the BDDL relation this reproduces:
#:     ONTOP(packing_box.n.02_1, floor.n.01_5)
#: (`floor.n.01_5` there; `floor.n.01_1` here -- the V4 file numbered floors by
#: station and BDDL's own parser cannot carry two `- floor.n.01` lines, see the
#: activity definition.)
BOX_XY = (-0.0672, 0.415)

#: `V4_Task_Staging/task_markers/destination_s1_m5`, for reporting only: the
#: goal is a room-level predicate (`ontop` the bedroom floor), not this point.
DESTINATION_XY = (-1.5553878130256362, -1.0229410990222918)

#: One model exists for this synset, so pinning it changes nothing today and
#: keeps a re-sample from silently swapping the object if more are added.
SAMPLING_WHITELIST = {
    "packing_box.n.02": {"packing_box": {"cjhskr": None}},
}

#: The scene edits V4 makes, as a file rather than a literal: 84 objects to
#: remove, two to move, and the box's mass. See its own _comment.
SCENE_EDITS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "coop2", "team_layouts", "v4_s1_v4_ll_scene.json",
)

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5


def build_config(instance_id):
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": None},
        "scene": {
            "type": "InteractiveTraversableScene",
            # No load_room_types filter: it bakes into the template, which *is*
            # the scene file every later run loads, and floors are not exempt
            # from it -- the rest of the building would load with no ground.
            "scene_model": SCENE_MODEL,
            "seg_map_resolution": 0.1,
        },
        "robots": [
            {
                "model": "r1",
                "name": f"agent_{i}",
                "obs_modalities": [],
                "default_reset_mode": "tuck",
                # Parked off the floor plan so it is never in the sampler's way;
                # stripped from the template afterwards regardless.
                "position": [-80.0 - 2.0 * i, -80.0, 0.05],
            }
            for i in range(N_ROBOTS)
        ],
        "task": {
            "type": "BehaviorTask",
            "activity_name": ACTIVITY,
            "activity_definition_id": 0,
            "activity_instance_id": instance_id,
            "online_object_sampling": True,
            "sampling_whitelist": SAMPLING_WHITELIST,
            "use_presampled_robot_pose": False,
        },
    }


def _apply_scene_edits(env, edits):
    """Reproduce V4's own edits to the scene: remove, move, re-weigh.

    Removal rather than deactivation. V4 does this with a USD layer that sets
    `active = false`, which has no equivalent here -- and it does not need one,
    because `save_task` dumps whatever the scene holds, so an object removed
    before the dump is simply absent from the cached instance every later run
    loads.
    """
    removed, missing = [], []
    for name in edits.get("deactivate", []):
        obj = env.scene.object_registry("name", name)
        if obj is None:
            missing.append(name)
            continue
        env.scene.remove_object(obj)
        removed.append(name)
    print(f"\nremoved {len(removed)} objects V4 filters out"
          + (f"; {len(missing)} were not in this scene: {missing[:4]}" if missing else ""))

    import torch as th  # noqa: PLC0415

    for name, position in (edits.get("move") or {}).items():
        obj = env.scene.object_registry("name", name)
        if obj is None:
            print(f"  cannot move {name}: not in this scene")
            continue
        before = obj.get_position_orientation()[0]
        obj.set_position_orientation(position=th.tensor(position, dtype=th.float32))
        print(f"  moved {name}: ({float(before[0]):+.3f}, {float(before[1]):+.3f}) -> "
              f"({position[0]:+.3f}, {position[1]:+.3f})")

    mass = edits.get("box_mass_kg")
    if mass:
        box = (env.task.object_scope or {}).get("packing_box.n.02_1")
        box = getattr(box, "wrapped_obj", box)
        if box is not None:
            try:
                # V4 stages the box at 8 g so a 7 cm drone's suction cup can
                # lift it. The symbolic grasp does not read mass -- it teleports
                # and welds -- but the physics after a release does.
                box.root_link.mass = float(mass)
                print(f"  set {box.name} mass to {mass} kg")
            except Exception as error:  # noqa: BLE001
                print(f"  could not set box mass: {type(error).__name__}: {error}")


def _written_template_path(save_dir, fname):
    """Where save_task put @fname, or None."""
    import omnigibson as og  # noqa: PLC0415

    candidates = []
    if save_dir:
        candidates.append(os.path.join(save_dir, f"{fname}.json"))
    # Only real strings. `gm` is a MacroDict, and reading a key it does not have
    # returns an empty sub-dict rather than raising or yielding the getattr
    # default -- which handed os.path.join a MacroDict and took the process down
    # with a segfault, after the sampling had already succeeded.
    for root in (getattr(og.macros.gm, "DATASET_PATH", None),
                 getattr(og.macros.gm, "CUSTOM_DATASET_PATH", None)):
        if isinstance(root, str) and root:
            candidates.append(os.path.join(root, "scenes", SCENE_MODEL, "json", f"{fname}.json"))
    for base in ("datasets/2026-challenge-task-instances", "datasets/behavior-1k-assets"):
        candidates.append(os.path.join(base, "scenes", SCENE_MODEL, "json", f"{fname}.json"))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--settle", type=int, default=200)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    env = og.Environment(configs=build_config(args.instance_id))
    assert env.task.feedback is None, f"Sampling failed: {env.task.feedback}"
    print(f"\nSampling succeeded for {ACTIVITY} on {SCENE_MODEL}")

    scope = env.task.object_scope
    box = scope["packing_box.n.02_1"]
    floor_a = scope["floor.n.01_1"]
    floor_b = scope["floor.n.01_2"]
    assert scope["agent.n.01_1"] is env.robots[0], "agent.n.01_1 is not env.robots[0]"

    def room_of(xy):
        try:
            return env.scene.seg_map.get_room_instance_by_point(th.tensor(xy, dtype=th.float32))
        except Exception as exc:  # noqa: BLE001
            return f"(unknown: {type(exc).__name__})"

    sampled_xy = box.get_position_orientation()[0][:2]
    print(f"\nbound scope:")
    print(f"  packing_box.n.02_1 -> {box.name} ({box.category}-{box.model})")
    print(f"  floor.n.01_1       -> {floor_a.name}   (childs_room per the BDDL)")
    print(f"  floor.n.01_2       -> {floor_b.name}   (bedroom per the BDDL)")
    print(f"\nthe sampler put the box at ({float(sampled_xy[0]):+.3f}, {float(sampled_xy[1]):+.3f})"
          f" in {room_of([float(sampled_xy[0]), float(sampled_xy[1])])}")
    print(f"V4 staging puts it at      ({BOX_XY[0]:+.3f}, {BOX_XY[1]:+.3f})"
          f" in {room_of(list(BOX_XY))}")
    print(f"V4 destination marker      ({DESTINATION_XY[0]:+.3f}, {DESTINATION_XY[1]:+.3f})"
          f" in {room_of(list(DESTINATION_XY))}")

    og.sim.play()
    env.task.reset(env)

    # After the task reset, not before. `reset` restores the scene from its own
    # initial file, so anything removed beforehand comes straight back -- the
    # first attempt reported "removed 84 objects" and then saved a template with
    # all eleven doors, nineteen pictures and sixteen light switches still in it.
    edits = {}
    if os.path.exists(SCENE_EDITS):
        with open(SCENE_EDITS) as handle:
            edits = json.load(handle)
        _apply_scene_edits(env, edits)

    # Wake the task objects, or every check below reads a lie: a sleeping PhysX
    # actor reports no contacts, so OnTop comes back False for a box that is
    # sitting still exactly where it should be.
    robot_names = {r.name for r in env.robots}
    for entity in scope.values():
        obj = getattr(entity, "wrapped_obj", entity)
        if obj is None or getattr(obj, "name", None) in robot_names:
            continue
        if getattr(obj, "kinematic_only", True):
            continue
        try:
            obj.sleep_threshold = 0.0
            obj.wake()
        except Exception:  # noqa: BLE001 - never block sampling on this
            pass

    # Pin the box to the V4 coordinate. Dropped from a little above the staged
    # height so it settles onto whatever is actually there, rather than being
    # forced to a z that was measured against a different floor mesh.
    z = float(box.get_position_orientation()[0][2])
    box.set_position_orientation(position=th.tensor([BOX_XY[0], BOX_XY[1], z + 0.05]))
    box.keep_still()
    for _ in range(args.settle):
        og.sim.step()

    final = box.get_position_orientation()[0]
    on_a = bool(box.states[object_states.OnTop].get_value(floor_a))
    print(f"\nafter a {args.settle}-step settle:")
    print(f"  box at ({float(final[0]):+.3f}, {float(final[1]):+.3f}, {float(final[2]):+.3f})"
          f"  in {room_of([float(final[0]), float(final[1])])}")
    print(f"  ontop(packing_box.n.02_1, floor.n.01_1): {on_a}")
    drift = float(th.norm(final[:2] - th.tensor(BOX_XY, dtype=final.dtype)))
    print(f"  drift from the V4 coordinate: {drift:.3f} m")

    if not on_a:
        print("\nBOX IS NOT ON THE CHILD'S ROOM FLOOR -- not saving")
        og.shutdown()
        sys.exit(1)

    goal_met, breakdown = env.task.compiled_task.check_goal(env.task._evaluate_predicate)
    print(f"\ngoal already met: {goal_met}  {breakdown}   (must be False)")
    assert not goal_met, "the goal holds at t=0 -- the task would be trivial"

    if args.no_save:
        print("\n--no_save given; nothing written")
    else:
        env.scene.update_initial_file()
        env.task.save_task(env=env, save_dir=args.save_dir, override=True, task_relevant_only=False)
        fname = env.task.get_cached_activity_scene_filename(
            scene_model=SCENE_MODEL,
            activity_name=ACTIVITY,
            activity_definition_id=0,
            activity_instance_id=args.instance_id,
        )
        print(f"\nwrote instance {fname}")

        # Strip the sampler's robot out of the template. save_task dumps the
        # whole scene, robots included, and the template *is* the scene file
        # every episode loads -- so whatever robot the sampler happened to use
        # would come up in every later run, and `Environment._load_robots` is
        # guarded by `if len(self.scene.robots) == 0`, which means the team
        # layout's robots are then silently skipped.
        #
        # Leaving it in is not a cosmetic problem. The r1 baked in here was
        # later handed this task's controller config -- `base: JointController`,
        # which an r1's holonomic base does not accept -- and the run died with
        # KeyError('JointController') before the first step.
        written = _written_template_path(args.save_dir, fname)
        if written is not None:
            removed = strip_robots_from_template(written)
            print(f"stripped the sampler's robots from the template: {removed}")
        else:
            print("WARNING: could not locate the written template to strip its robots")

    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
