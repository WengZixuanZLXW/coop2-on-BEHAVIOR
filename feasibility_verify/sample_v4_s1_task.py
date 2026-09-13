"""Sample and cache the other COOHAVIOR S1 instances, with V4's own layout.

`sample_v4_s1_v4_ll.py` did S1-V4-LL and is left alone; this is the same
procedure for the rest of the S1 family, table-driven because they differ only
in which boxes exist, where they start and what they must end up on. The two
fiddly helpers -- the scene edits and finding the written template -- are
imported from that script rather than copied, so there is one implementation of
each.

    python feasibility_verify/sample_v4_s1_task.py --task lh
    python feasibility_verify/sample_v4_s1_task.py --task hh
    python feasibility_verify/sample_v4_s1_task.py --task hl

S1-V4-HL's goal holds at t=0 by construction; see its TASKS entry for why that is accepted.
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
from feasibility_verify.sample_v4_s1_v4_ll import (
    _apply_scene_edits,
    _written_template_path,
)

SCENE_MODEL = "Merom_1_int"

#: V4 makes the same furniture edits for all four S1 difficulties -- its filter
#: is one USD layer the four staging scenes all reference -- so they share one
#: file. Only the box mass differs, and that is per task below.
SCENE_EDITS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "coop2", "team_layouts", "v4_s1_v4_ll_scene.json",
)

#: Each task: the activity, and for every box its V4 staging coordinate, the
#: floor instance it starts on and the room that floor is in. Coordinates are
#: `V4_Task_Staging/task_boxes/box_s1_m<N>` read off the staging USD; the room
#: is asserted at runtime rather than trusted, because a coordinate in the wrong
#: room is exactly the mistake this table can make.
TASKS = {
    "lh": {
        "activity": "v4_s1_v4_lh",
        # One box, same spawn as LL -- the difference is the destination, which
        # is a bedroom cabinet rather than the bedroom floor.
        "boxes": [("notebook.n.01_1", (-0.0672, 0.415), "floor.n.01_1", "childs_room_0")],
        # LH is one of V4's two *heavy* tasks. The authority on box mass is
        # COOHAVIOR/behavior_style_task/tasks.modified.json: `tasks[].box_mass_kg` is 0.02 for LH and HH,
        # 0.008 for LL and HL (the staging USDs agree where they say anything;
        # S2/S3's say nothing). V4 calls both weights `packing_box.n.02_N`, so
        # nothing symbolic could tell them apart; here the heavy cargo is a
        # notebook and the light one a die.
        #
        # The weight is a *capability* constraint, not decoration -- that file's
        # `condition_physical_execution_contract.LH_HH` says the drone "is not a
        # load-bearing role for the 20 g box" and heavy packages require
        # pull_car + push_car + robot_arm together, where the 8 g box needs only
        # a push_car and the drone "may suction-lift/transport" it. Making the
        # weights two objects is what lets that be stated symbolically. It is
        # NOT enforced yet: the symbolic grasp is mass-blind and the Crazyflie
        # will still be offered grasp(notebook).
        "cargo": "notebook.n.01",
        "box_mass_kg": 0.02,
        "whitelist": {
            "notebook.n.01": {"notebook": {"aanuhi": None}},
            # bedroom_0 holds four cabinets and `inroom` binds by room *type*,
            # so without this the goal can land on any of them. `jrhgeu` is the
            # model V4 lists as an S1 checkpoint support.
            "cabinet.n.01": {"bottom_cabinet": {"jrhgeu": None, "slgzfc": None, "dajebq": None}},
        },
        "expect_rooms": {"cabinet.n.01_1": "bedroom_0"},
    },
    "hh": {
        "activity": "v4_s1_v4_hh",
        # Five boxes, one per station, each in its own room. The station order
        # is V4's own route (audit_index: M5, M3, M2, M1, M4) and the BDDL
        # numbers its floors to match, so floor.n.01_N is station M<N>.
        "boxes": [
            ("notebook.n.01_1", (-0.0672, 0.4150), "floor.n.01_5", "childs_room_0"),
            ("notebook.n.01_2", (-1.7051, 3.7108), "floor.n.01_3", "kitchen_0"),
            ("notebook.n.01_3", (-2.1632, 6.8168), "floor.n.01_2", "dining_room_0"),
            ("notebook.n.01_4", (3.3584, 7.8162), "floor.n.01_1", "living_room_0"),
            ("notebook.n.01_5", (4.2805, 0.5946), "floor.n.01_4", "bedroom_0"),
        ],
        # Heavy, like LH: `box_mass_kg` 0.02 in COOHAVIOR/behavior_style_task/tasks.modified.json.
        "cargo": "notebook.n.01",
        "box_mass_kg": 0.02,
        # Five rooms have to sample, not one, so this is above upstream's 10.
        # It is NOT what fixed dining_room_0 -- that was the traversability map,
        # see the scene config below. Raising attempts to 30 there bought four
        # and a half minutes of failure and nothing else, because the room had
        # no reachable floor at all on the map being read.
        "attempts": 30,
        "whitelist": {
            # 35 notebook models exist, so pinning is what keeps a re-sample
            # from swapping the object. aanuhi is 0.151 x 0.120 x 0.028 --
            # far smaller than the packing_box this used to be (0.380 x 0.468),
            # which is incidental here but not nothing: the box's 0.30 m
            # half-diagonal was what made dining_room_0 hard to place in.
            "notebook.n.01": {"notebook": {"aanuhi": None}},
            # One bookcase in the scene, so this only guards a future re-draw.
            # The armchairs and beds are deliberately unpinned: dining_room_0's
            # three armchairs are all model `qplklw` and both beds are `zrumze`,
            # so a whitelist cannot separate them. Whichever binds is reported
            # below and protected from the furniture filter.
            "bookcase.n.01": {"bookcase": {"owvfik": None}},
        },
        "expect_rooms": {
            "bookcase.n.01_1": "kitchen_0",
            "armchair.n.01_1": "dining_room_0",
            "coffee_table.n.01_1": "living_room_0",
            "bed.n.01_1": "childs_room_0",
            "bed.n.01_2": "bedroom_0",
        },
    },
    "hl": {
        "activity": "v4_s1_v4_hl",
        # Five boxes, the same five stations as HH; only m3 differs
        # ((-0.6903, 3.9197) here against HH's (-1.7051, 3.7108)), read off
        # S1-V4-HL.usda's own box prims rather than copied from HH.
        "boxes": [
            ("die.n.01_1", (-0.0672, 0.4150), "floor.n.01_5", "childs_room_0"),
            ("die.n.01_2", (-0.6903, 3.9197), "floor.n.01_3", "kitchen_0"),
            ("die.n.01_3", (-2.1632, 6.8168), "floor.n.01_2", "dining_room_0"),
            ("die.n.01_4", (3.3584, 7.8162), "floor.n.01_1", "living_room_0"),
            ("die.n.01_5", (4.2805, 0.5946), "floor.n.01_4", "bedroom_0"),
        ],
        # Light, like LL: box_mass_kg 0.008 in tasks.modified.json, so dice.
        "cargo": "die.n.01",
        "box_mass_kg": 0.008,
        "attempts": 30,
        "whitelist": {"die.n.01": {"dice": {"iswudu": 2.0}}},
        "expect_rooms": {},
        # The goal holds at t=0, and that is the task as written. V4's five HL
        # goals are `ontop(box_i, floor.n.01_target_i)` with the target floor
        # `inroom` the same room the box starts in: its targets are *positions*
        # on a floor. Ours is a room-level predicate and Merom_1_int has one
        # floor object per room, so start and target are the same object and
        # every goal clause is literally an initial condition. COOHAVIOR's real
        # task structure is the ordered package sequence in tasks.modified.json
        # (`packages[].depends_on`, `routes`), which a supervision layer will
        # enforce later; the BDDL goal is only the final state (user,
        # 2026-09-12). So the "goal already met" refusal is switched off for
        # this one task and no other. Until that layer exists, `check_goal`
        # is the only authority over `terminated`, and a run of this activity
        # ends at env_step 0.
        "allow_trivial_goal": True,
    },
}

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


def build_config(spec, instance_id):
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": None},
        "scene": {
            "type": "InteractiveTraversableScene",
            # No load_room_types filter: it bakes into the template, which *is*
            # the scene file every later run loads, and floors are not exempt
            # from it -- the rest of the building would load with no ground.
            "scene_model": SCENE_MODEL,
            "seg_map_resolution": 0.1,
            # The map that matches the scene we actually run. COOHAVIOR
            # deactivates 84 of Merom_1_int's furniture objects, so the baked
            # `floor_trav_0.png` -- which has all of it -- describes a scene
            # this task is not in.
            #
            # It is not a preference, it is the difference between sampling and
            # not. `sample_kinematics(use_trav_map=True)` is a *reachability*
            # test: it erodes the floor map by the robot's radius and dilates by
            # arm reach, asking whether a robot could stand somewhere and reach
            # the pose. Measured per room, free area after eroding by R1's
            # 0.62 m (m2, with objects -> without):
            #
            #     dining_room_0   0.00 -> 1.23        living_room_0  3.65 -> 11.33
            #     childs_room_0   0.36 -> 5.35        bedroom_0      0.67 ->  7.06
            #     kitchen_0       0.41 -> 3.98
            #
            # dining_room_0 is *zero* with objects, so `ontop(box, that floor)`
            # could never sample at any attempt count -- it dropped the dining
            # room out of the room intersection and failed the whole activity.
            # The other four scraped through on tenths of a square metre, which
            # is why only one room appeared to be the problem.
            #
            # Removing the furniture before sampling does not help: the map is
            # a baked PNG loaded from the dataset, and deleting objects at
            # runtime does not redraw it.
            "trav_map_with_objects": False,
        },
        "robots": [{
            "model": "r1",
            "name": "agent_0",
            "obs_modalities": [],
            "default_reset_mode": "tuck",
            # Parked off the floor plan; stripped from the template afterwards.
            "position": [-80.0, -80.0, 0.05],
        }],
        "task": {
            "type": "BehaviorTask",
            "activity_name": spec["activity"],
            "activity_definition_id": 0,
            "activity_instance_id": instance_id,
            "online_object_sampling": True,
            "sampling_whitelist": spec["whitelist"],
            "use_presampled_robot_pose": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--settle", type=int, default=200)
    parser.add_argument("--attempts", type=int, default=None,
                        help="sampling attempts per condition; overrides the task's own")
    args = parser.parse_args()

    spec = TASKS[args.task]
    attempts = args.attempts or spec.get("attempts", 5)
    macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = attempts
    macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = attempts
    print(f"sampling attempts: {attempts}")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    activity = spec["activity"]

    env = og.Environment(configs=build_config(spec, args.instance_id))
    assert env.task.feedback is None, f"Sampling failed: {env.task.feedback}"
    print(f"\nSampling succeeded for {activity} on {SCENE_MODEL}")

    scope = env.task.object_scope
    assert scope["agent.n.01_1"] is env.robots[0], "agent.n.01_1 is not env.robots[0]"

    def room_of(xy):
        try:
            return env.scene.seg_map.get_room_instance_by_point(th.tensor(xy, dtype=th.float32))
        except Exception as exc:  # noqa: BLE001
            return f"(unknown: {type(exc).__name__})"

    def seg_room_of_object(obj):
        """The room the object is actually standing in.

        Not `obj.in_rooms`, which is a scene annotation and can disagree with
        the geometry: `bottom_cabinet_jrhgeu_1` is annotated `bedroom_0` and
        stands in `childs_room_0`. The seg map is what the world model, the
        room listing and `inspect_scene` all read, so it is what the agent will
        be shown -- and checking the annotation instead let a bad binding pass.
        """
        try:
            return room_of([float(c) for c in obj.get_position_orientation()[0][:2]])
        except Exception as exc:  # noqa: BLE001
            return f"(unknown: {type(exc).__name__})"

    print("\nbound scope:  (annotated in_rooms | where it actually stands)")
    robot_names = {r.name for r in env.robots}
    for entity_id, entity in sorted(scope.items()):
        obj = getattr(entity, "wrapped_obj", entity)
        if obj is None or getattr(obj, "name", None) in robot_names:
            continue
        annotated = list(getattr(obj, "in_rooms", None) or [])
        actual = seg_room_of_object(obj)
        flag = "" if actual in annotated else "   <-- DISAGREE"
        print(f"  {entity_id:22} -> {obj.name:28} {str(annotated):20} {actual}{flag}")

    # The binding this port depends on, checked rather than assumed. A goal
    # bound to the wrong room is not visible in any later run -- the task
    # simply never completes, or completes without anyone leaving the room.
    for entity_id, want in (spec.get("expect_rooms") or {}).items():
        obj = getattr(scope[entity_id], "wrapped_obj", scope[entity_id])
        actual = seg_room_of_object(obj)
        assert actual == want, (
            f"{entity_id} bound to {obj.name}, which stands in {actual}, not {want}"
            f" (its annotation says {list(getattr(obj, 'in_rooms', None) or [])})")
    print(f"  all {len(spec.get('expect_rooms') or {})} expected room bindings hold")

    og.sim.play()
    env.task.reset(env)

    # After the task reset, not before: reset restores the scene from its own
    # initial file, so anything removed beforehand comes straight back.
    edits = {}
    if os.path.exists(SCENE_EDITS):
        with open(SCENE_EDITS) as handle:
            edits = json.load(handle)
        edits = dict(edits, cargo_prefix=spec["cargo"])
        if spec.get("box_mass_kg") is not None:
            edits["box_mass_kg"] = spec["box_mass_kg"]
        _apply_scene_edits(env, edits)

    # Wake the task objects, or every check below reads a lie: a sleeping PhysX
    # actor reports no contacts, so OnTop comes back False for a box sitting
    # still exactly where it should be.
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

    # Pin every box to its V4 coordinate, dropped from a little above the
    # staged height so it settles onto whatever is actually there rather than
    # being forced to a z measured against a different floor mesh.
    # Remember what the sampler chose before overwriting it. Its poses are
    # already validated against the initial conditions, so they are the
    # fallback when a V4 coordinate turns out not to be a place the box can
    # actually sit -- see the retreat below.
    sampled_pose = {
        box_id: tuple(x.clone() for x in
                      getattr(scope[box_id], "wrapped_obj", scope[box_id]).get_position_orientation())
        for box_id, *_ in spec["boxes"]
    }

    print("\npinning boxes to their V4 staging coordinates:")
    for box_id, xy, floor_id, want_room in spec["boxes"]:
        box = getattr(scope[box_id], "wrapped_obj", scope[box_id])
        got = room_of(list(xy))
        assert got == want_room, f"{box_id}'s V4 coordinate {xy} is in {got}, not {want_room}"
        z = float(box.get_position_orientation()[0][2])
        box.set_position_orientation(position=th.tensor([xy[0], xy[1], z + 0.05]))
        box.keep_still()
        print(f"  {box_id:22} -> ({xy[0]:+.4f}, {xy[1]:+.4f}) in {got}")

    def report(label):
        """Where each box ended up, and whether its initial condition holds."""
        print(f"\n{label}")
        bad = []
        for box_id, xy, floor_id, want_room in spec["boxes"]:
            box = getattr(scope[box_id], "wrapped_obj", scope[box_id])
            final = box.get_position_orientation()[0]
            on_floor = bool(box.states[object_states.OnTop].get_value(scope[floor_id]))
            drift = float(th.norm(final[:2] - th.tensor(xy, dtype=final.dtype)))
            here = room_of([float(final[0]), float(final[1])])
            print(f"  {box_id:22} ({float(final[0]):+.3f}, {float(final[1]):+.3f}, "
                  f"{float(final[2]):+.3f}) in {here:16} ontop({floor_id}): {on_floor}  "
                  f"drift {drift:.3f} m")
            if not (on_floor and here == want_room):
                bad.append(box_id)
        return bad

    for _ in range(args.settle):
        og.sim.step()
    bad = report(f"after a {args.settle}-step settle:")

    # Retreat, per box, to the pose the sampler validated.
    #
    # V4's staging is explicitly a draft -- its own box asset is documented
    # "visual-only until collision is deliberately authored" -- and its boxes
    # are staged where they overlap furniture. S1-V4-HH's `box_s1_m2` sits at
    # exactly the coordinate its own BehaviorScene block moves
    # `armchair_qplklw_2` to, so a box with collision lands on the armchair,
    # and `ontop(notebook.n.01_3, floor.n.01_2)` -- which is what its BDDL
    # says -- is false.
    #
    # The BDDL is the task; the staging coordinate is only how the task is
    # reproduced. Where the two disagree the BDDL wins, and the deviation is
    # printed rather than hidden.
    if bad:
        print(f"\n{len(bad)} box(es) cannot sit at the V4 coordinate; "
              f"returning them to the sampler's own validated pose: {bad}")
        for box_id in bad:
            box = getattr(scope[box_id], "wrapped_obj", scope[box_id])
            position, orientation = sampled_pose[box_id]
            box.set_position_orientation(position=position, orientation=orientation)
            box.keep_still()
        for _ in range(args.settle):
            og.sim.step()
        bad = report("after returning them and settling again:")

    if bad:
        print(f"\nSTILL NOT WHERE THE INITIAL CONDITIONS SAY: {bad} -- not saving")
        og.shutdown()
        sys.exit(1)

    goal_met, breakdown = env.task.compiled_task.check_goal(env.task._evaluate_predicate)
    if spec.get("allow_trivial_goal"):
        print(f"\ngoal already met: {goal_met}  {breakdown}   (EXPECTED for this task: the BDDL "
              f"goal is only the final state; the ordered sub-tasks are enforced elsewhere)")
    else:
        print(f"\ngoal already met: {goal_met}  {breakdown}   (must be False)")
        assert not goal_met, "the goal holds at t=0 -- the task would be trivial"

    if args.no_save:
        print("\n--no_save given; nothing written")
    else:
        env.scene.update_initial_file()
        env.task.save_task(env=env, save_dir=args.save_dir, override=True,
                           task_relevant_only=False)
        fname = env.task.get_cached_activity_scene_filename(
            scene_model=SCENE_MODEL,
            activity_name=activity,
            activity_definition_id=0,
            activity_instance_id=args.instance_id,
        )
        print(f"\nwrote instance {fname}")

        # Strip the sampler's robot: the template *is* the scene file every
        # episode loads, and `Environment._load_robots` is guarded by
        # `if len(self.scene.robots) == 0`, so a robot left here silently
        # overrides the whole team layout.
        written = _written_template_path(args.save_dir, fname, scene_model=SCENE_MODEL)
        if written is not None:
            removed = strip_robots_from_template(written)
            print(f"stripped the sampler's robots from the template: {removed}")
        else:
            print("WARNING: could not locate the written template to strip its robots")

    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
