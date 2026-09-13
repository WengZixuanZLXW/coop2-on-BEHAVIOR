"""Sample and cache one S2/S3 activity instance on its Beechwood scene.

The S1 samplers pin every box to COOHAVIOR's staging coordinate. There is
nothing to pin here: COOHAVIOR's Beechwood frame is not our dataset's (its
stations fall outside or in the wrong rooms), so the cargo starts where the
sampler puts it on the station's floor, and every support is our scene's own
same-kind object in COOHAVIOR's room type -- see
feasibility_verify/make_v4_s2_s3_definitions.py for the mapping. What this
script checks is what matters for the route: every node bound, every support
standing in a room of the expected instance set (by the seg map, not the
annotation), the cargo `ontop` its floor after a settle, and the goal false.

    OMNIGIBSON_HEADLESS=1 python feasibility_verify/sample_v4_task.py --task s2_ll
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omnigibson as og  # noqa: E402
from omnigibson import object_states  # noqa: E402
from omnigibson.macros import gm, macros  # noqa: E402

from feasibility_verify.sample_coop_task_instance import strip_robots_from_template  # noqa: E402
from feasibility_verify.sample_v4_s1_v4_ll import _apply_scene_edits, _written_template_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARGO_WHITELIST = {"die.n.01": {"dice": {"iswudu": 2.0}}, "notebook.n.01": {"notebook": {"aanuhi": None}}}
LEVELS = {"ll": ("die.n.01", 0.008), "lh": ("notebook.n.01", 0.02), "hl": ("die.n.01", 0.008), "hh": ("notebook.n.01", 0.02)}

SCENES = {
    "s2": {
        "scene": "Beechwood_0_int",
        "edits": os.path.join(ROOT, "coop2", "team_layouts", "v4_s2_v4_ll_scene.json"),
        "whitelist": {
            "coffee_table.n.01": {"coffee_table": {"dnsjnv": None}},
            "footstool.n.01": {"ottoman": {"miftfy": None}},
            "breakfast_table.n.01": {"breakfast_table": {"skczfi": None}},
            "desk.n.01": {"desk": {"rzyfxk": None}},
            "bookcase.n.01": {"bookcase": {"owvfik": None}},
            # COOHAVIOR deactivates every tpuwys countertop we have; jveutp survive.
            "countertop.n.01": {"countertop": {"jveutp": None}},
            "sink.n.01": {"furniture_sink": {"zexzrc": None}},
            "straight_chair.n.01": {"straight_chair": {"dmcixv": None}},
        },
        # room instances each support may stand in (checked by the seg map)
        "expect": {
            "coffee_table.n.01_1": {"dining_room_0"}, "footstool.n.01_1": {"living_room_1"},
            "breakfast_table.n.01_1": {"living_room_1"},
            "desk.n.01_1": {"private_office_0"}, "desk.n.01_2": {"private_office_0"},
            "bookcase.n.01_1": {"kitchen_0"}, "countertop.n.01_1": {"kitchen_0"},
            "sink.n.01_1": {"bathroom_0"}, "straight_chair.n.01_1": {"dining_room_0"},
        },
    },
    "s3": {
        "scene": "Beechwood_1_int",
        "edits": os.path.join(ROOT, "coop2", "team_layouts", "v4_s3_v4_ll_scene.json"),
        "whitelist": {
            "cabinet.n.01": {"bottom_cabinet": {"jhymlr": None}, "bottom_cabinet_no_top": {"pluwfl": None}},
            "sofa.n.01": {"sofa": {"qnnwfx": None}},
            "bed.n.01": {"bed": {"zrumze": None}},
            "breakfast_table.n.01": {"breakfast_table": {"skczfi": None, "uhrsex": None}},
            "sink.n.01": {"multi_station_furniture_sink": {"yfaufu": None}},
        },
        "expect": {
            "cabinet.n.01_1": {"bedroom_0"}, "sofa.n.01_1": {"television_room_0"},
            "cabinet.n.01_2": {"television_room_0"}, "bed.n.01_1": {"childs_room_0"},
            "breakfast_table.n.01_1": {"childs_room_0"}, "cabinet.n.01_3": {"bathroom_1"},
            "sink.n.01_1": {"bathroom_1"}, "breakfast_table.n.01_2": {"playroom_0"},
            "breakfast_table.n.01_3": {"playroom_0"}, "bed.n.01_2": {"bedroom_0"},
        },
    },
}

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


def build_config(scene_model, activity, whitelist, instance_id):
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": None},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "seg_map_resolution": 0.1,
            # The no-object map: the edits remove most furniture, and the
            # sampler's reachability test must see the scene the task runs in.
            "trav_map_with_objects": False,
        },
        "robots": [{
            "model": "r1", "name": "agent_0", "obs_modalities": [], "default_reset_mode": "tuck",
            "position": [-80.0, -80.0, 0.05],
        }],
        "task": {
            "type": "BehaviorTask", "activity_name": activity, "activity_definition_id": 0,
            "activity_instance_id": instance_id, "online_object_sampling": True,
            "sampling_whitelist": whitelist, "use_presampled_robot_pose": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="e.g. s2_ll, s3_hh")
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--settle", type=int, default=200)
    parser.add_argument("--attempts", type=int, default=30)
    args = parser.parse_args()

    tag, level = args.task.split("_")
    spec = SCENES[tag]; cargo, mass = LEVELS[level]
    activity = f"v4_{tag}_v4_{level}"
    scene_model = spec["scene"]
    whitelist = dict(spec["whitelist"]); whitelist[cargo] = CARGO_WHITELIST[cargo]
    macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = args.attempts
    macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = args.attempts
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    import torch as th

    env = og.Environment(configs=build_config(scene_model, activity, whitelist, args.instance_id))
    assert env.task.feedback is None, f"Sampling failed: {env.task.feedback}"
    print(f"\nSampling succeeded for {activity} on {scene_model}")
    scope = env.task.object_scope
    robot_names = {r.name for r in env.robots}

    def room_of(xy):
        try:
            return env.scene.seg_map.get_room_instance_by_point(th.tensor(xy, dtype=th.float32))
        except Exception as exc:  # noqa: BLE001
            return f"(unknown: {type(exc).__name__})"

    def stands_in(obj):
        return room_of([float(c) for c in obj.get_position_orientation()[0][:2]])

    print("\nbound scope:  (annotated in_rooms | where it actually stands)")
    for entity_id, entity in sorted(scope.items()):
        obj = getattr(entity, "wrapped_obj", entity)
        if obj is None or getattr(obj, "name", None) in robot_names:
            continue
        annotated = list(getattr(obj, "in_rooms", None) or []); actual = stands_in(obj)
        print(f"  {entity_id:26} -> {obj.name:30} {str(annotated):22} {actual}{'' if actual in annotated else '   <-- DISAGREE'}")

    problems = []
    for entity_id, allowed in spec["expect"].items():
        if entity_id not in scope:
            continue
        obj = getattr(scope[entity_id], "wrapped_obj", scope[entity_id]); actual = stands_in(obj)
        if actual not in allowed:
            problems.append(f"{entity_id} bound to {obj.name}, which stands in {actual}, not in {sorted(allowed)}")
    assert not problems, "\n".join(problems)
    print(f"  all {len([e for e in spec['expect'] if e in scope])} expected room bindings hold")

    from coop2.behavior_env.route_spec import load_route_spec  # noqa: PLC0415
    route = load_route_spec(activity)
    print(f"\nroute bindings ({route.required_nodes} nodes):")
    starts = {}
    for r in route.routes:
        for node in r.nodes:
            entity = scope.get(node.support); obj = getattr(entity, "wrapped_obj", entity)
            assert obj is not None, f"{r.id} {node.id} names {node.support}, which did not bind"
            xy = obj.get_position_orientation()[0][:2]
            print(f"  {r.id} {node.id:<3} {node.support:<26} -> {obj.name:<28} ({float(xy[0]):+.2f}, {float(xy[1]):+.2f}) {stands_in(obj)}")
        starts[r.cargo] = r.start_support

    og.sim.play()
    env.task.reset(env)
    edits = json.load(open(spec["edits"]))
    edits = dict(edits, cargo_prefix=cargo, box_mass_kg=mass)
    _apply_scene_edits(env, edits)

    for entity in scope.values():
        obj = getattr(entity, "wrapped_obj", entity)
        if obj is None or getattr(obj, "name", None) in robot_names or getattr(obj, "kinematic_only", True):
            continue
        try:
            obj.sleep_threshold = 0.0; obj.wake()
        except Exception:  # noqa: BLE001
            pass
    for _ in range(args.settle):
        og.sim.step()

    print(f"\nafter a {args.settle}-step settle:")
    bad = []
    for cargo_id, floor_id in starts.items():
        box = getattr(scope[cargo_id], "wrapped_obj", scope[cargo_id]); final = box.get_position_orientation()[0]
        on_floor = bool(box.states[object_states.OnTop].get_value(scope[floor_id]))
        here = room_of([float(final[0]), float(final[1])])
        floor_room = stands_in(getattr(scope[floor_id], "wrapped_obj", scope[floor_id]))
        print(f"  {cargo_id:16} ({float(final[0]):+.3f}, {float(final[1]):+.3f}, {float(final[2]):+.3f}) in {here:18} "
              f"ontop({floor_id}): {on_floor}   (floor stands in {floor_room})")
        if not on_floor:
            bad.append(cargo_id)
    if bad:
        print(f"\nNOT ON THEIR FLOORS: {bad} -- not saving"); og.shutdown(); sys.exit(1)

    goal_met, breakdown = env.task.compiled_task.check_goal(env.task._evaluate_predicate)
    print(f"\ngoal already met: {goal_met}  {breakdown}   (must be False)")
    assert not goal_met, "the goal holds at t=0 -- the task would be trivial"

    if args.no_save:
        print("\n--no_save given; nothing written")
    else:
        env.scene.update_initial_file()
        env.task.save_task(env=env, save_dir=args.save_dir, override=True, task_relevant_only=False)
        fname = env.task.get_cached_activity_scene_filename(
            scene_model=scene_model, activity_name=activity, activity_definition_id=0,
            activity_instance_id=args.instance_id)
        print(f"\nwrote instance {fname}")
        written = _written_template_path(args.save_dir, fname, scene_model=scene_model)
        if written is not None:
            print(f"stripped the sampler's robots from the template: {strip_robots_from_template(written)}")
        else:
            print("WARNING: could not locate the written template to strip its robots")
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
