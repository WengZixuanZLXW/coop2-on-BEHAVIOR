"""Write the eight S2/S3 activity definitions (problem0.bddl + route.json) and the
two scene-edit files, from one table per scene.

COOHAVIOR's Beechwood coordinates are in a different frame from our dataset's
(its S2 stations fall outside our rooms, its S3 stations in the wrong rooms),
and several of the objects it names do not exist in our scenes
(breakfast_table_skczfi_3/4, countertop_tpuwys_6, every shelf_owvfik_*). So
nothing here is a coordinate: each route node is a support of the same kind
in the same room TYPE as COOHAVIOR's, taken from our scene as it stands, and
the cargo starts on the station's floor wherever the sampler puts it. Where
our scene lacks the kind in that room (S2's private office has no breakfast
table) the room's real furniture stands in (its two desks). Decision: the
route is the task and the BDDL follows it (user, 2026-09-12); a support that
is not in the scene is fixed by choosing one that is (user, 2026-09-13).

    python feasibility_verify/make_v4_s2_s3_definitions.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFS = os.path.join(ROOT, "bddl3", "bddl", "activity_definitions")
LAYOUTS = os.path.join(ROOT, "coop2", "team_layouts")
SCRATCH_DEACT = "/tmp/claude-1004/-home-zixuanwe-Desktop-BEHAVIOR-1K/32498237-bea7-46c4-af35-3f295ae2b8e9/scratchpad"

CARGO = {"ll": ("die.n.01", 0.008), "hl": ("die.n.01", 0.008),
         "lh": ("notebook.n.01", 0.02), "hh": ("notebook.n.01", 0.02)}
LIFT = {"die.n.01": ["arm", "drone"], "notebook.n.01": ["arm"]}

# Per scene: stations in COOHAVIOR's route order, each with its floor's room
# type and instance; supports as (our instance id, room type, COOHAVIOR name,
# our runtime candidates); the serial route (LL/LH) as node ids; the parallel
# routes (HL/HH) as (station index, C1 support, D support).
SCENES = {
    # BEHAVIOR's sampler binds `inroom <type>` objects to ONE room instance per
    # type (the intersection of their candidate rooms must be non-empty), so a
    # route cannot span both of Beechwood_0_int's living rooms or both of
    # Beechwood_1_int's children's rooms. COOHAVIOR's fifth S2 station
    # (living_room_0) and fourth S3 station (childs_room_1) therefore move to
    # the nearest room of another type that still has a support after the
    # deactivations: S2-G5 to bathroom_0 (its sink), S3-H4 to bathroom_1 (a
    # low cabinet and its sink). Found by the sampler's own error on
    # 2026-09-13; see PORTING_COOHAVIOR.md "S2 and S3".
    "s2": {
        "scene": "Beechwood_0_int",
        "coohavior": "S2",
        "stations": [  # (station, floor id, room type, room instance)
            ("G1", "floor.n.01_1", "dining_room", "dining_room_0"),
            ("G2", "floor.n.01_2", "living_room", "living_room_1"),
            ("G3", "floor.n.01_3", "private_office", "private_office_0"),
            ("G4", "floor.n.01_4", "kitchen", "kitchen_0"),
            ("G5", "floor.n.01_5", "bathroom", "bathroom_0"),
        ],
        "supports": {
            "coffee_table.n.01_1":     ("dining_room",    "breakfast_table_dnsjnv_0 (their category; ours is coffee_table_dnsjnv_0)"),
            "footstool.n.01_1":        ("living_room",    "ottoman_miftfy_0 (ottoman -> footstool.n.01 in our taxonomy)"),
            "breakfast_table.n.01_1":  ("living_room",    "breakfast_table_skczfi_1"),
            "desk.n.01_1":             ("private_office", "breakfast_table_skczfi_4 -- not in our scene; the office's desk_rzyfxk_0 stands in"),
            "desk.n.01_2":             ("private_office", "breakfast_table_skczfi_3 -- not in our scene; desk_rzyfxk_1 stands in"),
            "bookcase.n.01_1":         ("kitchen",        "shelf_owvfik_0 (our bookcase_owvfik_0)"),
            "countertop.n.01_1":       ("kitchen",        "countertop_tpuwys_6 -- not in our scene, and COOHAVIOR deactivates every tpuwys we have; countertop_jveutp_N stands in"),
            "sink.n.01_1":             ("bathroom",       "G5 was living_room_0 (shelf_owvfik_1, breakfast_table_skczfi_0): a second living_room instance cannot bind; furniture_sink_zexzrc_0 in bathroom_0 stands in"),
            "straight_chair.n.01_1":   ("dining_room",    "straight_chair_dmcixv_2"),
        },
        "serial": ["coffee_table.n.01_1", "footstool.n.01_1", "breakfast_table.n.01_1", "desk.n.01_1",
                   "desk.n.01_2", "bookcase.n.01_1", "countertop.n.01_1", "sink.n.01_1", "straight_chair.n.01_1"],
        "parallel": [  # (station index, C1, D)
            (0, "coffee_table.n.01_1", "footstool.n.01_1"),
            (1, "footstool.n.01_1", "desk.n.01_1"),
            (2, "desk.n.01_1", "bookcase.n.01_1"),
            (3, "countertop.n.01_1", "sink.n.01_1"),
            (4, "sink.n.01_1", "straight_chair.n.01_1"),
        ],
    },
    "s3": {
        "scene": "Beechwood_1_int",
        "coohavior": "S3",
        "stations": [
            ("H1", "floor.n.01_1", "bedroom", "bedroom_0"),
            ("H2", "floor.n.01_2", "television_room", "television_room_0"),
            ("H3", "floor.n.01_3", "childs_room", "childs_room_0"),
            ("H4", "floor.n.01_4", "bathroom", "bathroom_1"),
            ("H5", "floor.n.01_5", "playroom", "playroom_0"),
        ],
        "supports": {
            "cabinet.n.01_1":          ("bedroom",         "bottom_cabinet_jhymlr_0 (theirs, moved into the bedroom; ours already there is bottom_cabinet_jhymlr_1)"),
            "sofa.n.01_1":             ("television_room", "sofa_qnnwfx_0"),
            "cabinet.n.01_2":          ("television_room", "bottom_cabinet_jhymlr_2 (theirs; ours in that room is bottom_cabinet_jhymlr_0)"),
            "bed.n.01_1":              ("childs_room",     "bed_zrumze_2 (childs_room_0)"),
            "breakfast_table.n.01_1":  ("childs_room",     "breakfast_table_skczfi_1 (childs_room_0)"),
            "cabinet.n.01_3":          ("bathroom",        "H4 was childs_room_1 (bed_zrumze_1, bottom_cabinet_jhymlr_1): a second childs_room instance cannot bind; bottom_cabinet_no_top_pluwfl_N in bathroom_1 stands in"),
            "sink.n.01_1":             ("bathroom",        "multi_station_furniture_sink_yfaufu_0 in bathroom_1 stands in for bed_zrumze_1"),
            "breakfast_table.n.01_2":  ("playroom",        "breakfast_table_skczfi_0"),
            "breakfast_table.n.01_3":  ("playroom",        "breakfast_table_uhrsex_0"),
            "bed.n.01_2":              ("bedroom",         "bed_zrumze_0 (the destination; COOHAVIOR's bed.n.01_3)"),
        },
        "serial": ["cabinet.n.01_1", "sofa.n.01_1", "cabinet.n.01_2", "bed.n.01_1", "breakfast_table.n.01_1",
                   "cabinet.n.01_3", "sink.n.01_1", "breakfast_table.n.01_2", "breakfast_table.n.01_3", "bed.n.01_2"],
        "parallel": [
            (0, "cabinet.n.01_1", "sofa.n.01_1"),
            (1, "sofa.n.01_1", "bed.n.01_2"),
            (2, "bed.n.01_1", "cabinet.n.01_3"),
            (3, "cabinet.n.01_3", "breakfast_table.n.01_2"),
            (4, "breakfast_table.n.01_3", "bed.n.01_2"),
        ],
    },
}


def synset_of(instance: str) -> str:
    return instance.rsplit("_", 1)[0]


def objects_block(instances):
    by_synset = {}
    for inst in instances:
        by_synset.setdefault(synset_of(inst), []).append(inst)
    # One synset per line: BDDL's parser assigns, so a second line for the
    # same synset silently drops the first.
    return "\n".join(f"        {' '.join(insts)} - {syn}" for syn, insts in by_synset.items())


def write_bddl(tag, level, spec):
    activity = f"v4_{tag}_v4_{level}"
    cargo_syn, _ = CARGO[level]
    serial = level in ("ll", "lh")
    stations = spec["stations"]
    if serial:
        cargos = [f"{cargo_syn}_1"]
        floors = [stations[0][1]]
        starts = [(cargos[0], stations[0][1], stations[0][2])]
        goals = [(cargos[0], spec["serial"][-1])]
        node_supports = list(spec["serial"])
    else:
        cargos = [f"{cargo_syn}_{i+1}" for i in range(5)]
        floors = [st[1] for st in stations]
        starts = [(cargos[i], stations[si][1], stations[si][2]) for i, (si, _, _) in enumerate(spec["parallel"])]
        goals = [(cargos[i], d) for i, (_, _, d) in enumerate(spec["parallel"])]
        node_supports = []
        for _, c1, d in spec["parallel"]:
            for s in (c1, d):
                if s not in node_supports:
                    node_supports.append(s)
    # Every support the scene needs is declared in every task of the scene, so
    # the four instances of one scene are the same scene (S1's rule).
    supports = list(spec["supports"])
    instances = cargos + floors + supports + ["agent.n.01_1"]
    init = []
    for cargo, floor, room in starts:
        init.append(f"        (ontop {cargo} {floor})")
    seen = set()
    for _, floor, room, _inst in stations:
        if floor in floors and floor not in seen:
            init.append(f"        (inroom {floor} {room})"); seen.add(floor)
    for inst, (room, _) in spec["supports"].items():
        init.append(f"        (inroom {inst} {room})")
    init.append(f"        (ontop agent.n.01_1 {stations[0][1]})")
    goal = "\n".join(f"            (ontop {c} {d})" for c, d in goals)
    policy = "serial package dependency chain" if serial else "independent parallel packages"
    comment = f"""    ; COOHAVIOR {spec['coohavior']}-V4-{level.upper()} on {spec['scene']}: {policy}.
    ; {'One box, ten ordered supports through five stations' if serial else 'Five boxes, one per station; each goes one station on'};
    ; the order lives in route.json beside this file and the BDDL states the
    ; final support(s). Written by feasibility_verify/make_v4_s2_s3_definitions.py.
    ;
    ; COOHAVIOR's coordinates for this scene are in another frame and several of
    ; its objects are not in our dataset, so every support is a same-kind object
    ; in the same room type of our own scene, and the cargo starts wherever the
    ; sampler puts it on the station's floor. The mapping, node by node:
""" + "\n".join(f"    ;   {inst:26} {room:16} <- {their}" for inst, (room, their) in spec["supports"].items()) + """
    ; One synset per line -- the parser keeps only the last `- <synset>` group.
"""
    text = f"""(define (problem {activity}-0)
    (:domain behavior-1k)

{comment}
    (:objects
{objects_block(instances)}
    )

    (:init
{chr(10).join(init)}
    )

    (:goal
        (and
{goal}
        )
    )
)
"""
    d = os.path.join(DEFS, activity); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "problem0.bddl"), "w") as f:
        f.write(text)
    return activity, cargos, node_supports


def write_route(tag, level, spec, activity, cargos):
    cargo_syn, _ = CARGO[level]
    serial = level in ("ll", "lh")
    routes = []
    if serial:
        last = len(spec["serial"]) - 1
        nodes = [{"id": f"C{i+1}" if i < last else "D", "goal": ["ontop", cargos[0], s]}
                 for i, s in enumerate(spec["serial"])]
        routes.append({"id": f"{spec['coohavior']}-{spec['stations'][0][0]}", "cargo": cargos[0], "nodes": nodes})
    else:
        for i, (si, c1, d) in enumerate(spec["parallel"]):
            routes.append({"id": f"{spec['coohavior']}-{spec['stations'][si][0]}", "cargo": cargos[i],
                           "nodes": [{"id": "C1", "goal": ["ontop", cargos[i], c1]},
                                     {"id": "D", "goal": ["ontop", cargos[i], d]}]})
    payload = {
        "_comment": [
            f"The ordered sub-goals of COOHAVIOR {spec['coohavior']}-V4-{level.upper()} on {spec['scene']}, which the",
            "BDDL beside this file cannot state. Read by coop2.behavior_env.route_spec.",
            "Transcribed from COOHAVIOR/behavior_style_task/tasks.modified.json; supports are",
            "our scene's same-kind objects in COOHAVIOR's room types (see problem0.bddl's",
            "mapping comment): its Beechwood coordinates are in another frame and some of",
            "its objects are not in our dataset. Written by",
            "feasibility_verify/make_v4_s2_s3_definitions.py.",
        ],
        "activity": activity,
        "source": f"COOHAVIOR tasks.modified.json {spec['coohavior']}-V4-{level.upper()}",
        "policy": "serial" if serial else "parallel",
        "routes": routes,
        "lift": {cargo_syn: LIFT[cargo_syn]},
    }
    with open(os.path.join(DEFS, activity, "route.json"), "w") as f:
        json.dump(payload, f, indent=2); f.write("\n")


def write_scene_edits(tag, spec):
    src = os.path.join(SCRATCH_DEACT, f"deact_{spec['scene']}.json")
    deact = json.load(open(src))
    payload = {
        "_comment": [
            f"Scene edits COOHAVIOR {spec['coohavior']}-V4-* make to {spec['scene']}, reproduced here.",
            "",
            f"deactivate: the `active = false` overs in the staging scene's own BehaviorScene",
            f"block ({spec['coohavior']}-V4-HH.usda; identical across the four difficulties), kept",
            f"where the name exists in our dataset's scene. {len(deact['absent_in_ours'])} of COOHAVIOR's do not:",
            "  " + ", ".join(deact["absent_in_ours"]),
            "",
            "No moves. COOHAVIOR moves its route supports to station centres, but its",
            "Beechwood frame is not ours (its stations fall outside or in the wrong rooms",
            "here), so the supports stay where our scene has them. The sampler keeps",
            "anything the BDDL binds to, whatever this list says.",
            "",
            "box_mass_kg is per task (LL/HL 0.008, LH/HH 0.02); the sampler overrides it.",
        ],
        "deactivate": deact["deactivate"],
        "move": {},
        "box_mass_kg": 0.008,
    }
    path = os.path.join(LAYOUTS, f"v4_{tag}_v4_ll_scene.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2); f.write("\n")
    return path


def main():
    for tag, spec in SCENES.items():
        for level in ("ll", "lh", "hl", "hh"):
            activity, cargos, _ = write_bddl(tag, level, spec)
            write_route(tag, level, spec, activity, cargos)
            print(f"wrote {activity}")
        print("wrote", write_scene_edits(tag, spec))


if __name__ == "__main__":
    main()
