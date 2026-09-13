"""Write each routed activity's description.txt from its route.json + BDDL.

One sentence per carried object, by its id, listing in order the supports it
must be set on and the room each is in -- "Move die.n.01_1, in this order,
onto: the cabinet in the child's room, the bookcase in the kitchen, ...".
No weights, no word "route" (user, 2026-09-13): the lift rule is in the
system prompt and the marked state lines follow the sentence in ids. Where
one sentence names the same kind of support in the same room twice, the id
is added in parentheses so the two can be told apart.

    python feasibility_verify/make_task_descriptions.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NOUN = {
    "cabinet.n.01": "cabinet", "bookcase.n.01": "bookcase", "electric_refrigerator.n.01": "refrigerator",
    "armchair.n.01": "armchair", "breakfast_table.n.01": "breakfast table", "coffee_table.n.01": "coffee table",
    "bed.n.01": "bed", "footstool.n.01": "footstool", "desk.n.01": "desk", "countertop.n.01": "countertop",
    "sink.n.01": "sink", "straight_chair.n.01": "chair", "sofa.n.01": "sofa", "floor.n.01": "floor",
}
ROOM = {
    "childs_room": "child's room", "kitchen": "kitchen", "dining_room": "dining room", "living_room": "living room",
    "bedroom": "bedroom", "private_office": "private office", "bathroom": "bathroom",
    "television_room": "television room", "playroom": "playroom", "corridor": "corridor",
}
ACTIVITIES = [f"v4_s{s}_v4_{l}" for s in (1, 2, 3) for l in ("ll", "lh", "hl", "hh")]


def phrase(support: str, room: str, disambiguate: bool) -> str:
    synset = support.rsplit("_", 1)[0]
    noun = NOUN.get(synset, synset.split(".")[0].replace("_", " "))
    where = ROOM.get(room, (room or "").replace("_", " "))
    text = f"the {noun} in the {where}" if where else f"the {noun}"
    return f"{text} ({support})" if disambiguate else text


def describe(spec) -> str:
    sentences = []
    for route in spec.routes:
        keys = [(n.support.rsplit("_", 1)[0], n.room) for n in route.nodes]
        dup = {k for k in keys if keys.count(k) > 1}
        stops = [phrase(n.support, n.room, (n.support.rsplit("_", 1)[0], n.room) in dup) for n in route.nodes]
        sentences.append(f"Move {route.cargo}, in this order, onto: " + ", ".join(stops) + ".")
    return " ".join(sentences)


def main() -> int:
    from coop2.behavior_env.route_spec import load_route_spec
    from coop2.behavior_env.task_description import description_file_for

    for activity in ACTIVITIES:
        spec = load_route_spec(activity)
        text = describe(spec)
        with open(description_file_for(activity), "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"{activity}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
