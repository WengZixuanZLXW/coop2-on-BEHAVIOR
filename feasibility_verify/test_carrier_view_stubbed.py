"""CPU-only: what a base-locked arm and a carrier can be *shown* to do.

The engine already refuses to drive with a loaded arm, and already gates
load_onto/unload_from on reach. This is the other half: a verb the prompt
offers and the engine refuses costs a plan and an LLM round trip to discover,
and a verb the engine accepts but the prompt never offers is never used at all.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/zixuanwe/Desktop/BEHAVIOR-1K")
import coop2.behavior_env.symbolic_view as sv
from coop2.behavior_env.world_state import EntityObservation, SymbolicObservation

ROOM = "childs_room_0"
BOX = "packing_box.n.02_1"


def scene(agent, *, box_held_by=None, jackal_carrying=None, jackal_at=(0.5, 0.0),
          ridgeback_locked=True):
    ridgeback = EntityObservation("agent_0", "agent_0", "agent", [ROOM], position=(0, 0, 0),
                                  is_robot=True, base_locked_while_holding=ridgeback_locked)
    jackal = EntityObservation("agent_1", "agent_1", "agent", [ROOM],
                               position=(*jackal_at, 0.0), is_robot=True,
                               is_carrier=True, carrying=jackal_carrying)
    box = EntityObservation(BOX, "box_0", "packing_box", [ROOM], position=(0.3, 0, 0.5),
                            held_by=box_held_by)
    floor = EntityObservation("floor.n.01_1", "floor_0", "floor", [ROOM],
                              position=(0, 0, 0), is_fixed=True)
    entities = [ridgeback, jackal, box, floor]
    return SymbolicObservation(
        agent_id=agent, step=1, max_steps=2500, room=ROOM,
        entities={e.entity_id: e for e in entities}, facts=[],
        task_entity_ids={e.entity_id for e in entities},
    )


def verbs(observation, radius=1.5):
    return {(h.primitive, h.target_id)
            for h in sv.target_hints(observation, interaction_radius=radius)}


def main() -> int:
    print("test 1: a locked base holding something is offered no navigate_to")
    holding = scene("agent_0", box_held_by="agent_0")
    got = verbs(holding)
    assert not [v for v in got if v[0] == "navigate_to"], sorted(got)
    # The verbs that do not need driving survive.
    assert ("place_on_top", "floor.n.01_1") in got, sorted(got)
    assert ("release", BOX) in got
    text = sv.render_symbolic_view(holding, interaction_radius=1.5)
    # The consequence is an absence, and an absence explains nothing.
    assert "base is locked" in text and "load_onto a carrier" in text, text
    print("  ok: no navigate_to anywhere, and the Holding line says why")

    print("\ntest 2: the same robot with an empty hand drives normally")
    empty = scene("agent_0", box_held_by=None)
    got = verbs(empty)
    assert ("navigate_to", "floor.n.01_1") in got, sorted(got)
    assert ("navigate_to", BOX) in got
    assert ("grasp", BOX) in got
    assert "base is locked" not in sv.render_symbolic_view(empty, interaction_radius=1.5)
    print("  ok: the constraint is about the load, not about the robot")

    print("\ntest 3: a robot that never declared the constraint is untouched")
    free = scene("agent_0", box_held_by="agent_0", ridgeback_locked=False)
    assert ("navigate_to", "floor.n.01_1") in verbs(free)
    print("  ok: only a robot that says its base locks is grounded")

    print("\ntest 4: a carrier in reach is a target for the cargo in your hand")
    got = verbs(holding)
    assert ("load_onto", "agent_1") in got, sorted(got)
    assert ("unload_from", "agent_1") not in got, "nothing is up there yet"
    print("  ok: load_onto offered, unload_from withheld")

    print("\ntest 5: a loaded carrier, with an empty hand, offers unload_from")
    loaded = scene("agent_0", box_held_by="agent_1", jackal_carrying=BOX)
    got = verbs(loaded)
    assert ("unload_from", "agent_1") in got, sorted(got)
    assert ("load_onto", "agent_1") not in got, "your hand is empty"
    # And the box is reachable through its carrier and through nothing else --
    # not grasp, which would weld a second joint onto one that already has one,
    # and not navigate_to, which aims at where it sits on a robot about to drive.
    assert not [v for v in got if v[1] == BOX], sorted(got)
    text = sv.render_symbolic_view(loaded, interaction_radius=1.5)
    # One line, on the carrier: the id, what is on it, and how to get it back.
    line = next(l for l in text.splitlines() if l.strip().startswith("- agent_1"))
    assert f"[carrier, carrying {BOX}]" in line and "unload_from" in line, line
    assert not [l for l in text.splitlines() if l.strip().startswith("- " + BOX)], text
    print("  ok: carrier and cargo are one line, and unload_from is the only way back")

    print("\ntest 6: out of reach, neither verb is offered")
    far = scene("agent_0", box_held_by="agent_0", jackal_at=(6.0, 0.0))
    got = verbs(far)
    assert ("load_onto", "agent_1") not in got, sorted(got)
    assert ("unreachable", "agent_1") in got
    # Grounded, so it cannot even drive over -- and the note has to say so, or
    # "unreachable" reads as "navigate_to first" and there is no navigate_to.
    note = next(h.note for h in sv.target_hints(far, interaction_radius=1.5)
                if h.primitive == "unreachable" and h.target_id == "agent_1")
    assert "base is locked" in note, note
    print("  ok: gated on reach, like every other manipulation")

    print("\ntest 7: a carrier is not holding what it carries")
    # It has no arm. Reading cargo as held offered it place_on_top and release,
    # for an object it cannot let go of.
    own = scene("agent_1", box_held_by="agent_1", jackal_carrying=BOX)
    text = sv.render_symbolic_view(own, interaction_radius=1.5)
    assert "Holding: nothing" in text, text
    # A robot is filtered out of its own listing, so without this the carrier is
    # the one agent that cannot see what it is carrying.
    assert f"On your back: {BOX}" in text and "unload_from(you)" in text, text
    got = verbs(own)
    assert not [v for v in got if v[0] in ("release", "place_on_top")], sorted(got)
    assert ("navigate_to", "floor.n.01_1") in got, "a carrier's whole job is driving"
    assert ("load_onto", "agent_1") not in got, "cannot load onto itself"
    print("  ok: it drives, and is offered neither its own back nor an arm's verbs")

    print("\ntest 8: the full handoff is expressible at every step")
    # A -> B -> C -> D, each state offering exactly the next move of
    #   grasp -> load_onto -> (carrier drives) -> unload_from -> place_on_top
    a = verbs(scene("agent_0", box_held_by=None))
    assert ("grasp", BOX) in a
    b = verbs(scene("agent_0", box_held_by="agent_0"))
    assert ("load_onto", "agent_1") in b and not [v for v in b if v[0] == "navigate_to"]
    c = verbs(scene("agent_1", box_held_by="agent_1", jackal_carrying=BOX))
    assert ("navigate_to", "floor.n.01_1") in c
    d = verbs(scene("agent_0", box_held_by="agent_1", jackal_carrying=BOX))
    assert ("unload_from", "agent_1") in d
    e = verbs(scene("agent_0", box_held_by="agent_0"))
    assert ("place_on_top", "floor.n.01_1") in e
    print("  ok: grasp -> load_onto -> drive -> unload_from -> place_on_top")

    print("\ntest 9: the world model reports cargo as held, and says by whom")
    # The observation half above is only as true as what feeds it. This runs the
    # real BehaviorWorldState against a registered load.
    sys.path.insert(0, "/home/zixuanwe/Desktop/BEHAVIOR-1K/feasibility_verify")
    import test_world_state_stubbed as T  # noqa: PLC0415
    import coop2.behavior_env.world_state as ws  # noqa: PLC0415
    from coop2.behavior_env import carrier  # noqa: PLC0415

    seg = T.FakeSegMap()
    ridgeback = T.FakeObject("agent_0", "agent", [0.0, 0.0, 0.0])
    jackal = T.FakeObject("agent_1", "agent", [0.5, 0.0, 0.0])
    jackal.is_carrier = True
    box = T.FakeObject("box_0", "packing_box", [0.5, 0.0, 0.4])
    scene_ = T.FakeScene([ridgeback, jackal, box], seg_map=seg)
    world = ws.BehaviorWorldState(T.FakeEnv(scene_, [ridgeback, jackal]))
    world.start()

    assert world.held_objects() == {}, world.held_objects()
    carrier._CARGO[jackal.name] = ("/World/agent_1/cargo_constraint", box)
    try:
        world.step()
        assert world.held_objects() == {"box_0": "agent_1"}, world.held_objects()
        entities = world.entities()
        assert entities["agent_1"].carrying == world.entity_id_for(box), entities["agent_1"]
        assert entities[world.entity_id_for(box)].held_by == "agent_1"
        assert entities["agent_0"].carrying is None
        # And the cross-robot view every gate asks.
        assert carrier.carrier_holding(box, [ridgeback, jackal]) is jackal
        assert carrier.carrier_holding(box, [ridgeback]) is None
    finally:
        carrier._CARGO.pop(jackal.name, None)
    assert world.held_objects() == {}, "a cleared load must free the object"
    print("  ok: held_objects covers cargo, and the carrier says what it carries")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
