"""CPU-only: the activity's goal, rendered in the ids the agent must write.

A goal is a tree -- quantifiers and connectives over predicates -- and the
first version of this read it as a flat predicate with atom arguments. That
crashed `env.reset()` on `unhashable type: 'list'` for every activity whose
goal is quantified, i.e. both apple tasks, and survived a day because
`v4_s1_v4_ll` has the only flat goal in the repo and was the task in hand.

The other half is what a quantifier must NOT become: expanding
`forall ?apple.n.01` into its nine instances would name all nine apples in
every prompt from step 0 and delete the exploration `coop_nine_apples_hall`
exists to pose.

The three goals below are the real ones, copied from what
`Conditions(activity, 0, "behavior-1k")` parses -- not invented shapes.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/zixuanwe/Desktop/BEHAVIOR-1K")
from coop2.behavior_env.symbolic_view import render_goal_terms

NINE_APPLES_GOAL = [["forall", ["?apple.n.01", "-", "apple.n.01"],
                     ["ontop", "?apple.n.01", "?coffee_table.n.01_1"]]]
NINE_APPLES_INIT = [["inroom", "floor.n.01_1", "empty_room"]]

V4_GOAL = [["ontop", "packing_box.n.02_1", "floor.n.01_2"]]
V4_INIT = [["inroom", "floor.n.01_1", "childs_room"],
           ["inroom", "floor.n.01_2", "bedroom"]]


def main() -> int:
    print("1. a quantified goal renders at all (this is the crash)")
    text = render_goal_terms(NINE_APPLES_GOAL, NINE_APPLES_INIT)
    print(f"   {text}")
    assert text, "a quantified goal must produce text"

    print("\n2. it names the destination, without the `?` BDDL puts on every term")
    assert "coffee_table.n.01_1" in text, text
    assert "?" not in text, f"a `?`-prefixed id is not one the agent can act on: {text}"
    assert text.startswith("ontop("), f"the goal is still a predicate: {text}"

    print("\n3. it does NOT expand the quantifier into the instances")
    for instance in [f"apple.n.01_{i}" for i in range(1, 10)]:
        assert instance not in text, (
            f"{instance} leaked: naming the apples deletes the exploration problem")
    assert "every apple.n.01" in text, text

    print("\n4. a flat goal is unchanged, and still says which room to carry it to")
    v4 = render_goal_terms(V4_GOAL, V4_INIT)
    print(f"   {v4}")
    assert v4 == "ontop(packing_box.n.02_1, floor.n.01_2)   [floor.n.01_2 is in the bedroom]", v4

    print("\n5. the room note follows a concrete id out of a quantifier")
    scoped = render_goal_terms(
        [["forall", ["?apple.n.01", "-", "apple.n.01"],
          ["ontop", "?apple.n.01", "?coffee_table.n.01_1"]]],
        [["inroom", "coffee_table.n.01_1", "living_room"]])
    print(f"   {scoped}")
    assert "[coffee_table.n.01_1 is in the living_room]" in scoped, scoped

    print("\n6. nesting and the shapes we do not use are rendered, not crashed")
    for goal in (
        [["and", ["ontop", "?a_1", "?b_1"], ["not", ["ontop", "?a_1", "?c_1"]]]],
        [["exists", ["?x.n.01", "-", "x.n.01"], ["inside", "?x.n.01", "?box_1"]]],
        [["forpairs", ["?x.n.01", "-", "x.n.01"], ["?y.n.01", "-", "y.n.01"],
          ["ontop", "?x.n.01", "?y.n.01"]]],
        [["forn", ["2"], ["?x.n.01", "-", "x.n.01"], ["ontop", "?x.n.01", "?t_1"]]],
    ):
        rendered = render_goal_terms(goal, [])
        print(f"   {rendered}")
        assert rendered and "?" not in rendered, rendered

    print("\n7. no goal means no section, not an empty one")
    assert render_goal_terms([], []) is None
    assert render_goal_terms(None, None) is None
    assert render_goal_terms([[]], []) is None

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
