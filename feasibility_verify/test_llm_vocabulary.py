"""The three-piece M5 consistency check: schema <-> L2 <-> outcome codes.

Structured output *constrains* the model, so the Pydantic union is the exact
set of things an agent can ever say. If it disagrees with L2's verb table in
either direction the failure is quiet and expensive: a verb only in the schema
becomes a wasted decision every time it is chosen, and a verb only in L2 is
dead code the model can never reach.

Needs no GPU and no API key -- it compares two tables.

Run:
    python feasibility_verify/test_llm_vocabulary.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.action.behavior_action import (  # noqa: E402
    BEHAVIOR_ACTION_SCHEMA,
    BEHAVIOR_ACTION_TO_PRIMITIVE,
    COMMUNICATION_ACTIONS,
)
from coop2.cognitive.agent.base_llm_agent import BaseLLMAgent  # noqa: E402
from coop2.cognitive.agent.llm_client import LLMAction, Task, parse_task  # noqa: E402


def main() -> int:
    def ok(message):
        print(f"  PASS {message}")

    schema_verbs = {
        model.model_fields["action_type"].default for model in LLMAction.__args__
    }
    l2_verbs = set(BEHAVIOR_ACTION_TO_PRIMITIVE) | set(COMMUNICATION_ACTIONS)

    # Verbs L2 must accept that the LLM never emits, with the reason. L3's plan
    # executor returns {"action_type": "noop"} on its no-plan path, so treating
    # noop as unknown turned every barrier-closed step into a reported failure.
    INTERNAL_ONLY = {"noop"}

    print("test 1: the LLM schema and L2 agree exactly")
    only_schema = schema_verbs - l2_verbs
    only_l2 = l2_verbs - schema_verbs - INTERNAL_ONLY
    assert not only_schema, f"the model can emit verbs L2 rejects: {sorted(only_schema)}"
    assert not only_l2, f"L2 handles verbs the model can never emit: {sorted(only_l2)}"
    assert INTERNAL_ONLY <= l2_verbs, f"L2 dropped an internal verb: {sorted(INTERNAL_ONLY - l2_verbs)}"
    ok(f"{len(schema_verbs)} emittable verbs + {sorted(INTERNAL_ONLY)} from L3")

    print("test 2: no crafter vocabulary survives")
    # These were emittable before the rewrite; L2 would have judged every one of
    # them 'invalid', so a whole episode would have been wasted decisions.
    for stale in ("move", "collect", "craft", "sleep", "noop", "navigate", "place"):
        assert stale not in schema_verbs, f"crafter verb {stale!r} still in the schema"
    ok("move/collect/craft/sleep/noop/navigate/place all gone")

    print("test 3: BEHAVIOR_ACTION_SCHEMA covers the same verbs")
    assert set(BEHAVIOR_ACTION_SCHEMA) == schema_verbs, (
        set(BEHAVIOR_ACTION_SCHEMA) ^ schema_verbs
    )  # the prompt-facing table lists only what the model may say, so no noop
    ok("the prompt-facing schema table matches the Pydantic union")

    print("test 4: every target-taking action really has a target field")
    for model in LLMAction.__args__:
        verb = model.model_fields["action_type"].default
        needs_target = bool(
            [f for f in BEHAVIOR_ACTION_SCHEMA[verb] if f.get("type") == "entity_id"]
        )
        has_target = "target" in model.model_fields
        assert needs_target == has_target, f"{verb}: schema says target={needs_target}, model says {has_target}"
    ok("target fields line up with the schema table")

    print("test 5: a task is one free string, and a predicate one parses back")
    # It was four fields -- an enum, object_type, an always-1 object_id, a
    # reference -- and the model filled them in whether or not the plan had a
    # goal (user, 2026-09-14). One string now, read back only where the ids are
    # actually needed.
    assert parse_task("ontop(apple.n.01_1, table.n.02_1)") == ("ontop", "apple.n.01_1", "table.n.02_1")
    assert parse_task("open(electric_refrigerator.n.01_1)") == ("open", "electric_refrigerator.n.01_1", None)
    # A name that is not a predicate parses to nothing, and must not raise:
    # that is the whole point of letting the team name its own tasks.
    for free in ("standby(jackal_1)", "transport die to bedroom", "", "waiting"):
        parsed = parse_task(free)
        assert parsed is None or parsed[0] == "standby", (free, parsed)
    for token in (t.value for t in Task):
        assert token.islower() and " " not in token, token
    ok("ontop(a, b) and open(a) parse; free names parse to None without raising")

    print("test 6: every goal predicate has an action that can achieve it")
    from coop2.cognitive.agent.cognitive_agent import _ensure_task_terminal_action  # noqa: PLC0415

    for task in Task:
        actions = []
        _ensure_task_terminal_action(f"{task.value}(apple.n.01_1, table.n.02_1)", actions)
        assert actions, f"no terminal action for goal {task.value!r}"
        assert actions[0].action_type in l2_verbs, actions[0].action_type
    ok("a plan that states a goal but omits the achieving action gets one appended")

    agent = object.__new__(BaseLLMAgent)
    agent.plan_count, agent.agent_id, agent.env_step = 3, "agent_0", 1234
    fallback = agent._generate_fallback_plan()
    # The fallback is issued where nothing can check it -- the LLM is already
    # down -- so it is the one plan whose verbs are never model output. It used
    # to be crafter's move/collect, which the engine answered `invalid` and `do`.
    assert fallback.actions, "a fallback with no actions freezes the world"
    for action in fallback.actions:
        assert action.action_type in l2_verbs or action.action_type in COMMUNICATION_ACTIONS, action.action_type
    ok("the fallback plan is built from verbs L2 actually has")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
