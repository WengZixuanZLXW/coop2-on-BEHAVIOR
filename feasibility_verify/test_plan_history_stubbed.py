"""CPU-only checks on the agent's own plan history in the prompt.

Every call used to start from a blank slate: the model was told the world and
its current plan, and nothing about what it had already tried. So it re-proposed
plans that had just failed, and the reason they failed -- a teammate holds the
target, there is no floor space around it -- was never in front of it.

Three things have to hold, and none is visible by reading one file:
the model's reasoning has to survive `parse_plan_response`, an outcome has to be
recorded exactly where a plan terminates, and an interrupt must not look like
one.

Run:
    python feasibility_verify/test_plan_history_stubbed.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.agent import SimpleAgent
from coop2.cognitive.agent.cognitive_agent import parse_plan_response
from coop2.cognitive.agent.llm_client import (
    LLMPlanResponse, NavigateToAction, Task, TaskSpecification,
)
from coop2.cognitive.agent.prompts import build_observation_prompt, format_plan_history
from coop2.cognitive.action.action import SymbolicAction
from coop2.cognitive.plan.plan import SymbolicPlan, SymbolicPlanStatus


def ok(message: str) -> None:
    print(f"  ok: {message}")


def main() -> int:
    print("test 1: the model's reasoning survives the parse")
    response = LLMPlanResponse(
        task=TaskSpecification(task=Task.ONTOP, object_type="apple.n.01_1",
                               reference="coffee_table.n.01_1"),
        actions=[NavigateToAction(target="apple.n.01_1")],
        reasoning="agent_0 is nearest this apple and nobody has claimed it",
    )
    plan = parse_plan_response(response, agent_id="agent_0", env_step=0, plan_id=1)
    # It was dropped here: the model explained every plan and the explanation
    # was discarded at the door.
    assert plan.metadata.get("reasoning") == response.reasoning, plan.metadata
    ok("parse_plan_response keeps it in plan.metadata")

    print("test 2: only a finished plan becomes an outcome")
    agent = SimpleAgent("agent_0")
    agent.plan = plan
    assert agent.plan_history == []

    # Interrupted, then resumed. The same plan is still running, so recording it
    # would report the robot as having done something it is still doing.
    plan.status = SymbolicPlanStatus.INTERRUPTED
    assert agent.plan_history == [], "an interrupt is not an outcome"

    plan.status = SymbolicPlanStatus.EXECUTING
    plan.complete_success(step=42)
    agent.record_plan_outcome(plan, True, "", env_step=42)
    assert len(agent.plan_history) == 1, agent.plan_history
    entry = agent.plan_history[0]
    assert entry["succeeded"] and entry["plan_id"] == 1
    assert entry["reasoning"] == response.reasoning, "the why did not reach the history"
    ok("success recorded with its reasoning; interrupt+resume recorded nothing")

    print("test 3: a failure carries the reason the agent needs")
    failed = parse_plan_response(response, agent_id="agent_0", env_step=42, plan_id=2)
    agent.record_plan_outcome(
        failed, False,
        "grasp: PRE_CONDITION_ERROR: apple_122 is currently held by agent_3",
        env_step=80,
    )
    block = format_plan_history(agent.plan_history)
    assert "#1 [DONE]" in block and "#2 [FAILED]" in block, block
    assert "held by agent_3" in block, block
    assert "you chose it because" in block, block
    ok("both outcomes render, the failure with its cause")

    print("test 4: a long primitive error is cut down, not pasted whole")
    agent.record_plan_outcome(
        failed, False,
        "An error occurred during each attempt of this action.\n\nAttempt 0: "
        "PLANNING_ERROR: Cannot reach picture_zsirgc_0: there is no free floor "
        "space around it to stand on (need a spot 1.3-1.9 m away, clear of "
        "walls, furniture and other agents). Another agent may already be "
        "standing there. Additional info: {'object': 'picture_zsirgc_0', "
        "'sampling_range': [1.34, 1.94], 'rejected_by': {'room': 80}}",
        env_step=90,
    )
    line = [l for l in format_plan_history(agent.plan_history).splitlines()
            if "it failed because" in l][-1]
    # Five of these at full length would outweigh the room listing they exist
    # to inform, and they are newline-ridden besides.
    assert len(line) < 240, len(line)
    assert "\n" not in line and line.endswith("...")
    ok(f"folded to one line of {len(line)} chars")

    print("test 5: the block reaches the prompt, after the world and before memory")
    prompt = build_observation_prompt(
        env_step=100, agent_id="agent_0", agent_names=["agent_0", "agent_1"],
        symbolic_view="living_room_0:\n  - apple.n.01_1  -> grasp",
        plan_history=agent.plan_history,
    )
    assert "YOUR FINISHED PLANS" in prompt, prompt
    assert prompt.index("SYMBOLIC VIEW") < prompt.index("YOUR FINISHED PLANS"), (
        "the world it must decide against comes first, its own record second"
    )
    # An agent with no finished plans gets no empty heading.
    fresh = build_observation_prompt(
        env_step=0, agent_id="agent_0", agent_names=["agent_0"], plan_history=[],
    )
    assert "YOUR FINISHED PLANS" not in fresh
    ok("present when there is history, absent when there is none")

    print("test 6: only the last few, so the prompt does not grow without bound")
    many = [
        {"plan_id": i, "specification": f"spec_{i}", "reasoning": "r",
         "succeeded": True, "reason": ""}
        for i in range(20)
    ]
    block = format_plan_history(many, limit=5)
    assert "spec_19" in block and "spec_15" in block and "spec_14" not in block, block
    ok("5 most recent kept, older ones dropped")
    block = format_plan_history(many)
    assert "spec_19" in block and "spec_17" in block and "spec_16" not in block, block
    ok("and the default is the last three")

    print("test 7: the plan's actions are in the record, not only its goal")
    # Two plans can share "ontop(apple, table)" and differ in every step; "do
    # not do that again" needs the steps.
    block = format_plan_history(agent.plan_history)
    assert "plan: navigate_to(apple.n.01_1)" in block, block
    assert block.index("#1 [DONE]") < block.index("plan: navigate_to") < block.index("you chose it because"), block
    ok("each entry lists its actions, between the goal and the reasoning")

    print("test 8: a plan replaced after an interrupt is an outcome, marked as one")
    replaced = parse_plan_response(response, agent_id="agent_0", env_step=90, plan_id=3)
    agent.record_plan_outcome(
        replaced, False, "you replanned after an interrupt, before it finished",
        env_step=95, status="replaced",
    )
    block = format_plan_history(agent.plan_history)
    assert "#3 [REPLACED]" in block, block
    assert "it was abandoned because: you replanned" in block, block
    assert "#3 [FAILED]" not in block, "abandoning a plan is not failing it"
    assert agent.plan_history[-1]["status"] == "replaced"
    assert agent.plan_history[0]["status"] == "done" and agent.plan_history[1]["status"] == "failed"
    ok("REPLACED is its own mark, and older entries still derive their status")

    print("test 9: a team hold is not a plan the model made, and is hidden")
    hold = SymbolicPlan(
        specification="wait_for_team(alpha)",
        actions=[SymbolicAction(action_type="wait", args={"ticks": 600})] * 30,
        plan_id=4, agent_id="agent_0", created_at_step=100,
    )
    agent.record_plan_outcome(hold, False, "recalled", env_step=700, status="replaced")
    block = format_plan_history(agent.plan_history)
    assert "wait_for_team" not in block and "#4" not in block, block
    assert "#3 [REPLACED]" in block, "hiding the hold must not hide its neighbours"
    assert format_plan_history([agent.plan_history[-1]]) == "", "a history of only holds is no history"
    ok("recorded for the metrics, absent from the prompt")

    print("test 10: the record survives into the team prompt, per robot")
    from coop2.comm_topology.llm_team import create_llm_team_topology

    class Stub:
        model = "stub"

    agents = create_llm_team_topology(
        llm_client=Stub(), teams={"alpha": ["agent_0", "agent_1"]}, verbose=False,
    )
    brain = agents["agent_0"].brain
    for a in agents.values():
        a.symbolic_view = f"view for {a.agent_id}"
        a.observe({}, 0)
    agents["agent_0"].plan_history = agent.plan_history
    user = brain._build_team_prompt([brain.members["agent_0"], brain.members["agent_1"]])[1]["content"]
    a0 = user.index("=== ROBOT agent_0 ==="); a1 = user.index("=== ROBOT agent_1 ===")
    assert "YOUR FINISHED PLANS" in user[a0:a1], "agent_0's history is missing from its own block"
    assert "YOUR FINISHED PLANS" not in user[a1:], "agent_1 has no history and must show none"
    assert "plan: navigate_to(apple.n.01_1)" in user[a0:a1]
    ok("each robot's block carries its own plans and their actions")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
