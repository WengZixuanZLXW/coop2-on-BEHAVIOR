"""CPU-only checks on the prompt one LLM gets when it drives a whole team.

The team path was borrowing the single-robot system prompt with the team's name
in the robot slot, so the call that has to answer with four plans was opening
with "You are agent 'team_0'" and being asked for one. Nothing failed -- the
model just worked from a description of a job it was not doing -- which is why
this is pinned by a test rather than left to reading.

Run:
    python feasibility_verify/test_team_prompt_stubbed.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.prompts import (
    ENV_DESCRIPTION,
    TEAM_ENV_DESCRIPTION,
    build_system_prompt,
    build_team_system_prompt,
)
from coop2.comm_topology.llm_team import create_llm_team_topology


def ok(message: str) -> None:
    print(f"  ok: {message}")


#: Rules the world imposes whoever is being told about them. They are worded
#: differently in the two descriptions (one speaks to a robot, the other about
#: robots), so only the load-bearing token is matched.
SHARED_RULES = ("TOO_FAR", "unreachable", "held by", "room listing",
                "ticks per metre", "navigate_to", "wait",
                # The carrier rules. Two robots that cannot do each other's job
                # is the premise of a whole activity, and the two descriptions
                # state it in different persons -- so they drift silently unless
                # every load-bearing token is required of both.
                "BASE_LOCKED", "load_onto", "unload_from", "carrier",
                "no arm", "On your back:", "within reach",
                # The route rule: only the NEXT support counts.
                "route", "NEXT", "out of order",
                # The lift rule, stated concretely: the drone cannot lift the notebook.
                "drone cannot lift the notebook", "CANNOT_LIFT",
                # Cross-room navigation: a task id is a target from any room.
                "valid navigate_to target", "through the house",
                # Rooms are navigable by name.
                "Rooms in this house", "navigate_to(<room name>)")


def main() -> int:
    members = ["agent_0", "agent_1", "agent_2", "agent_3"]
    prompt = build_team_system_prompt("team_0", members, max_actions=6)

    print("test 1: the prompt addresses a controller, not a robot")
    assert "You are agent" not in prompt, prompt.splitlines()[0]
    assert "team_0" in prompt.splitlines()[0], prompt.splitlines()[0]
    for name in members:
        assert name in prompt, f"{name} is not named in the prompt"
    # It is given four observations and must answer with four plans. The
    # single-robot prompt says "Choose one task and at most N actions", which
    # reads as one plan in total.
    assert "EVERY robot" in prompt, prompt
    ok("names the team and all four robots, and asks for a plan each")

    print("test 2: the world's rules survive the change of person")
    for rule in SHARED_RULES:
        assert rule in prompt, f"team prompt lost the rule {rule!r}"
        assert rule in ENV_DESCRIPTION, f"single-robot prompt lost the rule {rule!r}"
    # Two descriptions of one world drift apart silently; this is the guard.
    assert TEAM_ENV_DESCRIPTION in prompt
    ok(f"all {len(SHARED_RULES)} shared rules present in both descriptions")

    print("test 3: the single-robot prompt is untouched by this")
    solo = build_system_prompt("agent_0", max_actions=6)
    assert solo.startswith("You are agent 'agent_0'"), solo.splitlines()[0]
    assert "You command TEAM" not in solo
    ok("the per-robot topologies still get their own framing")

    print("test 4: a brain actually uses it")
    class Stub:
        model = "stub"

    agents = create_llm_team_topology(
        llm_client=Stub(), teams={"team_0": members}, verbose=False,
        goal_instruction="put the apples on the table",
    )
    brain = agents["agent_0"].brain
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    built = brain._build_team_prompt([brain.members[n] for n in members])
    system, user = built[0]["content"], built[1]["content"]
    assert "You command TEAM 'team_0'" in system, system.splitlines()[0]
    assert "TEAM CONTROLLER" in system
    for name in members:
        assert f"=== ROBOT {name} ===" in user
        assert user.count(f"view for {name}") == 1
    assert "put the apples on the table" in user
    ok("the team call carries the team system prompt and four observations")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
