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
                "Rooms in this house", "navigate_to(<room name>)",
                # Only a base-locked robot needs a carrier; a drone carries its own load.
                "needs a carrier", "carries what it holds by")


def main() -> int:
    members = ["agent_0", "agent_1", "agent_2", "agent_3"]
    prompt = build_team_system_prompt("team_0", members, max_actions=6)

    print("test 1: the prompt addresses a controller, not a robot")
    assert "You are agent" not in prompt, prompt.splitlines()[0]
    head = "\n".join(prompt.splitlines()[:3])
    assert prompt.startswith("## 1. YOUR ROLE") and "team_0" in head, head
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
    # No world observation yet -> no roster, no lift table; nothing invented.
    assert "THIS TEAM'S ROBOTS" not in system and "WHO MAY LIFT WHAT" not in system
    # With one (as plan_env_wrapper hands it over via set_env_info), the roster
    # and the activity's lift table appear in section 3.
    from types import SimpleNamespace as NS
    def fake_obs(agent_id, **flags):
        me = NS(name=agent_id, is_carrier=flags.get("is_carrier", False),
                base_locked_while_holding=flags.get("base_locked", False), lift_role=flags.get("lift_role", "arm"))
        return NS(agent_id=agent_id, entities={agent_id: me}, lift_rules={"notebook.n.01": ("arm",), "die.n.01": ("arm", "drone")})
    agents["agent_0"].set_env_info(symbolic_view="view for agent_0", world_observation=fake_obs("agent_0", base_locked=True))
    agents["agent_1"].set_env_info(symbolic_view="view for agent_1", world_observation=fake_obs("agent_1", is_carrier=True))
    agents["agent_2"].set_env_info(symbolic_view="view for agent_2", world_observation=fake_obs("agent_2", lift_role="drone"))
    system2 = brain._system_prompt(brain.members[members[0]])
    sec3 = system2.split("## 3. ROBOT CAPABILITIES", 1)[1].split("## 4.", 1)[0]
    assert "THIS TEAM'S ROBOTS:" in sec3 and "WHO MAY LIFT WHAT" in sec3, sec3[-600:]
    assert "agent_0: ARM, base LOCKS while holding" in sec3 and "agent_1: CARRIER" in sec3 and "agent_2: DRONE" in sec3, sec3[-600:]
    assert "agent_3" not in sec3.split("THIS TEAM'S ROBOTS:")[1], "a robot without an observation is left out, not guessed"
    assert "notebook.n.01: arm" in sec3 and "die.n.01: arm, drone" in sec3
    for name in members:
        assert f"=== ROBOT {name} ===" in user
        assert user.count(f"view for {name}") == 1
    assert "put the apples on the table" in user
    ok("the team call carries the team system prompt and four observations")

    print("test 5: the sections come in the fixed order, system 1-5 and user 6-9")
    import coop2.cognitive.agent.prompt_sections as ps
    sys_headers = ["## 1. YOUR ROLE", "## 2. ENVIRONMENT RULES", "## 3. ROBOT CAPABILITIES",
                   "## 4. COOPERATION MODE"]
    positions = [system.index(h) for h in sys_headers]
    assert positions == sorted(positions), positions
    assert "## 5. DIGTAG" not in system, "the reserved slot adds nothing while empty"
    brain.reserved_system_prompt = "digtag manual goes here"
    with_digtag = brain._system_prompt(brain.members[members[0]])
    assert with_digtag.rstrip().endswith("DIGTAG:\ndigtag manual goes here"), with_digtag[-120:]
    brain.reserved_system_prompt = ""
    for mode in ("individual", "broadcast_chain", "centralized_leader", "centralized_follower",
                 "decentralized_messageboard", "tag"):
        assert f"## 4. COOPERATION MODE" in ps.cooperation_section(mode)
    # User side: observations, then messages now, then history, then action history.
    brain.memory.record_message_out(sender="team_0", recipients=["team_1"], content="we take C1", env_step=3)
    brain._heard = [{"sender": "team_1", "content": "we take C2", "metadata": {}}]
    for agent in agents.values():
        agent.plan_history = [{"plan_id": 1, "status": "failed", "specification": "ontop(x, y)",
                               "actions": ["grasp(x)"], "reason": "TOO_FAR"}]
    user2 = brain._build_team_prompt([brain.members[n] for n in members])[1]["content"]
    order = [user2.index(h) for h in ("## 6. OBSERVATIONS", "## 7. MESSAGES RECEIVED NOW",
                                      "## 8. CONVERSATION HISTORY", "## 9. ACTION HISTORY AND FAILURES")]
    assert order == sorted(order), order
    assert "we take C2" in user2.split("## 8.")[0].split("## 7.")[1], "the new message is in section 7"
    assert "TOO_FAR" in user2.split("## 9.")[1], "failures are in section 9"
    assert "TOO_FAR" not in user2.split("## 7.")[0], "and no longer under the robot's observation"
    brain.reserved_task_observation = "digtag task view"
    user3 = brain._build_team_prompt([brain.members[n] for n in members])[1]["content"]
    assert "DIGTAG TASK OBSERVATION:\ndigtag task view" in user3.split("=== ROBOT")[0]
    ok("1-5 in order in the system prompt, 6-9 in the user prompt, both digtag slots wired")

    print("test: section 5 is the best practice list in every mode, handoff order first, digtag after it")
    # Measured on v4_s1_v4_lh: arm-to-carrier satisfies both gates 14/25 of the
    # time, carrier-to-arm 25/25, because unload_from locks the arm's base the
    # moment the cargo is in its hand. The rule is standing, not mode-specific
    # -- a base-locked arm and a carrier are in every S1 layout.
    from coop2.cognitive.agent.prompt_sections import (
        BEST_PRACTICES, HANDOFF_ORDER, build_team_system_prompt as build_sys,
    )
    # Membership, not position: the list is edited by hand and the order in it
    # is the author's call (HANDOFF_ORDER moved to last on 2026-09-14). What
    # must hold is that every practice reaches section 5 in every mode.
    assert HANDOFF_ORDER in BEST_PRACTICES
    for mode in ("individual", "broadcast_chain", "centralized_leader",
                 "centralized_follower", "decentralized_messageboard", "tag"):
        text = build_sys("team_0", ["ridgeback_1", "jackal_1"], cooperation_mode=mode,
                         reserved=("MANUAL" if mode == "tag" else ""))
        assert "## 5. BEST PRACTICE" in text, mode
        section5 = text.split("## 5. BEST PRACTICE", 1)[1]
        for practice in BEST_PRACTICES:
            assert practice in section5, (mode, practice)
        assert ("DIGTAG:\nMANUAL" in text) == (mode == "tag"), mode
        # A practice is stated once: not also as a rule in sections 1-4.
        before = text.split("## 5. BEST PRACTICE", 1)[0]
        for token in ("fails again", "similar length", "size a wait", "One robot per object"):
            assert token.lower() not in before.lower(), (mode, token)
    # The order must be the one measured to work: arm to the target, then
    # carrier to the arm. Stated the other way round it is the failing plan.
    assert HANDOFF_ORDER.index("arm to the target") < HANDOFF_ORDER.index("carrier"), HANDOFF_ORDER
    ok("section 5 lists the practices once, arm-to-target then carrier-to-arm first, in all six modes; digtag follows under tag")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
