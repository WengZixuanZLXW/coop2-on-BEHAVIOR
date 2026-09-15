"""CPU-only checks on `decentralized_messageboard`: individual planning, one board.

What is pinned: the mode has its own section 4 and no wiring (nobody waits,
nobody is addressed); the board is section 7 of every planning prompt, with
the other teams' posts since this team last planned marked (new) and its own
labelled "You"; the plan call asks for `board_post` and appends it to the one
board every brain in the run shares, so the next team to plan reads it; the
notify tool is reserved -- off by default, no field offered, and even when
switched on it only records unless a deliverer is installed.

Run:
    python feasibility_verify/test_messageboard_stubbed.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.llm_client import (
    LLMMessageboardNotifyPlanResponse,
    LLMMessageboardPlanResponse,
    NavigateToAction,
    NotifyRequest,
    Task,
    TeamAgentPlan,
)
from coop2.cognitive.agent.prompt_sections import (
    COOPERATION_RULES,
    MESSAGEBOARD_NOTIFY_RULES,
    cooperation_section,
)
from coop2.comm_topology.llm_team import (
    TEAM_BRAIN_ROLES,
    MessageboardTeamBrain,
    TeamBrain,
    create_llm_team_topology,
    reset_notify_budgets,
)
from coop2.comm_topology.message_board import MessageBoard, render_board


def ok(message: str) -> None:
    print(f"  ok: {message}")


USAGE = {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4, "latency_seconds": 0.1}


class StubClient:
    """Answers the board schema with a plan per robot and a scripted post."""

    model = "stub"

    def __init__(self):
        self.calls = 0
        self.formats = []
        self.prompts = []
        self.post = "we take die.n.01_1: drone_1 grasps it and places it on C1; C2 onward is yours"
        self.notify = None

    @staticmethod
    def _members_in(messages):
        text = messages[-1]["content"]
        return [line.split()[2] for line in text.split("\n") if line.startswith("=== ROBOT ")]

    def generate(self, messages, response_format=None, temperature=0.7):
        self.calls += 1
        self.formats.append(response_format)
        self.prompts.append(messages)
        plans = [
            TeamAgentPlan(
                agent_id=name,
                task="ontop(die.n.01_1, bed.n.01_1)",
                actions=[NavigateToAction(target="die.n.01_1")],
                reasoning=f"{name} goes",
            )
            for name in self._members_in(messages)
        ]
        fields = dict(plans=plans, reasoning="split", board_post=self.post)
        if response_format is LLMMessageboardNotifyPlanResponse:
            return LLMMessageboardNotifyPlanResponse(notify=self.notify, **fields), dict(USAGE)
        return LLMMessageboardPlanResponse(**fields), dict(USAGE)

    def generate_team_plan(self, messages, temperature=0.7):
        raise AssertionError("the board mode must ask for its own response format")


def make_run(teams):
    client = StubClient()
    agents = create_llm_team_topology(
        llm_client=client, teams=teams, topology="decentralized_messageboard", verbose=False
    )
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    brains = {name: agents[members[0]].brain for name, members in teams.items()}
    return client, agents, brains


def members(brain):
    return [brain.members[n] for n in brain.member_ids]


def main() -> int:
    teams = {"team_1": ["drone_1", "jackal_1"], "team_2": ["drone_2"], "team_3": ["tiago_3"]}

    print("test 1: the mode exists, has its own section 4, and wires nothing")
    assert "decentralized_messageboard" in TEAM_BRAIN_ROLES
    sec4 = cooperation_section("decentralized_messageboard")
    assert sec4.startswith("## 4. COOPERATION MODE: DECENTRALIZED MESSAGE BOARD"), sec4[:80]
    for token in ("board_post", "(new)", "individual mode", "nothing you write interrupts"):
        assert token in sec4, token
    assert MESSAGEBOARD_NOTIFY_RULES.strip() not in sec4, "notify is reserved, not in the default text"
    client, agents, brains = make_run(teams)
    for brain in brains.values():
        assert isinstance(brain, MessageboardTeamBrain)
        assert brain.wait_for == [] and brain.send_to == [], (brain.wait_for, brain.send_to)
        assert brain.COOPERATION_MODE == "decentralized_messageboard"
    assert len({id(b.board) for b in brains.values()}) == 1, "one board per run"
    system = brains["team_1"]._system_prompt(agents["drone_1"])
    assert "## 4. COOPERATION MODE: DECENTRALIZED MESSAGE BOARD" in system
    assert "NOTIFY (a tool)" not in system
    ok("three brains, one shared board, empty wiring, board section 4 without the notify rule")

    print("test 2: an empty board is shown as empty, in section 7, between 6 and 9")
    b1 = brains["team_1"]
    user = b1._build_team_prompt(members(b1))[1]["content"]
    assert "## 7. MESSAGE BOARD" in user, user
    sec7 = user.split("## 7. MESSAGE BOARD", 1)[1]
    assert "nothing posted yet" in sec7.split("\n\n")[0]
    assert user.index("## 6. OBSERVATIONS") < user.index("## 7. MESSAGE BOARD")
    assert "## 7. MESSAGES RECEIVED NOW" not in user, "no direct messages in this mode"
    assert "`board_post`" in user.rsplit("\n\n", 1)[1], "the closing asks for the post"
    ok("empty board rendered, closing instruction asks for board_post")

    print("test 3: the planning call uses the board schema and posts to the board")
    plans = b1._generate_team_plans()
    assert set(plans) == {"drone_1", "jackal_1"}
    assert client.formats == [LLMMessageboardPlanResponse], client.formats
    assert len(b1.board) == 1
    post = b1.board.posts()[0]
    assert post.team == "team_1" and post.content == client.post and post.round == 1, post
    ok("board_post appended with the team's name, in the same call as the plans")

    print("test 4: the next team to plan reads it as (new); the poster reads it as its own")
    b2 = brains["team_2"]
    user2 = b2._build_team_prompt(members(b2))[1]["content"]
    sec7 = user2.split("## 7. MESSAGE BOARD", 1)[1].split("\n\n")[0]
    assert "team_1 (new): we take die.n.01_1" in sec7, sec7
    assert "You (" not in sec7
    # Seen once, it is no longer new the next time team_2 plans.
    user2b = b2._build_team_prompt(members(b2))[1]["content"]
    sec7b = user2b.split("## 7. MESSAGE BOARD", 1)[1].split("\n\n")[0]
    assert "team_1: we take die.n.01_1" in sec7b and "(new)" not in sec7b, sec7b
    # The poster sees its own line labelled You, never as new.
    user1 = b1._build_team_prompt(members(b1))[1]["content"]
    sec7c = user1.split("## 7. MESSAGE BOARD", 1)[1].split("\n\n")[0]
    assert "You (team_1): we take die.n.01_1" in sec7c and "(new)" not in sec7c, sec7c
    ok("(new) once per reader, own posts labelled You")

    print("test 5: posts accumulate oldest first, with step numbers; blank posts are dropped")
    client.post = "team_2: drone_2 waits; C2 is ours once team_1 places on C1"
    b2._generate_team_plans()
    client.post = "   "
    brains["team_3"]._generate_team_plans()
    assert len(b1.board) == 2, "a blank post posts nothing"
    user3 = brains["team_3"]._build_team_prompt(members(brains["team_3"]))[1]["content"]
    sec7 = user3.split("## 7. MESSAGE BOARD", 1)[1].split("\n\n")[0]
    lines = [line for line in sec7.split("\n") if line.strip().startswith("[step")]
    assert len(lines) == 2 and "team_1" in lines[0] and "team_2" in lines[1], lines
    assert all(line.strip().startswith("[step 0]") for line in lines), lines
    ok("two posts in order, each stamped with its env step")

    print("test 6: the run's record")
    records = b1.board.to_records()
    assert [p["team"] for p in records["posts"]] == ["team_1", "team_2"]
    assert records["notifies"] == []
    ok("to_records lists posts and (empty) notify requests")

    print("test 7: the notify tool is reserved -- schema, rule and delivery all behind one flag")
    assert "notify" not in LLMMessageboardPlanResponse.model_fields
    assert "notify" in LLMMessageboardNotifyPlanResponse.model_fields
    assert MessageboardTeamBrain.NOTIFY_TOOL_ENABLED is False
    # Switched on, on one brain: the schema widens, section 4 gains the rule,
    # and a request is recorded on the board -- and delivered to nobody,
    # because no deliverer is installed.
    b3 = brains["team_3"]
    b3.NOTIFY_TOOL_ENABLED = True
    try:
        assert b3.plan_response_format is LLMMessageboardNotifyPlanResponse
        assert "NOTIFY (a tool)" in b3._system_prompt(agents["tiago_3"]).split("## 4.")[1]
        client.post = "tiago_3 holds die.n.01_1"
        client.notify = NotifyRequest(teams=["team_1", "team_3"], content="stop: we hold the die", reasoning="they are heading for it")
        b3._generate_team_plans()
        notifies = b3.board.notifies()
        assert len(notifies) == 1 and notifies[0].sender == "team_3", notifies
        assert notifies[0].targets == ["team_1"], "the sender is not among its own targets"
        assert notifies[0].delivered == [], "no broker is attached, so there was nowhere to send"
        assert b1._heard == [] and agents["drone_1"].get_messages(clear_buffer=False) == [], "nobody was interrupted"
        assert b3.notify_left == 1, "an undeliverable request spends no budget"
        # With a broker the factory's deliverer sends: the named team's robots
        # are interrupted, the budget is spent, and the record says who.
        from coop2.cognitive.agent.agent import AgentState
        from coop2.cognitive.messages import MessageBroker
        broker = MessageBroker(agents)
        for agent in agents.values():
            agent.message_broker = broker
        for name in ("drone_1", "jackal_1"):
            agents[name]._set_state(AgentState.W, timestamp=0.0, env_step=0)
        b3._generate_team_plans()
        record = b3.board.notifies()[-1]
        assert record.delivered == ["team_1"], record
        assert agents["drone_1"].state is AgentState.I and agents["jackal_1"].state is AgentState.I
        sent = [m for m in broker.get_message_log() if (m.get("metadata") or {}).get("type") == "notify"]
        assert len(sent) == 1 and sent[0]["sender"] == "team_3" and sent[0]["recipients"] == ["team_1"], sent
        assert sent[0]["metadata"]["interrupts_execution"] is True
        assert b3.notify_left == 0
        b3._generate_team_plans()
        assert b3.board.notifies()[-1].delivered == [], "over the budget: dropped"
        assert len([m for m in broker.get_message_log()
                    if (m.get("metadata") or {}).get("type") == "notify"]) == 1
        reset_notify_budgets(agents)
        assert b3.notify_left == 1
        assert b3.board.to_records()["notifies"][1]["delivered"] == ["team_1"]
    finally:
        b3.NOTIFY_TOOL_ENABLED = False
    assert "NOTIFY (a tool)" not in b3._system_prompt(agents["tiago_3"])
    ok("off: no field, no rule; on: delivered through the broker as an interrupt, within the budget")

    print("test 8: the plain brain is untouched")
    class Plain:
        model = "stub"
    plain = create_llm_team_topology(llm_client=Plain(), teams={"t": ["a"]}, verbose=False)["a"].brain
    assert type(plain) is TeamBrain and not hasattr(plain, "board")
    for a in plain.members.values():
        a.symbolic_view = "v"; a.observe({}, 0)
    text = plain._build_team_prompt(members(plain))[1]["content"]
    assert "MESSAGE BOARD" not in text and "board_post" not in text
    assert "## 4. COOPERATION MODE: INDIVIDUAL" in plain._system_prompt(plain.members["a"])
    ok("individual mode has no board and asks for no post")

    print("test 9: render_board on its own")
    board = MessageBoard()
    board.post("team_1", "x" * 300, env_step=5)
    board.post("team_2", "short", env_step=9)
    text = render_board(board.posts(), reader="team_2", new_after=1)
    assert "[step 5] team_1: " in text and "..." in text, "old posts are cut short"
    assert "[step 9] You (team_2): short" in text
    text2 = render_board(board.posts(), reader="team_3", new_after=0)
    assert "team_1 (new): " in text2 and "team_2 (new): short" in text2
    assert "..." not in text2, "new posts are shown whole"
    ok("truncation only for old posts; new ones whole")

    assert "decentralized_messageboard" in COOPERATION_RULES
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
