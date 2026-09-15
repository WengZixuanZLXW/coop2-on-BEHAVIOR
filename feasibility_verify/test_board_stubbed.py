"""CPU-only checks on `board`: DIG-TAG's message-board ablation on our team
layer -- `tag` with the graph replaced by an unstructured shared board.

The ablation's claim is that the two modes differ **only** in the shared
space, so this file checks the same eight things
`test_tag_team_stubbed.py` checks, against the board: the space is section 6
of every prompt above the robots, shown in full; a team's `write` lands
before its `notify` goes out, so a notified team reads it; a notification
interrupts the named team, which answers with the same two fields from its
interrupt round; the budget drops a second notification and is reset after an
environment step; a notification arriving mid-call is answered before the
plans go out; and a run's record round-trips.

The round is `notifying_team.NotifyingTeamBrain`, shared with `tag`, so this
also pins that the shared base works through the board's seams -- which is
the part a copy would have let drift. No Isaac, no model.

Run:
    python feasibility_verify/test_board_stubbed.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.agent import AgentState
from coop2.cognitive.agent.llm_client import (
    InterruptDecision,
    LLMBoardInterruptResponse,
    LLMBoardPlanResponse,
    NavigateToAction,
    TeamAgentInterruptDecision,
    TeamAgentPlan,
)
from coop2.cognitive.agent.prompt_sections import BEST_PRACTICES, cooperation_section, current_messages_section
from coop2.cognitive.messages import MessageBroker
from coop2.comm_topology.llm_board import (
    BOARD_MANUAL,
    NOTIFY_MESSAGE_TYPE,
    BoardTeamBrain,
    board_rounds_of,
    save_board_outputs,
    shared_board,
)
from coop2.comm_topology.llm_team import (
    TEAM_BRAIN_ROLES,
    TeamBrain,
    create_llm_team_topology,
    reset_notify_budgets,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


USAGE = {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4, "latency_seconds": 0.1}

CLAIM = "team_1: drone_1 is taking die.n.01_1 to C1, leave it to us"
SEEN = "team_2: we saw the die move; we take C2"
NOTE = "team_1: the die is 2 m from drone_1"


class StubClient:
    """Answers the board schemas: plans per robot, and a scripted (write,
    notify) per (team, kind); records what every call saw. A hook may run
    inside a team's planning call, to stand in for a notification arriving
    then. Same shape as the tag stub -- the modes take the same calls."""

    model = "stub"

    def __init__(self):
        self.formats = []
        self.prompts = []
        self.script = {}            # (team, "plan"|"interrupt") -> [(write, notify), ...]
        self.during_plan = {}       # team -> callable run inside that team's planning call
        self.plan_calls = 0
        self.interrupt_calls = 0

    @staticmethod
    def _members_in(messages):
        text = messages[-1]["content"]
        return [line.split()[2] for line in text.split("\n") if line.startswith("=== ROBOT ")]

    @staticmethod
    def _team_in(messages):
        return re.search(r"You command TEAM '([^']+)'", messages[0]["content"]).group(1)

    def _next(self, team, kind):
        queue = self.script.get((team, kind), [])
        return queue.pop(0) if queue else (None, [])

    def generate(self, messages, response_format=None, temperature=0.7):
        self.formats.append(response_format)
        self.prompts.append(messages)
        team = self._team_in(messages)
        members = self._members_in(messages)
        if response_format is LLMBoardPlanResponse:
            self.plan_calls += 1
            hook = self.during_plan.pop(team, None)
            if hook is not None:
                hook()
            write, notify = self._next(team, "plan")
            plans = [
                TeamAgentPlan(
                    agent_id=name,
                    task="ontop(die.n.01_1, cabinet.n.01_5)",
                    actions=[NavigateToAction(target="die.n.01_1")],
                    reasoning=f"{name} goes",
                )
                for name in members
            ]
            return LLMBoardPlanResponse(plans=plans, reasoning="split", write=write, notify=notify), dict(USAGE)
        if response_format is LLMBoardInterruptResponse:
            self.interrupt_calls += 1
            write, notify = self._next(team, "interrupt")
            decisions = [TeamAgentInterruptDecision(agent_id=name, decision=InterruptDecision.RESUME,
                                                    reasoning="keep going") for name in members]
            return LLMBoardInterruptResponse(decisions=decisions, reasoning="noted", write=write,
                                             notify=notify), dict(USAGE)
        raise AssertionError(f"unexpected response format {response_format}")

    def generate_team_plan(self, messages, temperature=0.7):
        raise AssertionError("the board mode must ask for its own plan schema")

    def generate_team_interrupt_decision(self, messages, temperature=0.7):
        raise AssertionError("the board mode must ask for its own interrupt schema")


def make_run(teams, notify_budget=1):
    client = StubClient()
    agents = create_llm_team_topology(
        llm_client=client, teams=teams, topology="board", verbose=False, notify_budget=notify_budget
    )
    broker = MessageBroker(agents)
    for agent in agents.values():
        agent.message_broker = broker
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    brains = {name: agents[members[0]].brain for name, members in teams.items()}
    return client, agents, brains, broker


def members(brain):
    return [brain.members[n] for n in brain.member_ids]


def notifications(broker):
    return [m for m in broker.get_message_log() if (m.get("metadata") or {}).get("type") == NOTIFY_MESSAGE_TYPE]


def main() -> int:
    teams = {"team_1": ["drone_1", "jackal_1"], "team_2": ["drone_2"], "team_3": ["tiago_3"]}

    print("test 1: the mode exists, wires nothing, and every brain shares one board")
    assert "board" in TEAM_BRAIN_ROLES
    sec4 = cooperation_section("board")
    assert sec4.startswith("## 4. COOPERATION MODE: SHARED MESSAGE BOARD"), sec4[:80]
    for token in ("`write`", "notify", "section 5", "section 6", "budget", "INTERRUPTS"):
        assert token in sec4, token
    heading = current_messages_section("board", [{"sender": "team_2", "content": "x"}], lambda m: m["content"])
    assert heading.startswith("## 7. MESSAGES RECEIVED NOW -- a team notified you"), heading
    client, agents, brains, broker = make_run(teams)
    for brain in brains.values():
        assert isinstance(brain, BoardTeamBrain) and brain.COOPERATION_MODE == "board"
        assert brain.wait_for == [] and brain.send_to == []
        assert brain.notify_left == 1
    assert len({id(b.board) for b in brains.values()}) == 1
    assert shared_board(agents) is brains["team_1"].board
    system = brains["team_1"]._system_prompt(agents["drone_1"])
    assert "## 4. COOPERATION MODE: SHARED MESSAGE BOARD" in system
    assert system.rstrip().endswith("## 5.1 DIGTAG MANUAL\n" + BOARD_MANUAL.strip()), system[-200:]
    assert "## 5. BEST PRACTICE" in system
    for practice in BEST_PRACTICES:
        assert practice in system, practice
    assert "write(text)" in system
    # The ablation is one tool against the graph's eight: nothing structural.
    for graph_tool in ("open(goal", "update(task", "close(identity)", "attach(task"):
        assert graph_tool not in system, graph_tool
    ok("three BoardTeamBrains, one board, empty wiring, sections 4 and 5 in place")

    print("test 2: the empty board is section 6, above the robots, with the budget")
    b1 = brains["team_1"]
    user = b1._build_team_prompt(members(b1))[1]["content"]
    assert "## 6.1 DIGTAG TASK OBSERVATION" in user, user[:400]
    slot = user.index("## 6.1 DIGTAG TASK OBSERVATION")
    assert user.index("## 6. OBSERVATIONS") < slot < user.index("=== ROBOT drone_1 ===")
    assert "(nothing posted yet)" in user, user[:400]
    assert "Notifications left until the next environment step: 1" in user, user[-400:]
    assert "one notification may name several teams" in user, user[-400:]
    closing = user.rsplit("\n\n", 1)[1]
    assert "`write`" in closing and "`notify`" in closing, closing
    assert "## 7." not in user, "nothing was received"
    ok("board rendered empty in the reserved slot, before the first robot; closing asks for both fields")

    print("test 3: the write lands before the notification goes out")
    client.script[("team_1", "plan")] = [(CLAIM, ["team_2", "team_1", "nobody"])]
    agents["drone_2"]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    plans = b1._generate_team_plans()
    assert set(plans) == {"drone_1", "jackal_1"}
    assert client.formats == [LLMBoardPlanResponse], client.formats
    board = b1.board
    assert [p["text"] for p in board.history()] == [CLAIM]
    assert board.history()[0]["author"] == "team_1" and board.history()[0]["index"] == 0
    (round1,) = b1.round_records
    assert round1["stage"] == "planning" and round1["observed"] == []
    assert round1["write"] == CLAIM
    assert round1["notify"] == {"requested": ["team_2"], "sent": ["team_2"], "dropped": False}, round1["notify"]
    sent = notifications(broker)
    assert len(sent) == 1 and sent[0]["sender"] == "team_1" and sent[0]["recipients"] == ["team_2"], sent
    assert sent[0]["metadata"]["interrupts_execution"] is True
    assert "message board" in sent[0]["content"]
    assert agents["drone_2"].state is AgentState.I, agents["drone_2"].state
    assert b1.notify_left == 0
    ok("the post landed; team_2 interrupted, self and strangers dropped")

    print("test 4: the notified team answers from its interrupt round with the same two fields")
    client.script[("team_2", "interrupt")] = [(SEEN, ["team_1"])]
    agents["drone_2"].handle_interrupt()
    assert client.formats[-1] is LLMBoardInterruptResponse
    prompt = client.prompts[-1][1]["content"]
    assert "## 7. MESSAGES RECEIVED NOW -- a team notified you" in prompt
    assert "notification from team_1" in prompt
    assert f"[0] step 0 team_1: {CLAIM}" in prompt, "the post had landed before team_2 was woken"
    assert [p["text"] for p in board.history()] == [CLAIM, SEEN]
    b2 = brains["team_2"]
    (round2,) = b2.round_records
    assert round2["stage"] == "interrupted" and round2["observed"] == ["0"]
    assert round2["write"] == SEEN and round2["notify"]["sent"] == ["team_1"]
    # team_1's robots were in R (planning), which the broker does not stop:
    # the notification waits in their inboxes.
    assert len(notifications(broker)) == 2
    assert agents["drone_1"].state is AgentState.R
    assert [m["sender"] for m in agents["drone_1"].get_messages(clear_buffer=False)] == ["team_2"]
    ok("team_2 read the post, wrote its own, notified team_1 back")

    print("test 5: a second notification is dropped, and the budget resets after a step")
    client.script[("team_1", "plan")] = [(None, ["team_3"])]
    agents["tiago_3"]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    b1._generate_team_plans()
    assert b1.round_records[-1]["write"] is None, "null write posts nothing"
    assert len(board.history()) == 2
    assert b1.round_records[-1]["notify"] == {"requested": ["team_3"], "sent": [], "dropped": True}
    assert len(notifications(broker)) == 2 and agents["tiago_3"].state is AgentState.W
    reset_notify_budgets(agents)
    assert all(b.notify_left == 1 for b in brains.values())
    client.script[("team_1", "plan")] = [("   ", ["team_3"])]
    b1._generate_team_plans()
    assert b1.round_records[-1]["write"] is None and len(board.history()) == 2, "blank write posts nothing"
    assert b1.round_records[-1]["notify"]["sent"] == ["team_3"] and agents["tiago_3"].state is AgentState.I
    ok("budget of 1: second notify dropped and recorded; reset after the step; then sent")

    print("test 6: a notification that arrives during the planning call is answered before the plans go out")
    client, agents, brains, broker = make_run({"team_1": ["drone_1"], "team_2": ["drone_2"]})
    b1, b2 = brains["team_1"], brains["team_2"]
    # team_2 notifies team_1 while team_1's model call is in flight: team_1's
    # robot is in R, so the broker buffers rather than interrupts.
    client.during_plan["team_1"] = lambda: b2._say(
        "notification from team_2: read the shared message board", NOTIFY_MESSAGE_TYPE,
        interrupts=True, recipients=["team_1"])
    client.script[("team_1", "plan")] = [(CLAIM, [])]
    client.script[("team_1", "interrupt")] = [(NOTE, [])]
    agents["drone_1"]._set_state(AgentState.R, timestamp=0.0, env_step=0)
    plan = agents["drone_1"].handle_reasoning()
    assert plan is not None and plan.specification.startswith("ontop("), plan.specification
    assert client.plan_calls == 1 and client.interrupt_calls == 1
    stages = [r["stage"] for r in b1.round_records]
    assert stages == ["planning", "late_notify"], stages
    late_prompt = client.prompts[-1][1]["content"]
    assert "The team was interrupted by a message." in late_prompt
    assert "notification from team_2" in late_prompt
    assert CLAIM in late_prompt, "the late round saw the board as the planning round left it"
    assert [p["text"] for p in b1.board.history()] == [CLAIM, NOTE]
    assert agents["drone_1"].get_messages(clear_buffer=False) == [], "drained: it was answered"
    assert b1._heard == []
    assert agents["drone_1"].state is AgentState.R
    ok("planning round, then a late-notify round on the same board, before the robot took its plan")

    print("test 7: the run's record: board.json round-trips, rounds are in order")
    client, agents, brains, broker = make_run(teams)
    client.script[("team_1", "plan")] = [(CLAIM, ["team_2"])]
    client.script[("team_2", "interrupt")] = [(SEEN, [])]
    agents["drone_2"]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    brains["team_1"]._generate_team_plans()
    agents["drone_2"].handle_interrupt()
    with tempfile.TemporaryDirectory() as tmp:
        lines = save_board_outputs(agents, tmp)
        assert any(l.startswith("Message board saved") for l in lines), lines
        for name in ("board.json", "board_rounds.json"):
            assert os.path.exists(os.path.join(tmp, name)), (name, os.listdir(tmp))
        with open(os.path.join(tmp, "board.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        with open(os.path.join(tmp, "board_rounds.json"), encoding="utf-8") as handle:
            rounds = json.load(handle)
    assert saved["schema"] == "message_board.v1"
    assert [p["text"] for p in saved["posts"]] == [CLAIM, SEEN]
    assert [p["index"] for p in saved["posts"]] == [0, 1]
    assert [r["team"] for r in rounds] == ["team_1", "team_2"]
    assert [r["seq"] for r in rounds] == sorted(r["seq"] for r in rounds)
    assert rounds == board_rounds_of(agents)
    assert set(rounds[0]) >= {"seq", "team", "env_step", "stage", "observed", "reasoning", "write", "notify"}
    ok("board.json holds every post in order; rounds ordered and complete")

    print("test 8: the board is `tag` minus the graph, and the other modes are untouched")
    # The ablation, structurally: both brains are the same round, and neither
    # mode can gain a field the other has not -- a copy is what let that drift.
    from coop2.comm_topology.llm_tag import TagTeamBrain
    from coop2.comm_topology.notifying_team import NotifyingTeamBrain
    assert issubclass(BoardTeamBrain, NotifyingTeamBrain) and issubclass(TagTeamBrain, NotifyingTeamBrain)
    shared = {"reset_notify_budget", "_round", "_notify", "after_plan", "_refresh_task_observation",
              "_build_team_prompt", "_build_interrupt_prompt", "_call_plan_model", "_call_interrupt_model",
              "_on_plan_response", "_on_interrupt_response", "_team_names"}
    for name in sorted(shared):
        assert name not in vars(BoardTeamBrain), f"{name} is the shared round's, not the board's"
        assert name not in vars(TagTeamBrain), f"{name} is the shared round's, not the graph's"
    seams = {"_observe_space", "_space_section", "_observed", "_record_fields", "_apply", "_summary"}
    for name in sorted(seams):
        assert name in vars(BoardTeamBrain) and name in vars(TagTeamBrain), name

    class Plain:
        model = "stub"

    plain = create_llm_team_topology(llm_client=Plain(), teams={"t": ["a"]}, verbose=False)["a"].brain
    assert type(plain) is TeamBrain and not hasattr(plain, "board") and not hasattr(plain, "reset_notify_budget")
    for a in plain.members.values():
        a.symbolic_view = "v"
        a.observe({}, 0)
    text = plain._build_team_prompt(members(plain))[1]["content"]
    assert "DIGTAG TASK OBSERVATION" not in text and "message board" not in text
    ok("both modes share the round and differ only in the six seams; individual mode has neither")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
