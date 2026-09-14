"""CPU-only checks on `tag`: one team brain per DIG-TAG agent, one shared graph.

What is pinned is DIG-TAG's round on our team layer (their runtime tests,
`ma_crafter/tests/test_tag_runner.py`, said to teams): the graph is section
6 of every prompt, above the robots; a team's `tag_actions` land on the
shared graph in order before its `notify` goes out, so a notified team reads
what was done; a rejected action is recorded with the graph's reason and the
round goes on; a notification interrupts the named team, which answers with
the same two fields from its interrupt round; the budget drops a second
notification and is reset after an environment step; a notification that
arrives while a team is planning is answered before its plans go out; and a
run's record round-trips. No Isaac, no model.

Run:
    python feasibility_verify/test_tag_team_stubbed.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.agent import AgentState
from coop2.cognitive.agent.llm_client import (
    InterruptDecision,
    LLMTagInterruptResponse,
    LLMTagPlanResponse,
    NavigateToAction,
    TagToolCall,
    Task,
    TaskSpecification,
    TeamAgentInterruptDecision,
    TeamAgentPlan,
)
from coop2.cognitive.agent.prompt_sections import cooperation_section, current_messages_section
from coop2.cognitive.messages import MessageBroker
from coop2.comm_topology.llm_tag import (
    NOTIFY_MESSAGE_TYPE,
    TAG_MANUAL,
    TagTeamBrain,
    save_tag_outputs,
    shared_tag,
    tag_rounds_of,
)
from coop2.comm_topology.llm_team import (
    TEAM_BRAIN_ROLES,
    TeamBrain,
    create_llm_team_topology,
    reset_notify_budgets,
)
from coop2.dig_tag.tag import TAGParallelInterface


def ok(message: str) -> None:
    print(f"  ok: {message}")


USAGE = {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4, "latency_seconds": 0.1}

OPEN = TagToolCall(tool="open", goal="carry die.n.01_1 to C1", rule="ontop(die.n.01_1, cabinet.n.01_5)",
                   state="team_1: drone_1 doing it")
BAD_UPDATE = TagToolCall(tool="update", task="q9", state="nowhere")
ATTACH_Q1 = TagToolCall(tool="attach", task="q1", payload="the die is 2 m from drone_1")
UPDATE_Q1 = TagToolCall(tool="update", task="q1", state="team_2 saw it; we take C2")


class StubClient:
    """Answers the tag schemas: plans per robot, and scripted (actions, notify)
    per (team, kind); records what every call saw. A hook may run inside a
    team's planning call, to stand in for a notification arriving then."""

    model = "stub"

    def __init__(self):
        self.formats = []
        self.prompts = []
        self.script = {}            # (team, "plan"|"interrupt") -> [(tag_actions, notify), ...]
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
        return queue.pop(0) if queue else ([], [])

    def generate(self, messages, response_format=None, temperature=0.7):
        self.formats.append(response_format)
        self.prompts.append(messages)
        team = self._team_in(messages)
        members = self._members_in(messages)
        if response_format is LLMTagPlanResponse:
            self.plan_calls += 1
            hook = self.during_plan.pop(team, None)
            if hook is not None:
                hook()
            actions, notify = self._next(team, "plan")
            plans = [
                TeamAgentPlan(
                    agent_id=name,
                    task=TaskSpecification(task=Task.ONTOP, object_type="die.n.01_1", reference="cabinet.n.01_5"),
                    actions=[NavigateToAction(target="die.n.01_1")],
                    reasoning=f"{name} goes",
                )
                for name in members
            ]
            return LLMTagPlanResponse(plans=plans, reasoning="split", tag_actions=actions, notify=notify), dict(USAGE)
        if response_format is LLMTagInterruptResponse:
            self.interrupt_calls += 1
            actions, notify = self._next(team, "interrupt")
            decisions = [TeamAgentInterruptDecision(agent_id=name, decision=InterruptDecision.RESUME,
                                                    reasoning="keep going") for name in members]
            return LLMTagInterruptResponse(decisions=decisions, reasoning="noted", tag_actions=actions,
                                           notify=notify), dict(USAGE)
        raise AssertionError(f"unexpected response format {response_format}")

    def generate_team_plan(self, messages, temperature=0.7):
        raise AssertionError("the tag mode must ask for its own plan schema")

    def generate_team_interrupt_decision(self, messages, temperature=0.7):
        raise AssertionError("the tag mode must ask for its own interrupt schema")


def make_run(teams, notify_budget=1):
    client = StubClient()
    agents = create_llm_team_topology(
        llm_client=client, teams=teams, topology="tag", verbose=False, notify_budget=notify_budget
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

    print("test 1: the mode exists, wires nothing, and every brain shares one graph")
    assert "tag" in TEAM_BRAIN_ROLES
    sec4 = cooperation_section("tag")
    assert sec4.startswith("## 4. COOPERATION MODE: SHARED TASK GRAPH"), sec4[:80]
    for token in ("tag_actions", "notify", "section 5", "section 6", "budget", "INTERRUPTS"):
        assert token in sec4, token
    heading = current_messages_section("tag", [{"sender": "team_2", "content": "x"}], lambda m: m["content"])
    assert heading.startswith("## 7. MESSAGES RECEIVED NOW -- a team notified you"), heading
    client, agents, brains, broker = make_run(teams)
    for brain in brains.values():
        assert isinstance(brain, TagTeamBrain) and brain.COOPERATION_MODE == "tag"
        assert brain.wait_for == [] and brain.send_to == []
        assert brain.notify_left == 1
    assert len({id(b.tag) for b in brains.values()}) == 1
    assert shared_tag(agents) is brains["team_1"].tag
    system = brains["team_1"]._system_prompt(agents["drone_1"])
    assert "## 4. COOPERATION MODE: SHARED TASK GRAPH" in system
    assert system.rstrip().endswith("## 5. DIGTAG\n" + TAG_MANUAL.strip()), system[-200:]
    for tool in ("open(goal, rule, state)", "update(task, state)", "close(identity)", "attach(task, payload)"):
        assert tool in system, tool
    ok("three TagTeamBrains, one graph, empty wiring, sections 4 and 5 in place")

    print("test 2: the empty graph is section 6, above the robots, with the budget")
    b1 = brains["team_1"]
    user = b1._build_team_prompt(members(b1))[1]["content"]
    assert "DIGTAG TASK OBSERVATION:" in user, user[:400]
    slot = user.index("DIGTAG TASK OBSERVATION:")
    assert user.index("## 6. OBSERVATIONS") < slot < user.index("=== ROBOT drone_1 ===")
    assert "(none yet" in user and "Notify budget left until the next environment step: 1" in user
    closing = user.rsplit("\n\n", 1)[1]
    assert "`tag_actions`" in closing and "`notify`" in closing, closing
    assert "## 7." not in user, "nothing was received"
    ok("graph rendered empty in the reserved slot, before the first robot; closing asks for both fields")

    print("test 3: actions land in order (a rejected one recorded), then the notification goes out")
    client.script[("team_1", "plan")] = [([OPEN, BAD_UPDATE, ATTACH_Q1], ["team_2", "team_1", "nobody"])]
    agents["drone_2"]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    plans = b1._generate_team_plans()
    assert set(plans) == {"drone_1", "jackal_1"}
    assert client.formats == [LLMTagPlanResponse], client.formats
    graph = b1.tag.tag
    assert list(graph.tasks) == ["q1"] and graph.tasks["q1"].identity == "k1"
    assert graph.tasks["q1"].state == "team_1: drone_1 doing it"
    assert [a.issuer for a in graph.actions.values()] == ["team_1"]
    (phi,) = graph.evidence.values()
    assert (phi.target, phi.source) == ("q1", "team_1")
    (round1,) = b1.tag_rounds
    assert round1["stage"] == "planning" and round1["observed"] == []
    results = [a["result"] for a in round1["tag_actions"]]
    assert results[0] == "applied" and results[1].startswith("rejected") and results[2] == "applied", results
    assert "q9" in results[1]
    assert round1["notify"] == {"requested": ["team_2"], "sent": ["team_2"], "dropped": False}, round1["notify"]
    sent = notifications(broker)
    assert len(sent) == 1 and sent[0]["sender"] == "team_1" and sent[0]["recipients"] == ["team_2"], sent
    assert sent[0]["metadata"]["interrupts_execution"] is True
    assert agents["drone_2"].state is AgentState.I, agents["drone_2"].state
    assert b1.notify_left == 0
    ok("open applied, update of an unknown version rejected, attach applied; team_2 interrupted, self and strangers dropped")

    print("test 4: the notified team answers from its interrupt round with the same two fields")
    client.script[("team_2", "interrupt")] = [([UPDATE_Q1], ["team_1"])]
    agents["drone_2"].handle_interrupt()
    assert client.formats[-1] is LLMTagInterruptResponse
    prompt = client.prompts[-1][1]["content"]
    assert "## 7. MESSAGES RECEIVED NOW -- a team notified you" in prompt
    assert "notification from team_1" in prompt
    assert "q1 [task k1] goal: carry die.n.01_1 to C1" in prompt, "the open had landed before team_2 was woken"
    assert "history: open by team_1 -> q1" in prompt
    assert "evidence phi1 by team_1: the die is 2 m from drone_1" in prompt
    assert list(graph.tasks) == ["q1", "q2"] and graph.tasks["q2"].identity == "k1"
    assert graph.tasks["q2"].state == "team_2 saw it; we take C2"
    b2 = brains["team_2"]
    (round2,) = b2.tag_rounds
    assert round2["stage"] == "interrupted" and round2["observed"] == ["q1"]
    assert round2["notify"]["sent"] == ["team_1"]
    # team_1's robots were in R (planning), which the broker does not stop:
    # the notification waits in their inboxes.
    assert len(notifications(broker)) == 2
    assert agents["drone_1"].state is AgentState.R
    assert [m["sender"] for m in agents["drone_1"].get_messages(clear_buffer=False)] == ["team_2"]
    ok("team_2 read q1 and its history, updated it to q2 under k1, notified team_1 back")

    print("test 5: a second notification is dropped, and the budget resets after a step")
    client.script[("team_1", "plan")] = [([], ["team_3"])]
    agents["tiago_3"]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    b1._generate_team_plans()
    assert b1.tag_rounds[-1]["notify"] == {"requested": ["team_3"], "sent": [], "dropped": True}
    assert len(notifications(broker)) == 2 and agents["tiago_3"].state is AgentState.W
    reset_notify_budgets(agents)
    assert all(b.notify_left == 1 for b in brains.values())
    client.script[("team_1", "plan")] = [([], ["team_3"])]
    b1._generate_team_plans()
    assert b1.tag_rounds[-1]["notify"]["sent"] == ["team_3"] and agents["tiago_3"].state is AgentState.I
    ok("budget of 1: second notify dropped and recorded; reset after the step; then sent")

    print("test 6: a notification that arrives during the planning call is answered before the plans go out")
    client, agents, brains, broker = make_run({"team_1": ["drone_1"], "team_2": ["drone_2"]})
    b1, b2 = brains["team_1"], brains["team_2"]
    # team_2 notifies team_1 while team_1's model call is in flight: team_1's
    # robot is in R, so the broker buffers rather than interrupts.
    client.during_plan["team_1"] = lambda: b2._say(
        "notification from team_2: read the shared task graph", NOTIFY_MESSAGE_TYPE,
        interrupts=True, recipients=["team_1"])
    client.script[("team_1", "plan")] = [([OPEN], [])]
    client.script[("team_1", "interrupt")] = [([ATTACH_Q1], [])]
    agents["drone_1"]._set_state(AgentState.R, timestamp=0.0, env_step=0)
    plan = agents["drone_1"].handle_reasoning()
    assert plan is not None and plan.specification.startswith("ontop("), plan.specification
    assert client.plan_calls == 1 and client.interrupt_calls == 1
    stages = [r["stage"] for r in b1.tag_rounds]
    assert stages == ["planning", "late_notify"], stages
    late_prompt = client.prompts[-1][1]["content"]
    assert "The team was interrupted by a message." in late_prompt
    assert "notification from team_2" in late_prompt
    assert "q1 [task k1]" in late_prompt, "the late round saw the graph as the planning round left it"
    assert list(b1.tag.tag.evidence.values())[0].target == "q1"
    assert agents["drone_1"].get_messages(clear_buffer=False) == [], "drained: it was answered"
    assert b1._heard == []
    assert agents["drone_1"].state is AgentState.R
    ok("planning round, then a late-notify round on the same graph, before the robot took its plan")

    print("test 7: the run's record: tag.json round-trips, rounds are in order, the graph draws")
    client, agents, brains, broker = make_run(teams)
    client.script[("team_1", "plan")] = [([OPEN, ATTACH_Q1], ["team_2"])]
    client.script[("team_2", "interrupt")] = [([UPDATE_Q1], [])]
    agents["drone_2"]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    brains["team_1"]._generate_team_plans()
    agents["drone_2"].handle_interrupt()
    with tempfile.TemporaryDirectory() as tmp:
        lines = save_tag_outputs(agents, tmp)
        assert any(l.startswith("Task graph saved") for l in lines), lines
        assert not any(l.startswith("Task graph not drawn") for l in lines), lines
        for name in ("tag.json", "tag_rounds.json", "tag.pdf", "tag.png"):
            assert os.path.exists(os.path.join(tmp, name)), (name, os.listdir(tmp))
        loaded = TAGParallelInterface.load_json(os.path.join(tmp, "tag.json"))
        assert [v.identity for v in loaded.tag.tasks.values()] == ["k1", "k1"]
        assert loaded.tag.tasks["q2"].state == "team_2 saw it; we take C2"
        assert len(loaded.tag.evidence) == 1
        with open(os.path.join(tmp, "tag_rounds.json"), encoding="utf-8") as handle:
            rounds = json.load(handle)
    assert [r["team"] for r in rounds] == ["team_1", "team_2"]
    assert [r["seq"] for r in rounds] == sorted(r["seq"] for r in rounds)
    assert rounds == tag_rounds_of(agents)
    assert set(rounds[0]) >= {"seq", "team", "env_step", "stage", "observed", "reasoning", "tag_actions", "notify"}
    ok("tag.json reloads with both versions and the evidence; rounds ordered; pdf and png written")

    print("test 8: the other modes are untouched")

    class Plain:
        model = "stub"

    plain = create_llm_team_topology(llm_client=Plain(), teams={"t": ["a"]}, verbose=False)["a"].brain
    assert type(plain) is TeamBrain and not hasattr(plain, "tag") and not hasattr(plain, "reset_notify_budget")
    reset_notify_budgets({"a": plain.members["a"]})
    for a in plain.members.values():
        a.symbolic_view = "v"
        a.observe({}, 0)
    text = plain._build_team_prompt(members(plain))[1]["content"]
    assert "DIGTAG TASK OBSERVATION" not in text and "tag_actions" not in text
    assert "## 5. DIGTAG" not in plain._system_prompt(plain.members["a"])
    ok("individual mode has no graph, no manual, no budget")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
