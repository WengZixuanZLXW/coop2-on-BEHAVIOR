"""CPU-only: a chain team's broadcast is written by the model, in the plan call.

What it replaces was `_plan_summary`, a mechanical join of each robot's
`plan.specification`. With one cargo in the activity that made every team's
broadcast identical -- measured on the 2026-09-13 LH run, all three teams sent
`drone_N: ontop(notebook.n.01_1, cabinet.n.01_1); jackal_N:
holding(notebook.n.01_1); ...` -- so a team behind could not tell who actually
held the notebook or whose leg it was, and the three pairs took the cargo off
each other. The head of the chain never receives anything at all (0 of 11
planning calls had a sender in section 7), which is the topology, not a bug.

Here: the plan call asks for `broadcast`, the model's sentence is what goes out,
and the old summary survives only as the fallback for a response without one.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/zixuanwe/Desktop/BEHAVIOR-1K")

from coop2.cognitive.agent.llm_client import (
    InterruptDecision, LLMChainInterruptResponse, LLMChainPlanResponse, NavigateToAction,
    Task, TeamAgentInterruptDecision, TeamAgentPlan,
)
from coop2.comm_topology.llm_team import ChainTeamBrain, create_llm_team_topology

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
SENTENCE = ("team_1 has the notebook: ridgeback_1 carries it to C2 bookcase.n.01_1 this "
            "round with jackal_1. Leave notebook.n.01_1 alone; take a later leg.")
#: The other call that speaks down the chain. A different sentence, so a test
#: can tell which of the two a relay came out of.
RELAY = ("Nothing changed for us: ridgeback_1 still holds notebook.n.01_1 and is going "
         "to bookcase.n.01_1. Carry on with your own leg.")


class StubClient:
    model = "stub"

    def __init__(self, broadcast=SENTENCE, relay=RELAY):
        self.calls, self.formats, self.broadcast = 0, [], broadcast
        self.relay = relay

    @staticmethod
    def _members_in(messages):
        text = messages[-1]["content"]
        return [line.split()[2] for line in text.split("\n") if line.startswith("=== ROBOT ")]

    def generate(self, messages, response_format=None, temperature=0.7):
        self.calls += 1
        self.formats.append(response_format)
        if "decisions" in getattr(response_format, "model_fields", {}):
            # The interrupt round: the decisions and the relay in one call.
            return response_format(
                decisions=[
                    TeamAgentInterruptDecision(agent_id=name,
                                               decision=InterruptDecision.RESUME,
                                               reasoning="nothing changed", new_plan=None)
                    for name in self._members_in(messages)
                ],
                reasoning="nothing changed", broadcast=self.relay,
            ), dict(USAGE)
        plans = [
            TeamAgentPlan(
                agent_id=name,
                task="ontop(notebook.n.01_1, bookcase.n.01_1)",
                actions=[NavigateToAction(target="notebook.n.01_1")],
                reasoning=f"{name} goes",
            )
            for name in self._members_in(messages)
        ]
        return LLMChainPlanResponse(plans=plans, reasoning="split",
                                    broadcast=self.broadcast), dict(USAGE)

    def generate_team_plan(self, messages, temperature=0.7):
        raise AssertionError("the chain must ask for its own response format")

    def generate_team_interrupt_decision(self, messages, temperature=0.7):
        raise AssertionError("the chain's interrupt must ask for its own format too")


def make_run(broadcast=SENTENCE, relay=RELAY):
    teams = {"team_1": ["ridgeback_1", "jackal_1"], "team_2": ["ridgeback_2", "jackal_2"]}
    client = StubClient(broadcast, relay)
    agents = create_llm_team_topology(llm_client=client, teams=teams,
                                      topology="broadcast_chain", verbose=False)
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    brains = {name: agents[members[0]].brain for name, members in teams.items()}
    return client, agents, brains


def main() -> int:
    def ok(message):
        print(f"  PASS {message}")

    print("test 1: the chain asks for its own schema, which carries `broadcast`")
    assert "broadcast" in LLMChainPlanResponse.model_fields
    client, agents, brains = make_run()
    b1 = brains["team_1"]
    assert isinstance(b1, ChainTeamBrain)
    user = b1._build_team_prompt([agents["ridgeback_1"], agents["jackal_1"]])[1]["content"]
    assert "`broadcast`" in user.rsplit("\n\n", 1)[1], "the closing must ask for it"
    ok("schema has `broadcast`; the closing instruction asks the model to write it")

    print("\ntest 2: the model's sentence is what the teams behind receive")
    sent = []
    b1._say = lambda content, kind, interrupts=False, recipients=None: sent.append((content, kind))
    plans = b1._generate_team_plans()
    assert set(plans) == {"ridgeback_1", "jackal_1"}
    assert client.formats == [LLMChainPlanResponse], client.formats
    b1._pending_plans = plans
    b1.after_plan()
    assert sent and sent[0][0] == SENTENCE, sent
    assert sent[0][1] == "broadcast_chain", sent
    assert "ontop(" not in sent[0][0], "a specification dump is what this replaced"
    ok("one call produced the plans and the sentence; the sentence is what went out")

    print("\ntest 3: a response without one falls back to the old summary, never to silence")
    client2, agents2, brains2 = make_run(broadcast="   ")
    b2 = brains2["team_1"]
    sent2 = []
    b2._say = lambda content, kind, interrupts=False, recipients=None: sent2.append(content)
    b2._pending_plans = b2._generate_team_plans()
    b2.after_plan()
    assert sent2 and sent2[0].startswith("[team_1] "), sent2
    assert "ontop(" in sent2[0], "the fallback is the specification summary"
    ok("blank broadcast -> _plan_summary, so a round always says something")

    print("\ntest 4: the sentence is not carried into the next round")
    sent3 = []
    b1._say = lambda content, kind, interrupts=False, recipients=None: sent3.append(content)
    b1._pending_plans = {"ridgeback_1": plans["ridgeback_1"]}
    b1.after_plan()
    assert sent3 and sent3[0].startswith("[team_1] "), "a stale sentence must not be resent"
    ok("_pending_broadcast is cleared after sending")

    print("\ntest 5: the relay after an interrupt is written in that same call")
    # The other half of the change. A chain team relays once per interrupt
    # round whatever it decided, resume included, or the team behind waits out
    # its backstop -- and what it relayed was `_current_allocation()`, which has
    # the fault the plan summary had: one cargo, so every team's read the same.
    assert "broadcast" in LLMChainInterruptResponse.model_fields
    client5, agents5, brains5 = make_run()
    b5 = brains5["team_1"]
    assert "`broadcast`" in b5._interrupt_closing()
    sent5 = []
    b5._say = lambda content, kind, interrupts=False, recipients=None: sent5.append((content, kind))
    message = [{"sender": "team_0", "content": "we took the first leg",
                "metadata": {"type": "broadcast_chain"}}]
    decisions = b5._decide_interrupts(message)
    assert set(decisions) == {"ridgeback_1", "jackal_1"}, decisions
    assert client5.formats == [LLMChainInterruptResponse], client5.formats
    b5.after_interrupt(decisions)
    assert sent5 and sent5[0] == (RELAY, "broadcast_chain"), sent5
    ok("one call decided and wrote the relay; the relay is what went out")

    print("\ntest 6: an interrupt response without a sentence still relays")
    client6, agents6, brains6 = make_run(relay="")
    b6 = brains6["team_1"]
    sent6 = []
    b6._say = lambda content, kind, interrupts=False, recipients=None: sent6.append(content)
    b6.after_interrupt(b6._decide_interrupts(message))
    assert sent6 and sent6[0].startswith("[team_1] "), sent6
    ok("blank relay -> _current_allocation, so the team behind is never left waiting")

    print("\ntest 7: section 4 names this team's own upstream and downstream")
    # Section 4 is one text per mode, so it could only say "the teams ahead of
    # you" -- and a team reading a broadcast had to infer from the robot ids
    # whether the sender was ahead of it or behind, which is what it cannot
    # see (user, 2026-09-14).
    chain = {"team_1": ["r1"], "team_2": ["r2"], "team_3": ["r3"]}
    ring = create_llm_team_topology(llm_client=StubClient(), teams=chain,
                                    topology="broadcast_chain", verbose=False)
    section4 = {}
    for robot, team in (("r1", "team_1"), ("r2", "team_2"), ("r3", "team_3")):
        prompt = ring[robot].brain._system_prompt(ring[robot])
        section4[team] = [s for s in prompt.split("\n\n") if s.startswith("## 4.")][0]

    head = section4["team_1"]
    assert "UPSTREAM: none" in head and "head of the chain" in head, head
    assert "team_2, team_3" in head, "the head broadcasts to everyone after it"

    middle = section4["team_2"]
    assert "UPSTREAM (plan before you): team_1." in middle, middle
    assert "DOWNSTREAM (plan after you, on what you say): team_3." in middle, middle
    assert "team_2" not in middle.split("DOWNSTREAM")[1], "a team is not its own downstream"

    tail = section4["team_3"]
    # Every earlier team's broadcast reaches the tail and lands in section 7,
    # so both are named -- `wait_for` alone (team_2) would have it plan around
    # one commitment while reading two.
    assert "UPSTREAM (plan before you): team_1, team_2." in tail, tail
    assert "DOWNSTREAM: none" in tail and "you are the tail" in tail, tail
    ok("head, middle and tail each read their own neighbours by name")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
