"""CPU-only: a team's prompt carries what it said as well as what it heard.

Before this, `_heard` -- what arrived since the team last planned -- was the
whole of the team's message record, and it was cleared every round. So the
model saw the latest turn and nothing else: not the question it had asked,
not the allocation it had announced, not the answer it received two rounds
ago. It re-asked and re-announced, with the earlier exchange nowhere in front
of it.

The storage is the `AgentMemory` every robot already keeps, reused at team
level and holding only messages. What this pins:

  * a message the team sends is in its record, as "You told ...";
  * a message it receives is in its record once, not once per member;
  * the record survives the planning round that clears `_heard`;
  * what arrived since the last plan is marked (new), and only that;
  * the interrupt prompt quotes the record minus the messages it is deciding on;
  * the record is bounded.

Run:
    python feasibility_verify/test_team_message_memory_stubbed.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_llm_team_stubbed import StubClient  # noqa: E402  -- the team harness

from coop2.cognitive.messages import MessageBroker  # noqa: E402
from coop2.comm_topology.llm_team import (  # noqa: E402
    TEAM_MESSAGE_HISTORY, create_llm_team_topology,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


def centralized_pair():
    client = StubClient()
    teams = {"lead": ["agent_0", "agent_1"], "follow": ["agent_2", "agent_3"]}
    agents = create_llm_team_topology(
        llm_client=client, teams=teams, topology="centralized", verbose=False,
        goal_instruction="put the apples on the table",
    )
    broker = MessageBroker(agents)
    for agent in agents.values():
        agent.message_broker = broker
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    return client, agents, broker


def main() -> int:
    print("test 1: what the leader assigned is in the leader's own record")
    client, agents, broker = centralized_pair()
    lead, follow = agents["agent_0"].brain, agents["agent_2"].brain
    for name in ("agent_0", "agent_1"):
        agents[name].handle_reasoning()
    asks = [m for m in broker.get_message_log()
            if (m.get("metadata") or {}).get("type") == "leader_broadcast"]
    assert len(asks) == 1, asks
    block = lead._messages_block()
    assert "You told follow" in block, block
    # The leader's first word is the assignment, written by the (stub) model.
    assert "take C1 with your drone" in block, block
    assert "(new)" not in block, "the team's own words are never news to it"
    ok("sent once, recorded once, as the team's own words")

    print("test 2: the follower reads the request once, marked new, then answers")
    follow._collect_heard()
    block = follow._messages_block()
    assert block.count("From lead") == 1, block
    assert "From lead (new)" in block, block
    for name in ("agent_2", "agent_3"):
        agents[name].handle_reasoning()
    replies = [m for m in broker.get_message_log()
               if (m.get("metadata") or {}).get("type") == "follower_response"]
    assert len(replies) == 1, replies
    block = follow._messages_block()
    assert "From lead" in block, "the request must outlive the round that read it"
    assert "You told lead" in block, block
    assert block.index("From lead") < block.index("You told lead"), "oldest first"
    assert "(new)" not in block, "after planning, nothing is new until something arrives"
    ok("request and reply both in the record, in order, and the round did not erase them")

    print("test 3: the record reached the prompt the follower planned from")
    prompt = client.last_prompt
    # Since 2026-09-13 the news is section 7 and the record section 8, so the
    # request it was answering is quoted under MESSAGES RECEIVED NOW and not
    # repeated in the history.
    assert "## 7. MESSAGES RECEIVED NOW" in prompt, prompt[-600:]
    news = prompt.split("## 7.", 1)[1].split("## 8.", 1)[0] if "## 8." in prompt else prompt.split("## 7.", 1)[1]
    assert "From lead" in news, "the request it was answering was news at that moment"
    ok("the planning prompt quotes the exchange, the news in section 7")

    print("test 4: the leader hears the answer, and keeps its own question next to it")
    lead._collect_heard()
    block = lead._messages_block()
    assert "You told follow" in block and "From follow (new)" in block, block
    assert block.index("You told follow") < block.index("From follow"), block
    ok("question then answer, both directions, one record")

    print("test 5: the interrupt prompt does not quote a message twice")
    members = [lead.members["agent_0"], lead.members["agent_1"]]
    incoming = [dict(replies[0], content=replies[0]["content"])]
    user = lead._build_interrupt_prompt(members, incoming)[1]["content"]
    assert "## 7. MESSAGES RECEIVED NOW" in user
    body = replies[0]["content"]
    assert user.count(body) == 1, f"quoted {user.count(body)} times:\n{user}"
    assert "## 8. CONVERSATION HISTORY" in user, "the record minus the news is still there"
    assert "You told follow" in user
    ok("the news under section 7, the rest under section 8, each once")

    print("test 6: the record is bounded, and keeps the newest")
    for i in range(TEAM_MESSAGE_HISTORY * 3):
        follow._remember({"sender": "lead", "timestamp": 10.0 + i, "env_step": 100 + i,
                          "content": f"note {i}"})
    block = follow._messages_block()
    quoted = [line for line in block.splitlines() if line.startswith("  [step")]
    assert len(quoted) == TEAM_MESSAGE_HISTORY, len(quoted)
    assert f"note {TEAM_MESSAGE_HISTORY * 3 - 1}" in block and "note 0" not in block, block
    ok(f"{TEAM_MESSAGE_HISTORY} lines quoted, newest kept")

    print("test 7: a follower at its barrier waits while the leader is about to assign")
    import threading, time as _time
    from coop2.cognitive.agent.agent import AgentState
    client, agents, broker = centralized_pair()
    lead, follow = agents["agent_0"].brain, agents["agent_2"].brain
    # The leader's robots are all in R (nobody executing) and it has not sent
    # this round's assignment: it is about to, so the follower waits for it.
    assert not any(agents[n].state is AgentState.X for n in ("agent_0", "agent_1"))
    lead._assignment_sent = False
    def speak_later():
        _time.sleep(0.3)
        lead._say("[lead] follow: take C1", "leader_broadcast", interrupts=True)
        lead._assignment_sent = True
    threading.Thread(target=speak_later, daemon=True).start()
    t0 = _time.monotonic()
    follow.before_plan()
    waited = _time.monotonic() - t0
    assert 0.25 <= waited < 5.0, waited
    replies = [m for m in broker.get_message_log() if (m.get("metadata") or {}).get("type") == "follower_response"]
    assert len(replies) == 1, "the follower answered the assignment it waited for"
    # A leader with a robot still executing is mid-round: not waited for.
    client, agents, broker = centralized_pair()
    lead2 = agents["agent_0"].brain; lead2._assignment_sent = False
    agents["agent_0"]._set_state(AgentState.X)
    t0 = _time.monotonic(); agents["agent_2"].brain.before_plan()
    assert _time.monotonic() - t0 < 0.2, "leader executing -> no wait"
    # An assignment already sent this round: no wait either.
    agents["agent_0"]._set_state(AgentState.R); lead2._assignment_sent = True
    t0 = _time.monotonic(); agents["agent_2"].brain.before_plan()
    assert _time.monotonic() - t0 < 0.2, "assignment already out -> no wait"
    ok("waits while the leader is about to assign, answers when it lands, does not wait otherwise")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
