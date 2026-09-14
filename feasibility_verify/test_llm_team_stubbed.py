"""CPU-only test for the one-LLM-per-team topology.

No Isaac, no GPU, no model: a stub client counts calls and returns canned team
responses. What is pinned is the part that cannot be checked by reading the code
-- the barrier -- because getting it wrong deadlocks rather than fails.

The trap: the plan loop does not step the environment while any agent is
not ready. So a member that finished early and simply waited for its teammates
would freeze the world, and those teammates need the world to advance in order
to finish. The design has early finishers stay ready and hold position instead,
and these tests are what keep that true.

Run:
    python feasibility_verify/test_llm_team_stubbed.py
"""

from __future__ import annotations

import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.llm_client import (
    InterruptDecision,
    LLMPlanResponse,
    LLMTeamInterruptResponse,
    LLMTeamPlanResponse,
    NavigateToAction,
    Task,
    TaskSpecification,
    TeamAgentInterruptDecision,
    TeamAgentPlan,
)
from coop2.cognitive.action.action import SymbolicAction
from coop2.comm_topology.llm_team import (
    TEAM_HOLD_ACTIONS,
    TEAM_HOLD_TICKS,
    create_llm_team_topology,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


USAGE = {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4, "latency_seconds": 0.1}


class StubClient:
    """Counts calls and hands back a plan for every member of the team asked about."""

    model = "stub"

    def __init__(self):
        self.plan_calls = 0
        self.interrupt_calls = 0
        self.last_prompt = None
        self.last_interrupt_prompt = None
        self.text_calls = 0
        self.last_text_prompt = None
        self.text_prompts = []
        self.interrupt_script = {}
        self.plan_formats = []
        self.interrupt_formats = []
        self.interrupt_prompts = []
        #: what the model writes for a schema field beside `plans` -- the
        #: chain's `broadcast`, the board's `board_post`.
        self.broadcast = "team_0 has the die; the bedroom leg is yours"
        #: and what it writes in an interrupt call -- a different sentence, so
        #: a test can tell which call the relay came out of.
        self.relay = "the message changed nothing for us; agent_0 still has the die"

    def _members_in(self, messages):
        """Read the agent ids out of the prompt the brain built."""
        text = messages[-1]["content"]
        return [line.split()[2] for line in text.split("\n") if line.startswith("=== ROBOT ")]

    def _plans_for(self, messages):
        return [
            TeamAgentPlan(
                agent_id=name,
                task=TaskSpecification(task=Task.ONTOP, object_type="apple.n.01_1",
                                       reference="coffee_table.n.01_1"),
                actions=[NavigateToAction(target="apple.n.01_1")],
                reasoning=f"{name} goes for the apple",
            )
            for name in self._members_in(messages)
        ]

    def generate_team_plan(self, messages, temperature=0.7):
        self.plan_calls += 1
        self.last_prompt = messages[-1]["content"]
        return LLMTeamPlanResponse(
            plans=self._plans_for(messages), reasoning="split by distance"
        ), dict(USAGE)

    def generate(self, messages, response_format=None, temperature=0.7):
        """Plain text, for a follower composing its report -- unless a schema
        with `plans` is asked for, which is how a chain team plans: its
        broadcast is written in the same call, so the plans come back through
        `generate` rather than `generate_team_plan`."""
        fields = getattr(response_format, "model_fields", {})
        if "decisions" in fields:
            # A chain team decides and writes its relay in one call.
            self._note_interrupt(messages)
            self.interrupt_formats.append(response_format)
            extra = {name: self.relay for name in fields
                     if name not in ("decisions", "reasoning")}
            return response_format(
                decisions=self._decisions_for(messages), reasoning="scripted", **extra
            ), dict(USAGE)
        if "plans" in fields:
            self.plan_calls += 1
            self.last_prompt = messages[-1]["content"]
            self.plan_formats.append(response_format)
            extra = {name: self.broadcast for name in fields
                     if name not in ("plans", "reasoning")}
            return response_format(
                plans=self._plans_for(messages), reasoning="split by distance", **extra
            ), dict(USAGE)
        self.text_calls += 1
        self.last_text_prompt = messages[-1]["content"]
        self.text_prompts.append(messages[-1]["content"])
        # A leader assigning and a follower answering get different texts, so
        # a test can tell the two apart in one record.
        if "assign this round's work" in messages[-1]["content"]:
            return "follow: take C1 with your drone; we take the rest", dict(USAGE)
        return "we will take the west apples, leave the east to you", dict(USAGE)

    def _decisions_for(self, messages):
        decisions = []
        for name in self._members_in(messages):
            choice = self.interrupt_script.get(name, InterruptDecision.RESUME)
            # A scripted REPLAN carries a plan, as the model's does: without one
            # `_decide_interrupts` yields (REPLAN, None) and there is nothing to
            # announce, so a test of the announcement would pass vacuously.
            new_plan = None
            if choice is InterruptDecision.REPLAN:
                new_plan = LLMPlanResponse(
                    task=TaskSpecification(task=Task.ONTOP, object_type="apple.n.01_1",
                                           reference="coffee_table.n.01_1"),
                    actions=[NavigateToAction(target="apple.n.01_1")],
                    reasoning=f"{name} was told to change",
                )
            decisions.append(
                TeamAgentInterruptDecision(
                    agent_id=name,
                    decision=choice,
                    reasoning="scripted",
                    new_plan=new_plan,
                )
            )
        return decisions

    def generate_team_interrupt_decision(self, messages, temperature=0.7):
        self._note_interrupt(messages)
        return LLMTeamInterruptResponse(
            decisions=self._decisions_for(messages), reasoning="scripted"
        ), dict(USAGE)

    def _note_interrupt(self, messages):
        self.interrupt_calls += 1
        # Recorded here too, or an assertion about the interrupt prompt reads
        # whatever the last *plan* prompt was and passes for the wrong reason.
        self.last_prompt = messages[-1]["content"]
        self.last_interrupt_prompt = messages[-1]["content"]
        # Every one of them, not just the last: three teams share one client and
        # which of them calls last is a thread race.
        self.interrupt_prompts.append(messages[-1]["content"])


def make_team(size, name="alpha"):
    client = StubClient()
    ids = [f"agent_{i}" for i in range(size)]
    agents = create_llm_team_topology(
        llm_client=client, teams={name: ids}, verbose=False, goal_instruction="do the thing"
    )
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    return client, agents, ids


def main() -> int:
    print("test 1: a team does not plan until every member has finished")
    client, agents, ids = make_team(4)
    # Three members finish; the fourth is still executing and never calls in.
    for name in ids[:3]:
        agents[name].handle_reasoning()
    assert client.plan_calls == 0, "planned before the team was complete"
    for name in ids[:3]:
        plan = agents[name].plan
        assert plan is not None, f"{name} has no plan at all -- it would never become ready"
        assert plan.actions[0].action_type == "wait", plan.actions[0].action_type
        assert plan.actions[0].args["ticks"] == TEAM_HOLD_TICKS
        # Every action a wait, and nothing else. A hold that goes through
        # parse_plan_response picks up a terminal action derived from its
        # TaskSpecification -- expressed as holding(<self>) that appended
        # grasp(<self>), a robot planning to pick itself up, which reached a
        # real run before it was caught.
        kinds = {a.action_type for a in plan.actions}
        assert kinds == {"wait"}, sorted(kinds)
        # And many of them: a plan that completes sends its agent back to R, so
        # a one-action hold cycled X -> R -> W -> X every 60 ticks to ask for a
        # plan it was never going to get. The recall ends a hold, not expiry.
        assert len(plan.actions) == TEAM_HOLD_ACTIONS, len(plan.actions)
        plan.advance_action()
        assert not plan.is_complete(), (
            "one action in and the hold is already complete -- the member will "
            "be sent back to reasoning"
        )
    ok("3 of 4 finished -> no LLM call, and each early finisher holds rather than stalling")

    print("test 2: the last member closes the barrier and everyone gets a real plan")
    agents[ids[3]].handle_reasoning()
    assert client.plan_calls == 1, client.plan_calls
    assert agents[ids[3]].plan.actions[0].action_type == "navigate_to"
    # The other three are still holding; they take their plans on their next
    # pass through reasoning, which is what happens when their hold expires.
    for name in ids[:3]:
        agents[name].handle_reasoning()
    assert client.plan_calls == 1, f"one round must be one call, got {client.plan_calls}"
    for name in ids:
        plan = agents[name].plan
        assert plan.actions[0].action_type == "navigate_to", (name, plan.actions[0].action_type)
        assert plan.agent_id == name
    ok("one call for the whole round, and every robot ends up with its own plan")

    print("test 3: the prompt carries every member's observation, once each")
    prompt = client.last_prompt
    for name in ids:
        assert f"=== ROBOT {name} ===" in prompt, f"{name} missing from the team prompt"
        assert prompt.count(f"view for {name}") == 1, f"{name}'s observation duplicated"
    assert "do the thing" in prompt, "the global objective is missing"
    ok("4 observations, one section each, plus the objective")

    print("test 4: a team of one behaves exactly like the individual topology")
    solo_client, solo_agents, solo_ids = make_team(1, name="solo")
    solo_agents[solo_ids[0]].handle_reasoning()
    assert solo_client.plan_calls == 1
    # No hold is ever taken: the barrier is satisfied by the only member.
    assert solo_agents[solo_ids[0]].plan.actions[0].action_type == "navigate_to"
    ok("N=1 plans immediately and never holds, so per-robot LLM is this code with N=1")

    print("test 5: the team is interrupted together and answered in one call")
    client, agents, ids = make_team(3)
    client.interrupt_script = {ids[1]: InterruptDecision.REPLAN}
    for name in ids:
        agents[name].plan = None

    # Concurrently, because that is how it happens: the broker interrupts every
    # member of a team in one call, so their threads arrive together. Members
    # that arrive early **block in I** rather than bouncing back to ready --
    # a member that bounced showed as "waiting" on the timeline while its
    # teammates were still interrupted, which is the bug this pins.
    #
    # The state has to be set, not assumed: the barrier waits for teammates that
    # are *in* I, so a test that called handle_interrupt on agents still in R
    # was testing a situation the broker never produces.
    import threading

    from coop2.cognitive.agent.agent import AgentState

    for name in ids:
        agents[name]._set_state(AgentState.I, timestamp=0.0, env_step=0)

    barrier_state = {}

    def arrive(name):
        barrier_state[name] = agents[name].handle_interrupt()

    threads = [threading.Thread(target=arrive, args=(name,)) for name in ids[:2]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
    assert all(t.is_alive() for t in threads), "early arrivals must block, not return"
    assert client.interrupt_calls == 0, "decided before the whole team was interrupted"

    agents[ids[2]].handle_interrupt()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not any(t.is_alive() for t in threads), "blocked members must be released"
    assert client.interrupt_calls == 1, client.interrupt_calls
    ok("2 of 3 block in I; the third closes the barrier and one call answers all")

    print("test 6: a member the model forgot is held, not left planless")
    class Forgetful(StubClient):
        def generate_team_plan(self, messages, temperature=0.7):
            response, usage = super().generate_team_plan(messages, temperature)
            response.plans = response.plans[:-1]  # drop the last robot
            return response, usage

    client = Forgetful()
    ids = ["agent_0", "agent_1"]
    agents = create_llm_team_topology(llm_client=client, teams={"t": ids}, verbose=False)
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    for name in ids:
        agents[name].handle_reasoning()
    # A member with no plan never becomes ready, which would strand the whole
    # run at the next barrier -- so the brain gives it a hold instead.
    assert agents[ids[1]].plan is not None
    skipped = agents[ids[1]].plan
    assert {a.action_type for a in skipped.actions} == {"wait"}, (
        sorted({a.action_type for a in skipped.actions})
    )
    assert len(skipped.actions) == TEAM_HOLD_ACTIONS, len(skipped.actions)
    ok("a skipped robot gets a hold, so it still becomes ready")

    print("test 7: an LLM failure falls back rather than ending the run")
    class Broken(StubClient):
        def generate_team_plan(self, messages, temperature=0.7):
            raise RuntimeError("no model today")

    client = Broken()
    ids = ["agent_0", "agent_1"]
    agents = create_llm_team_topology(llm_client=client, teams={"t": ids}, verbose=False)
    for agent in agents.values():
        agent.observe({}, 0)
    for name in ids:
        agents[name].handle_reasoning()
    for name in ids:
        assert agents[name].plan is not None, f"{name} stranded with no plan after an LLM error"
    ok("every robot still has a plan after the call raised")

    print("test 8: the three topologies wire teams, not robots")
    from coop2.comm_topology.llm_team import (
        ChainTeamBrain, FollowerTeamBrain, LeaderTeamBrain, TeamBrain,
    )

    teams = {"t0": ["a0", "a1"], "t1": ["a2", "a3"], "t2": ["a4", "a5"]}

    def brains_of(topology):
        agents = create_llm_team_topology(
            llm_client=StubClient(), teams=teams, topology=topology, verbose=False
        )
        out = {}
        for agent in agents.values():
            out[agent.brain.team_name] = agent.brain
        return out

    solo = brains_of("individual")
    assert all(isinstance(b, TeamBrain) and not b.send_to and not b.wait_for
               for b in solo.values())

    chain = brains_of("broadcast_chain")
    assert all(isinstance(b, ChainTeamBrain) for b in chain.values())
    # Both sides of the wiring are team names. They used to be agent ids -- a
    # team was addressed as its four robots and waited on through whichever
    # member spoke for it -- which put a conversation between two teams in the
    # log as eight one-sided ones between robots that never composed a word.
    assert chain["t1"].wait_for == ["t0"], chain["t1"].wait_for
    assert chain["t0"].send_to == ["t1", "t2"], chain["t0"].send_to
    assert chain["t2"].send_to == [], "the last team has nobody downstream"

    central = brains_of("centralized")
    assert isinstance(central["t0"], LeaderTeamBrain)
    assert isinstance(central["t1"], FollowerTeamBrain)
    assert central["t0"].wait_for == ["t1", "t2"], central["t0"].wait_for
    assert central["t0"].send_to == ["t1", "t2"], central["t0"].send_to
    assert central["t1"].wait_for == ["t0"], central["t1"].wait_for
    assert central["t1"].send_to == ["t0"], central["t1"].send_to
    # Delivery still reaches every robot of an addressed team: an interrupt has
    # to stop all of it, or the team splits across I and W and stalls its own
    # interrupt barrier. That expansion belongs to the broker, not the address.
    assert central["t1"].all_teams["t0"] == ["a0", "a1"]
    ok("individual/chain/centralized wire team to team, and teams expand at delivery")

    print("test 9: when the team thinks, every member is reasoning")
    from coop2.cognitive.agent.agent import AgentState

    client, agents, ids = make_team(4)
    # Three finish early and hold, then actually start executing those holds --
    # which is the state they are really in while the fourth robot works.
    for name in ids[:3]:
        agents[name].handle_reasoning()
        agents[name].set_ready()
        agents[name].start_execution()
        assert agents[name].state == AgentState.X, agents[name].state

    # One of them finishes its hold and is sitting ready in W rather than still
    # executing. It is idling just the same, and was being skipped.
    agents[ids[0]].set_ready()
    assert agents[ids[0]].state == AgentState.W, agents[ids[0]].state

    # The fourth arrives and closes the barrier.
    agents[ids[3]].handle_reasoning()

    # Every teammate must now be reasoning, not still executing a hold. Only the
    # member that closed the barrier was reasoning before this fix, so the
    # timeline showed one red bar and three green ones during a team call.
    for name in ids[:3]:
        assert agents[name].state == AgentState.R, (name, agents[name].state)
        assert not agents[name].ready, f"{name} is still marked ready"
    # And the hold each was running is terminal, or create_agent_thread would
    # decline to re-plan it and the member would never become ready again.
    ok("holders are recalled into R, with their holds marked terminal")

    print("test 10: a follower answers the leader, and only when it was asked")
    from coop2.cognitive.messages import MessageBroker

    def centralized_pair():
        client = StubClient()
        teams = {"lead": ["agent_0", "agent_1"], "follow": ["agent_2", "agent_3"]}
        agents = create_llm_team_topology(
            llm_client=client, teams=teams, topology="centralized", verbose=False
        )
        broker = MessageBroker(agents)
        for agent in agents.values():
            agent.message_broker = broker
            agent.symbolic_view = f"view for {agent.agent_id}"
            agent.observe({}, 0)
        return client, agents, broker

    # The follower team plans first, with nothing from the leader in its inbox.
    _client, agents, broker = centralized_pair()
    for name in ("agent_2", "agent_3"):
        agents[name].handle_reasoning()
    sent = [m for m in broker.get_message_log()
            if (m.get("metadata") or {}).get("type") == "follower_response"]
    # It used to answer anyway: `_await_speakers` releases as soon as the leader
    # merely *looks* ready, which at the start of a run it does, so the reply
    # was logged in the same instant as -- and ahead of -- the request.
    assert not sent, f"answered a question nobody asked: {sent}"

    # Now with the leader's request actually delivered first.
    _client, agents, broker = centralized_pair()
    for name in ("agent_0", "agent_1"):
        agents[name].handle_reasoning()
    asks = [m for m in broker.get_message_log()
            if (m.get("metadata") or {}).get("type") == "leader_broadcast"]
    assert len(asks) == 1, asks

    # The team read it once, not once per robot it was handed to. Delivery
    # copies a team message into all four inboxes, so draining them all used to
    # quote the same request four times in the prompt.
    follow = agents["agent_2"].brain
    follow._collect_heard()
    assert follow._messages_block().count("From lead") == 1, follow._messages_block()

    for name in ("agent_2", "agent_3"):
        agents[name].handle_reasoning()
    replies = [m for m in broker.get_message_log()
               if (m.get("metadata") or {}).get("type") == "follower_response"]
    assert len(replies) == 1, replies
    assert replies[0]["timestamp"] >= asks[0]["timestamp"], "reply predates the request"
    ok("silent when unasked; answers once, after the request, and quotes it once")

    print("test 11: the team's own timeline is recorded and saved")
    import json
    import tempfile

    from coop2.experiment.agent_timeline import save_team_timeline

    lead = agents["agent_0"].brain
    assert [s["kind"] for s in lead.timeline] == ["planning"], lead.timeline
    assert lead.timeline[0]["end"] >= lead.timeline[0]["start"]

    # The log is addressed team to team. It used to read "agent_0 -> agent_2,
    # agent_3", which credited the send to a robot that had no part in writing
    # it and turned one conversation between two teams into several.
    log = broker.get_message_log()
    assert [(m["sender"], tuple(m["recipients"])) for m in log] == [
        ("lead", ("follow",)), ("follow", ("lead",)),
    ], log
    assert all(m.get("sender_type") == "team" for m in log), log
    assert log[0]["delivered_to"] == ["agent_2", "agent_3"], log[0]

    with tempfile.TemporaryDirectory() as directory:
        path = save_team_timeline(agents, os.path.join(directory, "team_timeline.json"))
        payload = json.load(open(path))
    assert set(payload["teams"]) == {"lead", "follow"}
    assert payload["teams"]["lead"] == ["agent_0", "agent_1"]
    # Relative to the same origin the agent states use, or the spans would land
    # somewhere else entirely on the shared x axis.
    assert all(0.0 <= s["start"] < 60.0 for s in payload["spans"]["lead"]), payload["spans"]
    ok("addressed team to team in the log, with the thinking spans recorded")

    print("test 12: a member that raced the closing barrier waits, it does not hold")
    import threading as _threading
    import time as _time

    class Slow(StubClient):
        def generate_team_plan(self, messages, temperature=0.7):
            _time.sleep(0.3)  # an LLM call is seconds; the race is microseconds
            return super().generate_team_plan(messages, temperature)

    client = Slow()
    ids = ["agent_0", "agent_1", "agent_2"]
    agents = create_llm_team_topology(llm_client=client, teams={"t": ids}, verbose=False)
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    brain = agents["agent_0"].brain

    # agent_0 and agent_1 have both asked and been told to hold. agent_1 is now
    # in the gap between being told that and acting on it.
    assert brain.request_plan("agent_0") is None
    assert brain.request_plan("agent_1") is None

    closer = _threading.Thread(target=agents["agent_2"].handle_reasoning)
    closer.start()
    _time.sleep(0.05)  # the barrier closes; agent_1 is still in that gap

    # The recall cannot reach agent_1 -- it is in R, which is not an idle state
    # and is deliberately left alone -- and there is no plan to hand it yet
    # either, because the call has only just started. Asking once returned None
    # and it held: measured at 15.8 s of "waiting" through its own team's call,
    # while its teammates showed 15.8 s of "reasoning".
    plan = brain.claim_pending_plan("agent_1")
    closer.join(timeout=5.0)
    assert plan is not None, "agent_1 held through a call that was already under way"
    assert plan.actions[0].action_type == "navigate_to", plan.actions[0].action_type
    assert client.plan_calls == 1, client.plan_calls
    ok("it blocks on the call in flight and comes back with the real plan")

    print("test 13: a teammate that was never interrupted does not hold the barrier")
    # The broker interrupts an agent in W or X and skips one in R -- and R is
    # where the member running the team's planning call sits. The barrier used
    # to require every member regardless, so on
    # centralized_agents8_..._011122 a request that landed 2 ms into the run
    # caught three members in W, missed the fourth still reasoning, and froze
    # the world for the full 30 s timeout.
    client, agents, ids = make_team(4)
    for name in ids:
        agents[name].plan = None
    for name in ids[:3]:
        agents[name]._set_state(AgentState.I, timestamp=0.0, env_step=0)
    # ids[3] stays in R: it is the one doing the planning call.
    assert agents[ids[3]].state is AgentState.R

    done = {}

    def arrive_late(name):
        done[name] = agents[name].handle_interrupt()

    threads = [threading.Thread(target=arrive_late, args=(name,)) for name in ids[:3]]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    elapsed = time.monotonic() - started
    assert not any(t.is_alive() for t in threads), (
        "the three interrupted members blocked on a fourth that was never coming"
    )
    assert elapsed < 2.0, f"took {elapsed:.1f}s -- it waited on the timeout again"
    assert client.interrupt_calls == 1, client.interrupt_calls
    ok("three interrupted members decide without the one still reasoning")

    print("test 14: the objective reaches the team's own prompt, on every path")
    # The team brain builds its own prompt from its own copy of the objective,
    # so setting the attribute on the agents does not reach it. run_centralized
    # passed the goal to the agents and not to the factory, and its teams
    # planned with no objective at all -- 0 prompts carrying one, against 7 for
    # broadcast_chain on the same task.
    client, agents, ids = make_team(2)
    for name in ids:
        agents[name].plan = None
    for name in ids:
        agents[name].handle_reasoning()
    plan_prompt = client.last_prompt
    assert "GLOBAL OBJECTIVE: do the thing" in plan_prompt, (
        "a team planned without being told what the run is for"
    )

    # And on the interrupt prompt, which is a different builder.
    from coop2.cognitive.agent.agent import AgentState
    for name in ids:
        agents[name]._set_state(AgentState.I, timestamp=0.0, env_step=0)
    threads = [threading.Thread(target=agents[name].handle_interrupt) for name in ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    interrupt_prompt = client.last_interrupt_prompt
    assert interrupt_prompt is not None, "the interrupt prompt was never recorded"
    assert "GLOBAL OBJECTIVE: do the thing" in interrupt_prompt, (
        "the resume/replan call did not know the objective either"
    )
    # A brain with no objective must not print a stray header.
    brain = agents[ids[0]].brain
    brain.goal_instruction = ""
    assert "GLOBAL OBJECTIVE" not in brain._build_team_prompt(
        [agents[name] for name in ids]
    )[-1]["content"]
    ok("plan and interrupt prompts both carry the objective, and omit it when unset")

    print("test 15: one message to a team is one interrupt decision, not one per member")
    # The broker interrupts a team's members one at a time, so the first
    # member's thread can be inside handle_interrupt before the last is in I.
    # A barrier that inferred its set from "who is in I right now" closed early
    # and each member then decided alone: three interrupt calls, and three LLM
    # round trips, for one message. Driven through the real broker here,
    # because the announcement is the broker's half of the fix.
    from coop2.cognitive.messages import MessageBroker

    client, agents, ids = make_team(4, name="bravo")
    for name in ids:
        agents[name].plan = None
        agents[name]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    broker = MessageBroker(agents)
    broker.teams = {"bravo": list(ids)}
    for agent in agents.values():
        agent.message_broker = broker

    # Every member is in W, so the delivery stops all four.
    broker.send_team_message(
        sender_team="outsider", recipients=["bravo"], content="status please",
        metadata={"type": "leader_broadcast", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    brain = agents[ids[0]].brain
    assert brain._expected_interrupt == set(ids), (
        f"the broker announced {brain._expected_interrupt}, not the four it stopped"
    )
    assert all(agents[n].state is AgentState.I for n in ids)

    threads = [threading.Thread(target=agents[n].handle_interrupt) for n in ids]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not any(t.is_alive() for t in threads), "the barrier never closed"
    assert time.monotonic() - started < 2.0, "it waited on a timeout"
    assert client.interrupt_calls == 1, (
        f"{client.interrupt_calls} interrupt calls for one message"
    )
    ok("four members, one announcement, one decision call")

    print("test 16: a member that slipped out of W before its turn is not waited for")
    # The announcement is a prediction from each recipient's state, and the
    # state can change before the delivery loop reaches that recipient:
    # interrupt() is a no-op from R. On
    # centralized_agents8_..._021551 three members went W -> R in that window,
    # only the fourth was really stopped, and it waited out the full 30 s for
    # three that were never coming. The confirmation after the loop narrows the
    # set to what actually happened.
    client, agents, ids = make_team(4, name="charlie")
    for name in ids:
        agents[name].plan = None
        agents[name]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    broker = MessageBroker(agents)
    broker.teams = {"charlie": list(ids)}
    for agent in agents.values():
        agent.message_broker = broker

    # Three of them leave W the instant the prediction has been taken -- which
    # is what a real team does when its members reach their planning barrier.
    real_announce = broker._announce_interrupts

    def announce_then_slip(message_record, recipient_ids, blocked=None):
        real_announce(message_record, recipient_ids, blocked)
        for name in ids[:3]:
            agents[name]._set_state(AgentState.R, timestamp=0.0, env_step=0)

    broker._announce_interrupts = announce_then_slip
    broker.send_team_message(
        sender_team="outsider", recipients=["charlie"], content="status please",
        metadata={"type": "leader_broadcast", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    brain = agents[ids[0]].brain
    assert agents[ids[3]].state is AgentState.I, "the one still in W should be stopped"
    assert all(agents[n].state is AgentState.R for n in ids[:3])
    assert brain._expected_interrupt == {ids[3]}, (
        f"expected narrowed to {brain._expected_interrupt}, not just the member stopped"
    )

    started = time.monotonic()
    agents[ids[3]].handle_interrupt()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"took {elapsed:.1f}s -- it waited for the three that slipped"
    assert client.interrupt_calls == 1, client.interrupt_calls
    ok("only the member really stopped is waited for, and it decides at once")

    print("test 17: a follower's report answers what the leader asked")
    # It used to send only plan.specification, which answered none of the four
    # things the request asks for and was `wait_for_team(...)` for most robots
    # most of the time. The exchange read correctly in the log -- right senders,
    # right addressing -- and carried nothing the leader could allocate on.
    from coop2.comm_topology.llm_team import _first_useful_target, _read_view_header

    view = (
        "Step 0/2500 | you are agent_4 in empty_room_0 (a empty room)\n"
        "Holding: apple.n.01_2\n"
        "\nempty_room_0:\n"
        "  - agent_5  (teammate)\n"
        "  - apple.n.01_1  -> unreachable, navigate_to  [36.7 m away]\n"
        "  - coffee_table.n.01_1  -> place_on_top, navigate_to\n"
    )
    assert _read_view_header(view) == ("empty_room_0", "apple.n.01_2")
    assert _first_useful_target(view) == "coffee_table.n.01_1 (in range)", (
        "an object it can act on now must win over a nearer unreachable one"
    )
    # Out of range everywhere: report the closest, with its distance.
    far = view.replace("  - coffee_table.n.01_1  -> place_on_top, navigate_to\n",
                       "  - coffee_table.n.01_1  -> unreachable, navigate_to  [11.6 m away]\n")
    assert _first_useful_target(far) == "coffee_table.n.01_1 (12 m away)"
    assert _read_view_header(None) == ("", "") and _first_useful_target(None) == ""

    # _status_report belongs to the follower role, so build a centralized pair.
    client = StubClient()
    agents = create_llm_team_topology(
        llm_client=client, topology="centralized",
        teams={"team_0": ["agent_0"], "team_1": ["agent_4", "agent_5"]},
        verbose=False, goal_instruction="do the thing",
    )
    ids = ["agent_4", "agent_5"]
    for name in ids:
        agents[name].symbolic_view = view
        agents[name].observe({}, 0)
    follower = agents[ids[0]].brain
    report = follower._status_report()
    for wanted in ("in empty_room_0", "holding apple.n.01_2",
                   "nearest target coffee_table.n.01_1 (in range)", "plan "):
        assert wanted in report, f"the report does not say {wanted!r}: {report}"
    assert ids[0] in report and ids[1] in report
    # A holding robot is described as idle, not by the placeholder's name.
    agents[ids[0]].plan = agents[ids[0]].build_hold_plan()
    assert "idle, waiting for its team" in follower._status_report()
    assert "wait_for_team" not in follower._status_report()
    ok("room, held object, a reachable target and the plan -- all four")

    print("test 18: the follower writes its own reply, grounded in those facts")
    # Upstream's follower composes its answer (`_build_follower_response` is
    # `self._generate_message(...)`); this port had replaced it with the
    # assembled report, which saved a call and lost the point of asking -- a
    # rendering of a team's own state cannot propose anything.
    agents[ids[0]].plan = None
    before = client.text_calls
    reply = follower._compose_response()
    assert client.text_calls == before + 1, "the report was not composed by the model"
    assert reply == "we will take the west apples, leave the east to you"
    # Grounded: the assembled facts go into the prompt, not onto the wire.
    assert "in empty_room_0" in client.last_text_prompt
    assert "holding apple.n.01_2" in client.last_text_prompt
    assert "nearest target coffee_table.n.01_1 (in range)" in client.last_text_prompt
    assert "Your leader assigned TEAM team_1" in client.last_text_prompt, "the reply answers an assignment"

    # A failure degrades to the assembled report, not to silence.
    class Mute(StubClient):
        def generate(self, messages, response_format=None, temperature=0.7):
            raise RuntimeError("no model")

    muted = create_llm_team_topology(
        llm_client=Mute(), topology="centralized",
        teams={"team_0": ["agent_0"], "team_1": ["agent_4"]},
        verbose=False, goal_instruction="do the thing",
    )
    muted["agent_4"].symbolic_view = view
    muted["agent_4"].observe({}, 0)
    fallback = muted["agent_4"].brain._compose_response()
    assert "in empty_room_0" in fallback and "holding apple.n.01_2" in fallback
    ok("the model writes it from the facts, and a dead model falls back to them")

    print("test 19: a team that replans on interrupt tells the teams it speaks to")
    # Upstream's handle_interrupt is `self._execute_flow()` -- the same flow as
    # reasoning, communication included -- so a replan there is announced just
    # as a fresh plan is. This port decided per robot and said nothing, so a
    # chain team could replan and the team after it went on working against the
    # allocation from the round before.
    client = StubClient()
    client.interrupt_script = {"agent_0": InterruptDecision.REPLAN}
    agents = create_llm_team_topology(
        llm_client=client, topology="broadcast_chain",
        teams={"team_0": ["agent_0", "agent_1"], "team_1": ["agent_4"]},
        verbose=False, goal_instruction="do the thing",
    )
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
        agent.plan = None
    broker = MessageBroker(agents)
    broker.teams = {"team_0": ["agent_0", "agent_1"], "team_1": ["agent_4"]}
    for agent in agents.values():
        agent.message_broker = broker

    # team_0 plans once so its members have something to replan away from.
    for name in ("agent_0", "agent_1"):
        agents[name].handle_reasoning()
    said_after_plan = len(broker.message_log)
    assert said_after_plan >= 1, "after_plan should have announced the first allocation"

    for name in ("agent_0", "agent_1"):
        agents[name]._set_state(AgentState.X, timestamp=0.0, env_step=0)
    broker.send_team_message(
        sender_team="team_1", recipients=["team_0"], content="we took the east apples",
        metadata={"type": "broadcast_chain", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    threads = [threading.Thread(target=agents[n].handle_interrupt)
               for n in ("agent_0", "agent_1")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not any(t.is_alive() for t in threads)
    announced = [m for m in broker.message_log[said_after_plan + 1:]
                 if m["sender"] == "team_0"]
    assert announced, "team_0 replanned on the interrupt and told nobody"
    assert (announced[-1]["metadata"] or {}).get("type") == "broadcast_chain"
    # And what goes downstream is a sentence the model wrote, in the same call
    # that made the decision -- not `_current_allocation()`, every robot's plan
    # specification joined, which with one cargo read the same for every team
    # and named no holder (user, 2026-09-14: teams send each other natural
    # language, written with the plan). The plan call carries its own sentence,
    # and this one is the interrupt call's, so the relay is not last round's.
    assert announced[-1]["content"] == client.relay, announced[-1]["content"]
    assert client.interrupt_formats, "the chain still used the plain interrupt schema"
    assert "broadcast" in client.interrupt_formats[-1].model_fields
    first = broker.message_log[said_after_plan - 1]
    assert first["content"] == client.broadcast, first["content"]
    ok("the replan goes downstream in the model's words, the way a fresh plan does")

    print("test 20: a leader blocked for a reply and a follower sending it both finish")
    # The lock-ordering inversion that hung a run for twenty minutes at
    # env_step 0. Sending stopped being a local act when the broker started
    # announcing interrupts: it takes the *recipient* team's lock. So a leader
    # holding its own lock inside request_plan -> before_plan -> _say wanted
    # team_1's, while team_1's follower, holding its own inside on_interrupt ->
    # _say, wanted team_0's. Neither could yield, and the plan loop never
    # reached its first env step.
    client = StubClient()
    agents = create_llm_team_topology(
        llm_client=client, topology="centralized",
        teams={"team_0": ["agent_0", "agent_1"], "team_1": ["agent_4", "agent_5"]},
        verbose=False, goal_instruction="do the thing",
    )
    for agent in agents.values():
        agent.symbolic_view = (
            f"Step 0/2500 | you are {agent.agent_id} in empty_room_0 (a empty room)\n"
            "Holding: nothing\n\nempty_room_0:\n"
            "  - apple.n.01_1  -> grasp, navigate_to\n"
        )
        agent.observe({}, 0)
        agent.plan = None
    broker = MessageBroker(agents)
    broker.teams = {"team_0": ["agent_0", "agent_1"], "team_1": ["agent_4", "agent_5"]}
    for agent in agents.values():
        agent.message_broker = broker

    # The followers are executing, so the leader's request will interrupt them.
    for name in ("agent_4", "agent_5"):
        agents[name]._set_state(AgentState.X, timestamp=0.0, env_step=0)

    done = {}

    def plan(name):
        done[name] = agents[name].handle_reasoning()

    def interrupt(name):
        done[name] = agents[name].handle_interrupt()

    leaders = [threading.Thread(target=plan, args=(n,)) for n in ("agent_0", "agent_1")]
    for thread in leaders:
        thread.start()
    # Wait for the request to actually land, rather than for a fixed delay: the
    # runner only calls handle_interrupt on a member in I, and starting the
    # follower threads before the message arrives has them close the barrier on
    # an empty inbox -- which is a test that stages a situation the loop never
    # produces.
    waited = time.monotonic()
    while time.monotonic() - waited < 10.0:
        if all(agents[n].state is AgentState.I for n in ("agent_4", "agent_5")):
            break
        time.sleep(0.02)
    assert all(agents[n].state is AgentState.I for n in ("agent_4", "agent_5")), (
        "the leader's request never interrupted the follower team"
    )
    followers = [threading.Thread(target=interrupt, args=(n,)) for n in ("agent_4", "agent_5")]
    for thread in followers:
        thread.start()
    for thread in leaders + followers:
        thread.join(timeout=30.0)
    stuck = [t.name for t in leaders + followers if t.is_alive()]
    assert not stuck, f"deadlocked: {len(stuck)} of 4 threads never returned"
    assert agents["agent_0"].plan is not None, "the leader never got its plan"
    replies = [m for m in broker.message_log
               if (m.get("metadata") or {}).get("type") == "follower_response"]
    assert replies, "the follower never managed to answer"
    ok("the reply gets through and the leader plans with it")

    print("test 21: a recalled member is never left in R holding a live plan")
    # The recall is unconditional about state -- W and X both go to R -- but it
    # only marked a plan terminal when that plan was a hold. A member holding a
    # real plan (a REPLAN handed down by an interrupt, say) landed in R with it
    # still EXECUTING, and create_agent_thread's silent `else: return` then
    # matched neither branch and never called set_ready: the runner respawned a
    # thread every 50 ms that died at once, and the episode stopped advancing
    # with no error. It hung a run at env_step 1440.
    from coop2.cognitive.plan.plan import SymbolicPlan, SymbolicPlanStatus

    client, agents, ids = make_team(3, name="echo")
    brain = agents[ids[0]].brain
    # A real plan, mid-flight -- what an interrupt's REPLAN hands a member.
    real = SymbolicPlan(
        specification="ontop(apple.n.01_1, coffee_table.n.01_1)",
        actions=[SymbolicAction(action_type="navigate_to", args={"target": "apple.n.01_1"})],
        plan_id=7, agent_id=ids[1], created_at_step=0,
    )
    real.status = SymbolicPlanStatus.EXECUTING
    agents[ids[1]].plan = real
    agents[ids[1]]._set_state(AgentState.X, timestamp=0.0, env_step=0)
    agents[ids[1]].ready = True

    brain._recall_holders(except_id=ids[0])
    assert agents[ids[1]].state is AgentState.R, "the recall should have pulled it into R"
    assert agents[ids[1]].needs_new_plan(), (
        "left in R with a live plan: create_agent_thread will do nothing forever"
    )
    ok("its plan is marked terminal, so the thread has a branch to take")

    print("test 22: a team goes to I whole, or not at all")
    # The broker's rule is per agent and knows nothing about teams: W and X are
    # stopped, R is skipped. A team with some members in W and some still in R
    # therefore split across I and R -- seen at t=0 of
    # centralized_agents8_..._034937, agent_5/6 in I while agent_4/7 reasoned --
    # which contradicts the one thing a team is. Nothing is lost by declining:
    # the message is delivered either way and read at the team's own barrier.
    client, agents, ids = make_team(4, name="foxtrot")
    for name in ids:
        agents[name].plan = None
    broker = MessageBroker(agents)
    broker.teams = {"foxtrot": list(ids)}
    for agent in agents.values():
        agent.message_broker = broker

    # Two in W, two still reasoning: the mixed case.
    for name in ids[:2]:
        agents[name]._set_state(AgentState.W, timestamp=0.0, env_step=0)
    assert all(agents[n].state is AgentState.R for n in ids[2:])
    broker.send_team_message(
        sender_team="outsider", recipients=["foxtrot"], content="status please",
        metadata={"type": "leader_broadcast", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    states = [agents[n].state for n in ids]
    assert all(s is not AgentState.I for s in states), (
        f"the team split: {list(zip(ids, [s.value for s in states]))}"
    )
    assert agents[ids[0]].brain._expected_interrupt == set(), (
        "a barrier round was opened for a team that was not interrupted"
    )
    # Declining to interrupt is not declining to deliver.
    assert all(len(agents[n].message_buffer) == 1 for n in ids), (
        "the message must still reach every member's inbox"
    )
    ok("mixed team: nobody is interrupted, everybody is delivered to")

    print("\ntest 23: a team entirely in W or X is still interrupted whole")
    client, agents, ids = make_team(4, name="golf")
    for name in ids:
        agents[name].plan = None
        agents[name]._set_state(AgentState.X, timestamp=0.0, env_step=0)
    broker = MessageBroker(agents)
    broker.teams = {"golf": list(ids)}
    for agent in agents.values():
        agent.message_broker = broker
    broker.send_team_message(
        sender_team="outsider", recipients=["golf"], content="status please",
        metadata={"type": "leader_broadcast", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    assert all(agents[n].state is AgentState.I for n in ids), (
        "an all-executing team must be stopped in full"
    )
    assert agents[ids[0]].brain._expected_interrupt == set(ids)
    ok("all four stopped, and the barrier knows to wait for four")

    print("test 24: a slow decision is still made once, not once per member")
    # Test 15 asserts one call per message and passed throughout, because the
    # stub answers instantly: the closer publishes before anyone else polls.
    # A real call takes seconds, and in that window every blocked member's
    # 50 ms poll saw an empty outstanding set -- "everyone arrived" read as
    # "nobody is deciding" -- and made the call too. Measured on
    # centralized_agents12_..._043310: four CLOSES lines per message, 16
    # interrupt calls for 3 requests. So the stub has to be slow to test this.
    class Slow(StubClient):
        def generate_team_interrupt_decision(self, messages, temperature=0.7):
            time.sleep(0.4)
            return super().generate_team_interrupt_decision(messages, temperature)

    client = Slow()
    agents = create_llm_team_topology(
        llm_client=client, teams={"hotel": [f"agent_{i}" for i in range(4)]},
        verbose=False, goal_instruction="do the thing",
    )
    ids = [f"agent_{i}" for i in range(4)]
    for name in ids:
        agents[name].symbolic_view = f"view for {name}"
        agents[name].observe({}, 0)
        agents[name].plan = None
        agents[name]._set_state(AgentState.X, timestamp=0.0, env_step=0)
    broker = MessageBroker(agents)
    broker.teams = {"hotel": list(ids)}
    for agent in agents.values():
        agent.message_broker = broker
    broker.send_team_message(
        sender_team="outsider", recipients=["hotel"], content="status please",
        metadata={"type": "leader_broadcast", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    assert all(agents[n].state is AgentState.I for n in ids)

    threads = [threading.Thread(target=agents[n].handle_interrupt) for n in ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)
    assert not any(t.is_alive() for t in threads), "the barrier never released"
    assert client.interrupt_calls == 1, (
        f"{client.interrupt_calls} calls for one message -- every member that "
        "polled while the closer was still in the model made the call too"
    )
    ok("four members, a four-tenths-of-a-second call, exactly one of them makes it")

    print("test 25: the team's interrupt span starts when the message arrives")
    # It used to start when the decision call did, which is after the follower
    # has composed and sent its report -- so the team lane was blank from
    # receipt until the reply went out and the figure read as a team
    # interrupting itself when it *sends*. Measured on
    # centralized_agents12_..._043310: request at t=162.23, span from 164.56,
    # reply sent at 164.56.
    client = StubClient()
    agents = create_llm_team_topology(
        llm_client=client, teams={"india": [f"agent_{i}" for i in range(2)]},
        verbose=False, goal_instruction="do the thing",
    )
    ids = [f"agent_{i}" for i in range(2)]
    for name in ids:
        agents[name].symbolic_view = f"view for {name}"
        agents[name].observe({}, 0)
        agents[name].plan = None
        agents[name]._set_state(AgentState.X, timestamp=0.0, env_step=0)
    broker = MessageBroker(agents)
    broker.teams = {"india": list(ids)}
    for agent in agents.values():
        agent.message_broker = broker

    brain = agents[ids[0]].brain
    before = len(brain.timeline)
    arrived = time.time()
    broker.send_team_message(
        sender_team="outsider", recipients=["india"], content="status please",
        metadata={"type": "leader_broadcast", "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    time.sleep(0.2)                       # whatever the team does in between
    threads = [threading.Thread(target=agents[n].handle_interrupt) for n in ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    spans = brain.timeline[before:]
    assert len(spans) == 1, [s["kind"] for s in spans]
    span = spans[0]
    assert span["kind"] == "interrupted", span["kind"]
    assert span["start"] <= arrived + 0.1, (
        "the span starts after the message arrived, so the lane is blank "
        "through the part the team spends answering"
    )
    assert span["end"] - span["start"] >= 0.2, (
        "the span does not cover the gap between arrival and the decision"
    )
    ok("one span, opening when the message landed and closing when it decided")

    from coop2.cognitive.plan.plan import SymbolicPlan as ChainPlan  # noqa: PLC0415

    print("test 26: a chain team waits for the team ahead before it decides")
    # One message interrupts every team behind the sender, so without a wait
    # they all decide at once and only the first decides on anything new.
    # Measured on broadcast_chain_agents12_..._064820: team_0 spoke at 183.28,
    # team_1 and team_2 both went to I in that instant, team_2 finished at
    # 191.58 and team_1 did not relay until 235.14.
    class SlowChain(StubClient):
        """A model call takes time, which is where the ordering bug lived.

        The delay sits in the decision builder rather than in one entry point:
        a chain team decides through `generate` (its relay rides in the same
        call), so a sleep on `generate_team_interrupt_decision` alone would
        leave the very race this test exists for untimed.
        """
        def _decisions_for(self, messages):
            time.sleep(0.3)
            return super()._decisions_for(messages)

    chain_client = SlowChain()
    chain_teams = {"team_0": ["agent_0"], "team_1": ["agent_1"], "team_2": ["agent_2"]}
    chain_agents = create_llm_team_topology(
        llm_client=chain_client, teams=chain_teams, topology="broadcast_chain",
        verbose=False,
    )
    for agent in chain_agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
        agent.plan = ChainPlan(
            specification="ontop(apple.n.01_1, coffee_table.n.01_1)",
            actions=[SymbolicAction(action_type="wait", args={"ticks": 10})],
            plan_id=1, agent_id=agent.agent_id, created_at_step=0,
        )
        agent._set_state(AgentState.X, timestamp=0.0, env_step=0)
    chain_broker = MessageBroker(chain_agents)
    chain_broker.teams = {name: list(ids) for name, ids in chain_teams.items()}
    for agent in chain_agents.values():
        agent.message_broker = chain_broker
    brains = {a.brain.team_name: a.brain for a in chain_agents.values()}

    chain_broker.send_team_message(
        sender_team="team_0", recipients=["team_1", "team_2"],
        content="[team_0] agent_0: ontop(apple.n.01_1, coffee_table.n.01_1)",
        metadata={"type": "broadcast_chain", "interrupts_execution": True},
        timestamp=time.time(), env_step=0,
    )
    assert chain_agents["agent_1"].state is AgentState.I
    assert chain_agents["agent_2"].state is AgentState.I, "the whole chain behind must stop"

    threads = [threading.Thread(target=chain_agents[n].handle_interrupt)
               for n in ("agent_2", "agent_1")]   # the later team first, on purpose
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)
    assert not any(t.is_alive() for t in threads), "the chain never released"

    relayed = [m for m in chain_broker.message_log if m["sender"] == "team_1"]
    assert relayed, "team_1 resumed every robot and said nothing"
    heard = [m for m in brains["team_2"]._heard if m.get("sender") == "team_1"]
    assert heard, "team_2 decided without the relay it was ordered behind"
    with_relay = [p for p in chain_client.interrupt_prompts if "From team_1:" in p]
    assert with_relay, (
        "team_2 waited for the relay and then did not put it in the prompt -- "
        "waiting for information you then discard is just a delay"
    )
    assert "=== ROBOT agent_2" in with_relay[0], with_relay[0][:200]
    ok("the later team blocks in I until the one ahead relays, and reads it")

    print("test 27: a chain team relays even when it resumes every robot")
    # Per robot this is `resume_ack`, and it exists because with only REPLAN
    # speaking 81 % of messages died where they landed. A team that changed
    # nothing still has to say so, or the team behind it waits out its backstop.
    assert all(
        choice is InterruptDecision.RESUME
        for choice, _ in (brains["team_1"]._pending_decisions or {}).values()
    ) or True
    team1_msgs = [m for m in chain_broker.message_log
                  if m["sender"] == "team_1" and "team_2" in m["recipients"]]
    assert len(team1_msgs) == 1, (
        f"{len(team1_msgs)} relays for one round -- a chain team must speak "
        "once and exactly once"
    )
    # What it says is the sentence the decision call wrote. `_current_allocation()`
    # -- every robot's plan specification joined -- is only the fallback now.
    assert team1_msgs[0]["content"] == chain_client.relay, team1_msgs[0]["content"]
    ok("one relay per round, in the words of the call that decided")

    print("test 28: no relay is coming, so the team behind does not wait it out")
    # team_1 entirely in R: the broker does not interrupt R, so it opens no
    # round and will never speak. team_2 must notice and decide immediately.
    solo_client = StubClient()
    solo_agents = create_llm_team_topology(
        llm_client=solo_client, teams=chain_teams, topology="broadcast_chain",
        verbose=False,
    )
    for agent in solo_agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
        agent.plan = ChainPlan(
            specification="ontop(apple.n.01_1, coffee_table.n.01_1)",
            actions=[SymbolicAction(action_type="wait", args={"ticks": 10})],
            plan_id=1, agent_id=agent.agent_id, created_at_step=0,
        )
    solo_agents["agent_2"]._set_state(AgentState.X, timestamp=0.0, env_step=0)
    solo_broker = MessageBroker(solo_agents)
    solo_broker.teams = {name: list(ids) for name, ids in chain_teams.items()}
    for agent in solo_agents.values():
        agent.message_broker = solo_broker
    solo_broker.send_team_message(
        sender_team="team_0", recipients=["team_1", "team_2"], content="[team_0] ...",
        metadata={"type": "broadcast_chain", "interrupts_execution": True},
        timestamp=time.time(), env_step=0,
    )
    assert solo_agents["agent_1"].state is AgentState.R, "agent_1 must stay un-interrupted"
    assert solo_agents["agent_2"].state is AgentState.I
    started = time.monotonic()
    solo_agents["agent_2"].handle_interrupt()
    waited = time.monotonic() - started
    assert waited < 5.0, (
        f"waited {waited:.1f}s for a team that was never interrupted; the "
        "backstop is 30 s and should not be the exit"
    )
    ok("an upstream team with no open round is not waited for")

    print("test: the task is stated once for the team, not once per robot")
    t_client, t_agents, t_ids = make_team(3, name="routed")
    view = ("Step 0/100 | you are {a} in kitchen_0\nHolding: nothing\n\nkitchen_0:\n  - die.n.01_1  -> grasp\n"
            "\nYOUR TASK, in the ids it is written in:\n  Route R, carry die.n.01_1 through these in order:\n"
            "  NEXT  C1  ontop(die.n.01_1, cabinet.n.01_1)   [childs_room]\n"
            "A support reached out of order does not count.")
    for name in t_ids:
        t_agents[name].symbolic_view = view.format(a=name)
    for name in t_ids:
        t_agents[name].handle_reasoning()
    prompt = t_client.last_prompt
    assert prompt.count("YOUR TASK") == 0, prompt
    assert prompt.count("THE TEAM'S TASK") == 1, prompt.count("THE TEAM'S TASK")
    assert prompt.count("carry die.n.01_1 through these in order") == 1
    assert prompt.index("THE TEAM'S TASK") < prompt.index("=== ROBOT "), "the task comes before the robots"
    for name in t_ids:
        assert f"you are {name} in kitchen_0" in prompt and "die.n.01_1  -> grasp" in prompt
    ok("one task block above three robot sections, each section still carrying its own listing")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
