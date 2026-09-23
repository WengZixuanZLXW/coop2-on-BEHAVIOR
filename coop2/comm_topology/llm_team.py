"""One LLM for a whole team of robots -- the unit every topology is built from.

A *team* has one brain: it sees every member's observation in a single prompt and
answers with one plan per robot, so the division of labour inside a team is made
once, with all of it visible, instead of emerging from N agents that cannot see
each other's intentions.

This is not a topology of its own. It is the **unit** the topologies are
expressed in: individual / broadcast_chain / centralized /
tag / board describe how teams talk to *each other*, while
inside every team it is always one LLM driving four robots. Set the team size to 1 and each topology collapses to its old
one-LLM-per-robot behaviour, which is what makes this a generalisation rather
than a replacement.

A team speaks through one member, its **spokesagent** (the first), because the
message broker is keyed by agent id and the ordering primitives the chain and the
leader already use take agent ids. But a team is *addressed* as a whole: messages
go to every member of the recipient team, so the interrupt reaches all of them.

Three rules follow from that, and each one costs something:

**The team plans together.** Members return to reasoning as a group, only once
every one of them has finished its plan. That is the requirement, and it has a
trap in it: the plan loop does not step the environment while any agent is
not ready (``run_*.py``: ``while not all(agent.ready)``). So a member that
finished early and simply waited would freeze the world, and its teammates --
who need the world to advance to finish *their* plans -- would never finish. It
would deadlock on the first uneven round, every time.

So a member that finishes early stays **ready** and holds position instead: it
takes a short ``wait`` plan and keeps taking one until the team is complete. The
world keeps advancing, the early finisher costs real ticks doing nothing, and
that idle time is the true price of a joint decision point rather than something
hidden. :class:`TeamBrain` tracks who is done through those holds
(``_awaiting``), so a member idling for three holds is still "waiting to plan",
not "planning again".

**Messages interrupt the whole team.** A message addressed to any member
interrupts all of them, and the brain answers for each in one call: resume or
replan, per robot. Per robot rather than team-wide because a message that
changes what one robot should do usually leaves the others' plans perfectly
good, and making everyone replan would throw away work the message never
contradicted. There is no deadlock here -- an interrupt arrives at every member
at once, so the barrier closes immediately.

**One LLM call per round, not N.** The brain is shared, and the runner starts a
thread per agent, so several members reach it at the same instant. The first one
through the lock makes the call; the rest read its result. Without that, four
threads would each issue the same team-wide prompt and three answers would be
thrown away.

A team of one is the degenerate case and it behaves exactly like the individual
topology: the barrier is satisfied immediately, no holds are ever taken, and the
prompt contains one observation. That is why ``team_config`` gives an unteamed
robot a team of its own -- one-LLM-per-robot is this code with N=1, not a second
code path.
"""

from __future__ import annotations

import threading
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from coop2.cognitive.agent import LLMClient
from coop2.cognitive.agent.base_llm_agent import BaseLLMAgent
from coop2.cognitive.agent.cognitive_agent import parse_plan_response
from coop2.cognitive.agent.llm_client import (
    InterruptDecision,
    LLMChainInterruptResponse,
    LLMChainPlanResponse,
)
from coop2.cognitive.agent.memory import AgentMemory
from coop2.cognitive.agent.prompt_sections import (  # noqa: E402
    action_history_section,
    assemble_user_prompt,
    conversation_history_section,
    current_messages_section,
    observations_section,
)
from coop2.cognitive.agent.prompts import (
    build_team_system_prompt, format_message_content_for_prompt, format_plan_history,
)
from coop2.cognitive.action.action import SymbolicAction
from coop2.cognitive.agent.agent import AgentState
from coop2.cognitive.plan import SymbolicPlan

__all__ = [
    "ChainTeamBrain",
    "FollowerTeamBrain",
    "LLMTeamAgent",
    "LeaderTeamBrain",
    "TEAM_BRAIN_ROLES",
    "TeamBrain",
    "create_llm_team_topology",
    "reset_notify_budgets",
]


#: Appended to the team system prompt. Only what the world description cannot
#: say: these are properties of being asked as a team, not of the house.
#: TEAM_ROLE moved into ``prompt_sections.role_section`` (2026-09-13): the
#: system prompt is assembled there, section by section, in a fixed order.

#: How many of the team's own messages -- sent and received, oldest first --
#: the prompt quotes. The memory keeps more (`TEAM_MESSAGE_MEMORY`), so this
#: is a rendering choice rather than a loss.
TEAM_MESSAGE_HISTORY = 8
TEAM_MESSAGE_MEMORY = 32

#: Ticks a member holds for while it waits for the rest of the team. Short
#: relative to a navigate (~100 ticks of settle plus 60/m) so the team regroups
#: soon after the last member lands, rather than overshooting by a long wait.
TEAM_HOLD_TICKS = 600

#: How many `wait` actions one hold plan carries.
#:
#: A hold must be ended by the team's recall, never by running out. The FSM
#: leaves an agent in X while a plan advances through its actions and only sends
#: it to R when the plan *completes* (`plan_env_wrapper`: `is_complete()` ->
#: `set_unready('plan_terminated')`). A one-action hold therefore completed every
#: 60 ticks and cycled the member X -> R -> W -> X to ask for a plan it was never
#: going to get: agent_5 in centralized_agents8_..._040159 logged dozens of
#: reasoning/waiting pairs at one env_step, and agent_2 in the next run recorded
#: 33 separate hold plans covering 2026 of 2500 steps.
#:
#: With many actions the plan stays EXECUTING and the member stays in X for the
#: whole wait, asking once. 600 x 64 is ~38 000 ticks, past any episode this
#: task is run at, so exhaustion is a backstop and not the mechanism --
#: `_recall_holders` marks the plan terminal the moment the barrier closes and
#: the wrapper aborts the wait in flight, so a long hold costs no extra latency.
TEAM_HOLD_ACTIONS = 64

#: How long a member blocks in I waiting for its teammates to be interrupted
#: too. Generous because it should never be reached: the broker interrupts every
#: member of a team in the same call, so they arrive within milliseconds.
#: Print every arrival at the interrupt barrier. For working out why a barrier
#: closes more often than a message arrives -- which the logs alone could not
#: answer.
TEAM_VERBOSE = bool(os.environ.get("COOP2_TEAM_VERBOSE"))

INTERRUPT_BARRIER_TIMEOUT = 30.0

#: How long a chain team waits for the team ahead of it to relay before making
#: its own interrupt decision. A backstop only: the normal exit is the relay
#: arriving, or the team ahead turning out to have no open round.
CHAIN_RELAY_TIMEOUT = 30.0

#: How long a leader waits for its followers' status reports before planning
#: without them.
#:
#: The wait is safe to make blocking -- which it was not before -- because a
#: reply costs neither an env step nor an LLM call: it is a status report
#: assembled from the team's own state, and it is sent from the interrupt path,
#: so a follower answers while still in I. Every state a follower team can be in
#: can answer without the world advancing: interrupted members answer from
#: `on_interrupt`, a team already at its own barrier answers from `before_plan`.
#: So this is a backstop, not the mechanism.
LEADER_REPLY_TIMEOUT = 30.0
#: How long a follower at its planning barrier waits for the leader's
#: assignment while the leader is composing one. At step 0 every team reaches
#: its barrier at once, and the assignment is an LLM call: without this the
#: followers planned before it arrived, and the leader then waited 30 s for
#: replies from teams that had already planned (run ..._062528, 2026-09-13).
FOLLOWER_ASSIGNMENT_TIMEOUT = 90.0

#: How long a member waits in R for a team call that is already under way,
#: before giving up and holding. Long, because it is bounding an LLM call that
#: nothing else can rescue, and the alternative to waiting is showing this robot
#: as idle through its own team's decision.
TEAM_PLAN_WAIT_TIMEOUT = 120.0

#: How long the closing member waits for a teammate that has committed to a hold
#: to actually reach W, so it can be recalled. Bounded and short: the gap it
#: covers is the microseconds between deciding to hold and going ready.
COMMITTED_HOLD_SETTLE = 1.0


class TeamBrain:
    """The shared LLM behind one team, and the barrier its members meet at.

    Holds no simulation state: members push their own observations in and pull
    their plan out. Every public method is safe to call from the per-agent
    threads the runners spawn.
    """

    def __init__(
        self,
        team_name: str,
        member_ids: List[str],
        llm_client: LLMClient,
        temperature: float = 0.7,
        verbose: bool = True,
        goal_instruction: str = "",
    ):
        self.team_name = team_name
        self.member_ids = list(member_ids)
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.goal_instruction = goal_instruction
        #: Sections 5 and 6b of the prompt, reserved for digtag (the next
        #: cooperation system): its manual, and its own task observation.
        #: Empty strings add nothing to the prompt.
        self.reserved_system_prompt = ""
        self.reserved_task_observation = ""

        self.members: Dict[str, "LLMTeamAgent"] = {}
        self._lock = threading.RLock()
        # A *second* lock, for the interrupt-barrier state the broker reaches
        # into (`expect_interrupt`, `confirm_interrupts`). It exists to break a
        # lock-ordering inversion, and the rule that keeps it working is: never
        # send a message, and never call the model, while holding either lock.
        #
        # The inversion, which hung a run for twenty minutes at env_step 0:
        # sending is not a local act any more -- the broker takes the
        # *recipient* team's lock to announce the interrupt. So a leader holding
        # its own lock inside `request_plan` -> `before_plan` -> `_say` wanted
        # team_1's lock, while team_1's follower, holding its own lock inside
        # `on_interrupt` -> `_say`, wanted team_0's. Neither could yield. With
        # the barrier state under its own short-held lock and every send moved
        # outside both, there is no cycle left to form.
        self._barrier_lock = threading.RLock()
        #: Members that have finished their plan and are waiting for the team.
        #: Set when a member enters reasoning, cleared when it takes its new
        #: plan -- so it stays set across the holds a member takes meanwhile.
        self._awaiting: set = set()
        # (sender, timestamp) of the leader requests this team has already
        # reported against, so the interrupt path and the planning path cannot
        # both answer the same question.
        self._answered: set = set()
        # Which members the broker is about to interrupt, announced before it
        # interrupts any of them. Empty means "no announcement", and the barrier
        # falls back to the whole team.
        self._expected_interrupt: set = set()
        # Whether a member has already claimed this round's decision. Without
        # it, "the barrier is closed" and "the call is under way" are the same
        # observation: the closer's model call takes seconds and only sets
        # `_decided` afterwards, so every other member's 50 ms poll saw an empty
        # outstanding set and made the call too. Measured on
        # centralized_agents12_..._043310 -- four CLOSES lines per message, and
        # 16 interrupt calls for 3 requests.
        self._deciding = False
        # When the broker announced this interrupt round, i.e. when the message
        # arrived. The team is interrupted from then until its decision is
        # published, and the report it composes in between is part of that --
        # recording only the `deciding` call left the lane blank from receipt
        # until the reply went out, which reads as a team interrupting itself
        # when it *sends*. Measured on centralized_agents12_..._043310:
        # request at t=162.23, deciding span from 164.56, reply sent at 164.56.
        self._interrupt_opened: Optional[float] = None
        #: Plans produced by the last team call, drained by their owners.
        self._pending_plans: Dict[str, SymbolicPlan] = {}
        #: Members that have been told they may hold this round and have not yet
        #: been recalled. Recorded under the lock at the moment the decision is
        #: made, which is what lets the recall find a member that is still on
        #: its way to W -- see ``_recall_holders``.
        self._committed: set = set()
        self._pending_decisions: Dict[str, Any] = {}
        self._interrupted: set = set()
        #: Set once a round's decisions exist, so members blocked in I wake up.
        self._decided = threading.Event()
        self.rounds = 0
        self.holds = 0
        #: Topology wiring, in **team names** on both sides. A team is the unit
        #: that speaks and is spoken to, so the wiring says so: "team_0 ->
        #: team_1", not "team_0 -> agent_4, agent_5, agent_6, agent_7". The
        #: broker expands a team into its robots at delivery time, because a
        #: team's inbox is the union of its robots' -- but that is delivery,
        #: not addressing, and the two were conflated.
        self.wait_for: List[str] = []
        self.send_to: List[str] = []
        #: Every team in the run, so a brain can resolve the teams it is wired
        #: to into the robots whose readiness it has to check.
        self.all_teams: Dict[str, List[str]] = {}
        #: Every agent in the run, for the readiness check the ordering uses.
        self.all_agents: Dict[str, Any] = {}
        #: The other teams' brains, by name. A chain team has to know whether
        #: the team ahead of it is inside an interrupt round of its own -- if it
        #: is, a relay is coming and waiting is right; if it is not, none is
        #: coming and waiting would freeze the world for the full timeout.
        #: Readiness cannot answer that: a team ahead that is *executing* is
        #: ready, and ready is what the old ordering took as "it has spoken".
        self.peers: Dict[str, "TeamBrain"] = {}
        #: What this team heard since it last planned, folded into its prompt.
        self._heard: List[Dict] = []
        #: Every message this team has sent or received, oldest first. ``_heard``
        #: is cleared each round, so on its own the model saw only the latest
        #: turn: it re-asked questions it had asked and re-announced allocations
        #: it had made, with the earlier exchange nowhere in front of it. This
        #: is the same ring buffer every robot already keeps (`AgentMemory`),
        #: holding only messages, so a chatty round cannot evict anything else.
        self.memory = AgentMemory(max_size=TEAM_MESSAGE_MEMORY)
        #: Keys of every message ever filed, so a copy that reaches the team by
        #: a second route in a later round is not recorded twice.
        self._remembered: set = set()
        #: When the team itself was thinking. Recorded first-hand: the team is
        #: the thing that thinks here, and only it knows when it started. (Its
        #: messages need no such record -- addressed as a team, they are already
        #: team-to-team in the broker's log.)
        self.timeline: List[Dict] = []

    @property
    def speaker(self) -> Optional["LLMTeamAgent"]:
        """The member that speaks for the team."""
        for name in self.member_ids:
            if name in self.members:
                return self.members[name]
        return None

    # -- topology hooks ----------------------------------------------------

    def before_plan(self) -> None:
        """Communication that must happen before the team plans. Default: none."""

    def after_plan(self) -> None:
        """Communication that follows from the plan. Default: none."""

    def _call_plan_model(self, prompt: List[Dict[str, str]]):
        """The model call of a planning round: ``(response, usage)``.

        Overridden by a brain that needs a wider response than the plans --
        the message board's ``board_post`` rides in the same call.
        """
        return self.llm_client.generate_team_plan(messages=prompt, temperature=self.temperature)

    def _on_plan_response(self, response: Any) -> None:
        """Once per successful planning call, with the parsed response, before
        the plans are handed out. Default: nothing."""

    def _call_interrupt_model(self, prompt: List[Dict[str, str]]):
        """The model call of an interrupt round: ``(response, usage)``.

        Overridden by a brain whose interrupt answer carries more than the
        per-robot decisions -- the task-graph mode's actions and notify ride
        in the same call, as they do in its planning call.
        """
        return self.llm_client.generate_team_interrupt_decision(
            messages=prompt, temperature=self.temperature
        )

    def _on_interrupt_response(self, response: Any) -> None:
        """Once per successful interrupt call, with the parsed response, before
        the decisions are handed out. Default: nothing."""

    def _interrupt_closing(self) -> str:
        """The closing instruction of an interrupt prompt."""
        return ("Per robot: 'resume' unless the message contradicts what it is doing, "
                "else 'replan' with its new_plan.")

    def before_interrupt_decision(self, messages: List[Dict]) -> List[Dict]:
        """Runs once per interrupt round, before the model is asked.

        Returns the messages the decision is made on: @messages, plus anything
        the hook waited for. Called holding no lock and outside the barrier, so
        an override may block -- which is the point, it is where an ordered
        topology makes a team wait its turn. Default: unchanged.
        """
        return messages

    def after_interrupt(self, decisions: Dict[str, Any]) -> None:
        """Communication that follows the team's interrupt decision.

        Upstream's handle_interrupt is `self._execute_flow()` -- the same flow
        as reasoning, communication included -- so what a team decides there is
        announced exactly as a fresh plan is. It takes the whole decision map
        and not only the replanned robots, because in an ordered topology the
        successor is *waiting* for this message: a team that resumed every robot
        still has to say so, or the team behind it waits out its timeout for a
        message that was never going to come. Default: none.
        """

    def on_interrupt(self, messages: List[Dict]) -> None:
        """Called as a member reaches the interrupt barrier, before deciding.

        The follower role answers its leader from here, which is what lets the
        leader block for the reply: answering needs neither an env step nor an
        LLM call, so a follower can do it while still in I. Called with
        ``_lock`` held, once per arriving member, so an override has to be
        idempotent within a round.
        """

    def _collect_heard(self) -> None:
        """Drain every member's inbox into the team's shared record.

        Safe to call more than once in a round: draining is destructive, so the
        second call simply finds nothing, and what the first call took is still
        in ``_heard`` for the prompt.
        """
        for name in self.member_ids:
            member = self.members.get(name)
            if member is None:
                continue
            for message in member.get_messages(clear_buffer=True) or []:
                # One message to the team, not one per robot. Delivery copies it
                # into every member's inbox -- an interrupt has to stop all of
                # them -- so draining four inboxes yields the same message four
                # times, and the prompt quoted it four times.
                self._remember(message)

    @staticmethod
    def _message_key(message: Dict):
        """Identity of a message, for de-duping and for round scoping."""
        return (message.get("sender"), message.get("timestamp"),
                str(message.get("content")))

    def _file_heard(self, messages: List[Dict]) -> None:
        """Record messages already taken off a member's inbox.

        ``handle_interrupt`` drains the inbox before the brain sees it, so
        without this an interrupting message informed the resume/replan call and
        then vanished -- it never reached ``_heard``, so it never reached the
        next planning prompt and ``_was_asked`` never saw it.

        That is not a cosmetic loss. A centralized leader's request interrupts
        by design, so from the second round on the follower team was answering a
        question it no longer held: measured on
        ``centralized_agents8_..._235226``, team_1 replied to the round-1 request
        (it was still in R, which the broker does not interrupt) and to neither
        of the two after it.

        Callers must hold ``_lock``.
        """
        for message in messages or []:
            self._remember(message)

    def _remember(self, message: Dict) -> bool:
        """File one incoming message: into this round's ``_heard`` and into the
        team's memory. Returns False if it was already there.

        Both routes in -- draining inboxes and the interrupt handler -- can
        deliver the same message, and each member holds a copy, so the
        de-duplication lives here rather than in each caller.
        """
        key = self._message_key(message)
        if key in self._remembered:
            return False
        self._remembered.add(key)
        self._heard.append(message)
        self.memory.record_message_in(
            sender=message.get("sender", "unknown"),
            recipients=[self.team_name],
            content=message.get("content", ""),
            timestamp=message.get("timestamp"),
            env_step=message.get("env_step") or 0,
        )
        return True

    def _env_step(self) -> int:
        return max((getattr(m, "env_step", 0) or 0 for m in self.members.values()), default=0)

    def _messages_block(self, heading: str = "## 8. CONVERSATION HISTORY",
                        exclude: Optional[List[Dict]] = None) -> str:
        """What this team has said and been told, oldest first.

        Both directions. A team shown only what it was told re-asks questions
        it has already asked and re-announces allocations it has already made.
        What arrived since the last plan is marked ``(new)``, so the model can
        tell the news from the record; older lines are cut short, because the
        record is context and the news is what it must act on. @exclude drops
        messages quoted elsewhere in the same prompt.
        """
        def identity(m):
            return (m.get("sender"), str(m.get("content", "")))

        skip = {identity(m) for m in (exclude or [])}
        new = {identity(m) for m in self._heard}
        events = [e for e in self.memory.get_messages() if identity(e) not in skip]
        # In the order they happened, not the order they were filed: a reply
        # is filed when the team next drains its inboxes, which can be after
        # the team's own later message was recorded (user, 2026-09-13). The
        # broker stamps every message; a stable sort keeps ties in file order.
        events = sorted(enumerate(events), key=lambda ie: (float(ie[1].get("timestamp") or 0.0), ie[0]))
        events = [e for _, e in events]
        if not events:
            return ""
        lines = [f"\n{heading} (oldest first):"]
        for event in events[-TEAM_MESSAGE_HISTORY:]:
            content = " ".join(str(format_message_content_for_prompt(event.get("content", ""))).split())
            if event.get("type") == "message_out":
                who, is_new = f"You told {', '.join(event.get('recipients') or [])}", False
            else:
                who, is_new = f"From {event.get('sender', 'unknown')}", identity(event) in new
            if not is_new and len(content) > 240:
                content = content[:237] + "..."
            lines.append(f"  [step {event.get('env_step', '?')}] {who}{' (new)' if is_new else ''}: {content}")
        return "\n".join(lines)

    def _say(self, content: str, kind: str, interrupts: bool,
             expected_reply: bool = False, recipients: Optional[List[str]] = None) -> None:
        """Send @content to every member of the teams this one addresses.

        ``expected_reply`` marks an answer the recipient asked for and is
        blocked waiting on, which the broker then delivers without interrupting
        anyone -- see the note there. ``recipients`` names the teams for this
        one send; the default is the topology's fixed ``send_to``. The notify
        tool is the case that chooses per call.
        """
        broker = self._broker()
        targets = list(recipients) if recipients is not None else list(self.send_to)
        if not targets or broker is None:
            return
        # Stamped before it goes out: a reply written synchronously inside the
        # delivery (a follower answering from on_interrupt) would otherwise be
        # stamped earlier than the message it answers, and the history sorted
        # by time would show the answer first.
        sent_at = time.time()
        broker.send_team_message(
            timestamp=sent_at,
            sender_team=self.team_name,
            recipients=targets,
            content=content,
            metadata={
                "type": kind,
                "interrupts_execution": interrupts,
                "expected_reply": expected_reply,
            },
        )
        # What the team said is part of its record too: the broker logs it, but
        # nothing on the team's side did, so its next prompt showed the other
        # teams' words and none of its own.
        self.memory.record_message_out(
            sender=self.team_name, recipients=targets, content=content,
            # On the broker's clock (relative to the agents' start), the same
            # clock every incoming message carries, so the two sort together.
            timestamp=broker._relative(sent_at, list(self.member_ids)), env_step=self._env_step(),
        )
        if self.verbose:
            print(f"  [{self.team_name}] -> {targets} ({kind}): {content[:60]}...")

    def _broker(self):
        """The message broker, with every team declared on it.

        Registration happens here rather than at construction because the broker
        is attached to the agents by the environment, after the topology is
        built. All teams are declared at once, not just this one: a team has to
        be known before it can be addressed, and the first team to speak would
        otherwise be addressing teams the broker had never heard of.
        """
        speaker = self.speaker
        broker = speaker.message_broker if speaker is not None else None
        if broker is None:
            return None
        for name, members in (self.all_teams or {self.team_name: self.member_ids}).items():
            if name not in broker.teams:
                broker.register_team(name, members)
        return broker

    def _members_of(self, team_names) -> List[str]:
        """The robots behind @team_names."""
        return [member for name in team_names for member in self.all_teams.get(name, [])]

    def _await_speakers(self, timeout: float = 30.0) -> None:
        """Block until the teams this one waits on have spoken, or committed."""
        speaker = self.speaker
        if speaker is None or not self.wait_for:
            return
        if self._heard_from(self.wait_for):
            # Already in hand: the brain collects every member's inbox before
            # the hooks run, so by here the message it is waiting for may have
            # been taken off the buffer `wait_for_messages_from` inspects.
            # Without this the team waits out the full timeout for something it
            # is already holding.
            return
        # Readiness is still per robot -- it is robots that execute -- so the
        # teams waited on are resolved to their members here. The wait itself is
        # on the *team* having spoken.
        upstream = self._members_of(self.wait_for)
        if not speaker._any_waiting_agent_not_ready(self.all_agents, upstream):
            return
        deadline = time.monotonic() + timeout
        while not speaker.wait_for_messages_from(self.wait_for):
            if not speaker._any_waiting_agent_not_ready(self.all_agents, upstream):
                break
            if time.monotonic() >= deadline:
                print(f"  [{self.team_name}] waited {timeout:.0f}s for {self.wait_for}; releasing")
                break
            time.sleep(0.05)

    def _heard_from(self, senders) -> bool:
        """Has anything from @senders reached the team this round?"""
        return any(message.get("sender") in set(senders) for message in self._heard)

    def _plan_summary(self, plans: Dict[str, SymbolicPlan]) -> str:
        """One line per robot, for telling another team what this one will do."""
        parts = [f"{name}: {plan.specification}" for name, plan in sorted(plans.items())]
        return f"[{self.team_name}] " + "; ".join(parts)

    # -- registration ------------------------------------------------------

    def register(self, agent: "LLMTeamAgent") -> None:
        self.members[agent.agent_id] = agent

    @property
    def size(self) -> int:
        return len(self.member_ids)

    # -- planning ----------------------------------------------------------

    def request_plan(self, agent_id: str) -> Optional[SymbolicPlan]:
        """The team's plan for @agent_id, or None if the team is not ready yet.

        None means "hold and ask again": teammates are still executing, and the
        caller must stay ready so the world keeps advancing for them.
        """
        with self._lock:
            if agent_id in self._pending_plans:
                self._awaiting.discard(agent_id)
                return self._pending_plans.pop(agent_id)

            self._awaiting.add(agent_id)
            if not self._awaiting.issuperset(self.member_ids):
                return None

            # Every member is done. Before anything else, pull the teammates
            # that are still holding out of their holds and into R, so the whole
            # team is *reasoning* while the one call happens -- and so that
            # anything this team says to another team is said by a team that has
            # already stopped acting. Without this only the member that closed
            # the barrier showed as reasoning and the rest kept executing.
            self._recall_holders(except_id=agent_id)

            # One call for the whole team; whichever thread got here first does
            # it and the others take from the result. The topology's
            # communication brackets that call: whoever this team waits on has
            # to have spoken before it plans, and whatever it tells other teams
            # follows from the plan it just made.
            started = time.time()
            # Twice, and both are load-bearing. Before, because a hook reads
            # what has already arrived -- FollowerTeamBrain._was_asked is a
            # query against _heard. After, because the hooks *wait*: the chain
            # blocks in _await_speakers for the team ahead of it and the leader
            # blocks for its followers' replies, and collecting only beforehand
            # built the prompt from a snapshot taken before the very message the
            # team had just waited for. Measured: team_1 waited for team_0,
            # received its allocation, and planned without it -- then quoted it
            # one round later, at env_step 3741, by which time it was stale.
            # _collect_heard is idempotent, so the second call is free when the
            # first already took everything.
            self._collect_heard()
            self.before_plan()
            self._collect_heard()
            self._pending_plans = self._generate_team_plans()
            self.rounds += 1
            self.after_plan()
            self._heard = []
            self._committed.clear()
            self.timeline.append(
                {"kind": "planning", "start": started, "end": time.time()}
            )
            self._awaiting.discard(agent_id)
            return self._pending_plans.pop(agent_id, None)


    def _recall_holders(self, except_id: str) -> None:
        """Bring every holding teammate into R for the duration of the call.

        A member waiting for the team is *executing* a hold, which is what keeps
        the world moving while its teammates finish. Once the barrier closes
        nobody needs the world any more, so the holds are ended here and the
        whole team reasons together.

        Ending a hold takes two steps, and one alone is not enough.
        ``set_unready`` puts the member in R, but ``create_agent_thread`` only
        calls ``handle_reasoning`` when ``needs_new_plan()`` is also true -- and
        that asks the *plan* whether it is finished. A member left in R with a
        live hold plan would answer no, return early, never become ready, and
        hang the run. So the hold is marked terminal first.

        The stale ``wait`` primitive it leaves in the engine is not a leak: the
        wrapper aborts it when the replacement plan is committed
        (``_reset_symbolic_action_state``), which is the same path an
        interrupt-and-replan already takes.
        """
        from coop2.cognitive.agent.agent import AgentState  # noqa: PLC0415
        from coop2.cognitive.plan.plan import SymbolicPlanStatus  # noqa: PLC0415

        for name in self.member_ids:
            if name == except_id:
                continue
            member = self.members.get(name)
            if member is None:
                continue
            if name in self._committed:
                # It was told it could hold and is between that decision and
                # actually going ready -- microseconds, but a real gap: it is
                # still in R, which the check below deliberately leaves alone,
                # and it would then sit in W through the whole call. Measured at
                # 10.9 s of "waiting" for agent_5 while its three teammates
                # reasoned. Wait for it to land rather than miss it; the call
                # this recall precedes takes seconds, so this costs nothing.
                settle = time.monotonic() + COMMITTED_HOLD_SETTLE
                while member.state == AgentState.R and time.monotonic() < settle:
                    time.sleep(0.001)
                self._committed.discard(name)
            # Mark whatever plan it has, not only a hold. The recall is
            # unconditional about *state* -- W and X are both pushed to R -- so
            # it has to be unconditional about the plan too, or it leaves an
            # agent in R holding a live plan. create_agent_thread then takes its
            # silent `else: return`: R with needs_new_plan() false matches
            # neither branch and never reaches set_ready, so the runner respawns
            # a thread every 50 ms that dies at once and the episode stops
            # advancing with no error and no traceback. That hung a run at
            # env_step 1440, and the only thing that showed it was a SIGUSR1
            # dump -- main thread in wait_for_state_change, not one agent thread
            # alive.
            #
            # Safe because the team is about to issue a plan for every member:
            # _generate_team_plans covers them all and claim_pending_plan hands
            # each one out, so the plan marked here is superseded either way. A
            # REPLAN handed down by an interrupt is exactly the live plan that
            # used to fall through this gap.
            plan = member.plan
            if plan is not None and plan.status not in (
                SymbolicPlanStatus.SUCCESS,
                SymbolicPlanStatus.FAILED,
                SymbolicPlanStatus.INTERRUPTED,
            ):
                plan.status = SymbolicPlanStatus.INTERRUPTED
            # Both W and X. A member that finished its hold and is sitting ready
            # is idling just as much as one still running it, and skipping the W
            # ones left a robot showing "waiting" while its teammates reasoned.
            # Members already in R or I are left alone -- they are not idling.
            if member.state in (AgentState.W, AgentState.X):
                member.set_unready(reason="team_recalled")

    def claim_pending_plan(self, agent_id: str) -> Optional[SymbolicPlan]:
        """Wait for this round's plan if the team is complete; else None.

        Closes a race the recall cannot. ``request_plan`` returns None and the
        caller then decides to hold; if the last teammate arrives in that gap,
        the recall finds this member in R -- which it deliberately leaves alone,
        R not being an idle state -- and the member then goes ready with a hold
        and sits in W through its own team's call. Measured at 15.8 s of
        "waiting" against 15.8 s of "reasoning" for its teammates.

        Asking once is not enough: the recall happens *before* the call, so at
        that moment there is no plan to hand back yet. What settles it is
        whether the team is complete. If it is, someone is about to make the
        call or is making it, so this member waits for the result -- in R, where
        it belongs, and safely: the world being frozen is what the whole team is
        waiting on anyway, and the call needs no simulation. If it is not, the
        member really does have teammates still working, and holds.
        """
        with self._lock:
            if agent_id in self._pending_plans:
                self._awaiting.discard(agent_id)
                return self._pending_plans.pop(agent_id)
            if not self._awaiting.issuperset(self.member_ids):
                # Teammates really are still working. Commit to the hold here,
                # inside the lock, so that a teammate arriving a moment later
                # cannot close the barrier without knowing this member is on its
                # way to W.
                self._committed.add(agent_id)
                # Counted here rather than by a separate note_hold() call: that
                # call took this same lock, so a member commiting to a hold
                # would have blocked on the closer that is spinning for it to
                # reach W -- the two waiting on each other for the full settle.
                self.holds += 1
                return None
            # The round this member is waiting to see finish. Counting rounds
            # rather than looking for an empty _pending_plans is what separates
            # "the call has not started" from "the call finished without me".
            started_at = self.rounds

        deadline = time.monotonic() + TEAM_PLAN_WAIT_TIMEOUT
        while True:
            # Blocking on the lock is most of the wait: whoever closed the
            # barrier holds it for the whole LLM call.
            with self._lock:
                if agent_id in self._pending_plans:
                    self._awaiting.discard(agent_id)
                    return self._pending_plans.pop(agent_id)
                if self.rounds != started_at:
                    # The call happened and had nothing for this member, which
                    # _generate_team_plans has already covered with a hold.
                    return None
            if time.monotonic() >= deadline:
                print(f"  [{self.team_name}] {agent_id} waited "
                      f"{TEAM_PLAN_WAIT_TIMEOUT:.0f}s for the team plan; holding")
                return None
            time.sleep(0.01)

    def _generate_team_plans(self) -> Dict[str, SymbolicPlan]:
        """One LLM call, one plan per member. Never raises."""
        members = [self.members[name] for name in self.member_ids if name in self.members]
        if not members:
            return {}
        anchor = members[0]
        prompt = self._build_team_prompt(members)

        if anchor._should_print_llm_io():
            anchor._print_llm_messages(f"Team Plan Generation [{self.team_name}]", prompt)
        if self.verbose:
            print(f"  [{self.team_name}] Calling LLM for {len(members)} plans...")

        try:
            response, usage = self._call_plan_model(prompt)
            anchor._record_llm_usage(
                usage, f"Team Plan Generation [{self.team_name}]", prompt, response
            )
        except Exception as error:  # noqa: BLE001 - a dead LLM must not end the run
            anchor._record_llm_error(error)
            print(f"  [{self.team_name}] LLM error: {error}")
            return {name: self.members[name]._generate_fallback_plan() for name in self.member_ids
                    if name in self.members}

        by_agent = {plan.agent_id: plan for plan in response.plans}
        plans: Dict[str, SymbolicPlan] = {}
        for member in members:
            entry = by_agent.get(member.agent_id)
            if entry is None:
                # The model skipped a robot. Hold it rather than leaving it with
                # no plan at all, which would strand the whole team at the next
                # barrier: a member with no plan never becomes ready.
                print(f"  [{self.team_name}] no plan returned for {member.agent_id}; holding it")
                plans[member.agent_id] = member.build_hold_plan()
                continue
            single = _as_plan_response(entry)
            plan = parse_plan_response(
                llm_response=single,
                agent_id=member.agent_id,
                env_step=member.env_step,
                plan_id=member.plan_count + 1,
            )
            plans[member.agent_id] = member._finalize_generated_plan(plan)
            if self.verbose:
                print(f"  [{member.agent_id}] Plan: {plan.specification}")
        if self.verbose and getattr(response, "reasoning", ""):
            print(f"  [{self.team_name}] allocation: {response.reasoning}")
        try:
            self._on_plan_response(response)
        except Exception as error:  # noqa: BLE001 - a bad post must not cost the plans
            print(f"  [{self.team_name}] plan response hook failed: {error}")
        return plans

    # -- interrupts --------------------------------------------------------

    def request_interrupt_decision(self, agent_id: str, messages: List[Dict]) -> Optional[Any]:
        """Resume/replan for @agent_id, once the interrupted teammates arrive.

        Nothing here holds a lock across a send or a model call -- see the note
        on ``_barrier_lock``.
        """
        with self._lock:
            # File first, unconditionally: these came off the member's inbox and
            # this is their only remaining route into the team's record.
            self._file_heard(messages)

        # Answer before deciding rather than after, so a leader blocked on this
        # reply is not made to wait for the follower's own model round trip.
        # Outside the lock: it sends.
        self.on_interrupt(messages)

        with self._barrier_lock:
            if TEAM_VERBOSE:
                member = self.members.get(agent_id)
                print(f"  [barrier {self.team_name}] {agent_id} arrives "
                      f"state={getattr(getattr(member, 'state', None), 'value', '?')} "
                      f"msgs={len(messages)} expected={sorted(self._expected_interrupt)} "
                      f"arrived={sorted(self._interrupted)} "
                      f"pending={sorted(self._pending_decisions)}")
            if agent_id in self._pending_decisions:
                return self._pending_decisions.pop(agent_id)
            member = self.members.get(agent_id)
            if member is not None and member.state is not AgentState.I:
                # Not interrupted, so not part of this round. `create_agent_thread`
                # only calls handle_interrupt for a member in I, and a member
                # that is not in one closes the barrier on nobody: it did that
                # once and spent a decision call answering an empty inbox.
                return None
            self._interrupted.add(agent_id)
            # Wait only for teammates that are actually *in* I. Requiring every
            # member assumed "a message interrupts every member at once", and
            # the broker interrupts an agent in W or X and skips one in R --
            # which is where the member running the team's planning call sits.
            closing = self._claim_decision()
        if closing:
            return self._decide_and_publish(agent_id, messages)

        # Not everyone has arrived. **Block here**, staying in I, rather than
        # returning and being marked ready again: a member that bounced straight
        # back to W showed as "waiting" while its teammates were still
        # interrupted, and the team is supposed to decide together.
        deadline = time.monotonic() + INTERRUPT_BARRIER_TIMEOUT
        while time.monotonic() < deadline:
            if self._decided.wait(timeout=0.05):
                with self._barrier_lock:
                    return self._pending_decisions.pop(agent_id, None)
            # A teammate this barrier was waiting for may have left I without
            # ever arriving -- never interrupted, or its own wait timed out, or
            # the broker's confirmation narrowed the set. Re-check rather than
            # hold the world until the deadline.
            with self._barrier_lock:
                closing = self._claim_decision()
            if closing:
                return self._decide_and_publish(agent_id, messages)
        print(f"  [{self.team_name}] waited {INTERRUPT_BARRIER_TIMEOUT:.0f}s for the team "
              f"to be interrupted and it never completed; resuming")
        with self._barrier_lock:
            # Close the round even though nothing was decided. A team waiting on
            # this one reads `has_open_interrupt_round`, and a round left open
            # here would make it wait out its own backstop every round after.
            self._interrupt_opened = None
        return None

    def has_open_interrupt_round(self) -> bool:
        """Has this team been interrupted and not yet published its decision?

        `_interrupt_opened` is stamped by `expect_interrupt`, which the broker
        calls for every addressed team *before* it stops anybody, and cleared by
        `_decide_and_publish`. So between those two points the answer is yes,
        and a team waiting on this one knows a message is still coming.
        """
        with self._barrier_lock:
            return self._interrupt_opened is not None

    def _claim_decision(self) -> bool:
        """May this member make the call? True for exactly one per round.

        Callers must hold ``_barrier_lock``.
        """
        if self._deciding or self._outstanding():
            return False
        self._deciding = True
        return True

    def _decide_and_publish(self, agent_id: str, messages: List[Dict]) -> Optional[Any]:
        """One model call for the whole team, then hand out the answers.

        Called holding nothing: the call itself and the announcement that
        follows it both reach outside this team.
        """
        if TEAM_VERBOSE:
            print(f"  [barrier {self.team_name}] {agent_id} CLOSES and decides "
                  f"(expected={sorted(self._expected_interrupt)} "
                  f"arrived={sorted(self._interrupted)})")
        # Before the model call and outside every lock, because an ordered
        # topology waits here -- see `before_interrupt_decision`.
        messages = self.before_interrupt_decision(messages)
        started = time.time()
        decisions = self._decide_interrupts(messages)
        with self._barrier_lock:
            self._pending_decisions = decisions
            self._deciding = False
            # From the message arriving, not from the model call starting: the
            # report composed in between is the team handling the interrupt too.
            self.timeline.append({
                "kind": "interrupted",
                "start": self._interrupt_opened
                if self._interrupt_opened is not None else started,
                "end": time.time(),
            })
            self._decided.set()
            mine = self._pending_decisions.pop(agent_id, None)
        self.after_interrupt(decisions)
        with self._barrier_lock:
            # Closed only now, after the announcement has actually gone out.
            # Per robot the ordering comes for free -- upstream broadcasts in
            # step 4 of `_execute_flow` and `create_agent_thread` calls
            # `set_ready()` only once the handler returns, so speaking strictly
            # precedes going ready. A team inverts that: `_decided.set()` above
            # releases four members, who go ready while `after_interrupt` has
            # not sent yet. Anything reading "is that team still going to
            # speak?" in that window would be told no and would be wrong, so the
            # round stays open across the send.
            self._interrupt_opened = None
        return mine

    def expect_interrupt(self, names: List[str]) -> None:
        """The broker is about to interrupt @names. Opens a new barrier round.

        Called before any of them is actually stopped, so the barrier knows how
        many to wait for instead of inferring it from who happens to be in I --
        an inference that closed early and spent one LLM call per member.
        """
        with self._barrier_lock:
            self._expected_interrupt = set(names)
            self._interrupt_opened = time.time()
            self._deciding = False
            self._interrupted.clear()
            self._pending_decisions = {}
            self._decided.clear()

    def confirm_interrupts(self, names: List[str]) -> None:
        """Narrow the expected set to the members really stopped.

        `expect_interrupt` runs before the delivery loop and is a prediction;
        this runs after it and is the fact. Narrowing rather than replacing,
        because a member that has already reached the barrier belongs in the
        round whatever the predicate thought of it.
        """
        with self._barrier_lock:
            if not self._expected_interrupt:
                return
            self._expected_interrupt = set(names) | set(self._interrupted)

    def _outstanding(self) -> List[str]:
        """Members this round is still waiting for.

        The set comes from the broker's announcement, which names only the
        members a delivery actually stops -- a member in R is not interrupted,
        so the barrier must not wait for it. With no announcement (a topology
        that never calls it, or a message that interrupted nobody) it falls back
        to the whole team, which is the old behaviour.

        Callers must hold ``_barrier_lock``.
        """
        expected = set(self._expected_interrupt)
        if not expected:
            # Nothing was announced -- a topology that does not use the hook, or
            # a member interrupted by something other than a team delivery. Fall
            # back to inspecting state: whoever is in I is coming, whoever is
            # not never will be. That is racy under a real delivery, which is
            # exactly why the broker announces; here it is the only signal.
            from coop2.cognitive.agent.agent import AgentState  # noqa: PLC0415

            expected = {
                name for name in self.member_ids
                if self.members.get(name) is not None
                and self.members[name].state is AgentState.I
            }
            expected |= self._interrupted
        return [name for name in expected if name not in self._interrupted]

    def _decide_interrupts(self, messages: List[Dict]) -> Dict[str, Any]:
        members = [self.members[name] for name in self.member_ids if name in self.members]
        if not members:
            return {}
        anchor = members[0]
        prompt = self._build_interrupt_prompt(members, messages)

        if anchor._should_print_llm_io():
            anchor._print_llm_messages(f"Team Interrupt [{self.team_name}]", prompt)
        try:
            response, usage = self._call_interrupt_model(prompt)
            anchor._record_llm_usage(
                usage, f"Team Interrupt [{self.team_name}]", prompt, response
            )
        except Exception as error:  # noqa: BLE001
            anchor._record_llm_error(error)
            print(f"  [{self.team_name}] interrupt LLM error: {error}; everyone resumes")
            return {name: (InterruptDecision.RESUME, None) for name in self.member_ids}

        decisions: Dict[str, Any] = {}
        by_agent = {d.agent_id: d for d in response.decisions}
        for member in members:
            entry = by_agent.get(member.agent_id)
            if entry is None or entry.decision == InterruptDecision.RESUME:
                decisions[member.agent_id] = (InterruptDecision.RESUME, None)
                continue
            plan = None
            if entry.new_plan is not None:
                plan = parse_plan_response(
                    llm_response=entry.new_plan,
                    agent_id=member.agent_id,
                    env_step=member.env_step,
                    plan_id=member.plan_count + 1,
                )
                plan = member._finalize_generated_plan(plan)
            decisions[member.agent_id] = (InterruptDecision.REPLAN, plan)
        if self.verbose:
            summary = ", ".join(
                f"{name}={decisions[name][0].value}" for name in decisions
            )
            print(f"  [{self.team_name}] interrupt: {summary}")
        try:
            self._on_interrupt_response(response)
        except Exception as error:  # noqa: BLE001 - a bad side effect must not cost the decisions
            print(f"  [{self.team_name}] interrupt response hook failed: {error}")
        return decisions

    # -- prompts -----------------------------------------------------------

    #: Section 4 of the system prompt. Subclasses name their mode.
    COOPERATION_MODE = "individual"

    def _system_prompt(self, anchor: "LLMTeamAgent") -> str:
        """Sections 1-5, in order: role, environment, robots, cooperation
        mode, the reserved digtag manual (``self.reserved_system_prompt``)."""
        members = [self.members[n] for n in self.member_ids if n in self.members]
        return build_team_system_prompt(
            self.team_name,
            [m.agent_id for m in members],
            max_actions=6,
            cooperation_mode=self.COOPERATION_MODE,
            robot_profiles=self._robot_profiles(members),
            lift_rules=self._lift_rules(members),
            reserved=getattr(self, "reserved_system_prompt", ""),
            cooperation_extra=self._cooperation_extra(),
        )

    def _cooperation_extra(self) -> str:
        """Text appended to section 4 by a mode with an optional tool. Default: none."""
        return ""

    @staticmethod
    def _world_observation(member: "LLMTeamAgent"):
        """The member's SymbolicObservation, if its last observe() carried one."""
        direct = getattr(member, "world_observation", None)
        if direct is not None:
            return direct
        raw = getattr(member, "observation", None)
        if isinstance(raw, dict):
            return raw.get("symbolic_world_state")
        return None

    def _robot_profiles(self, members) -> Dict[str, Dict[str, object]]:
        """``{robot: flags}`` for section 3's roster, read off each robot's own
        observation; a robot without one is left out rather than guessed."""
        profiles: Dict[str, Dict[str, object]] = {}
        for member in members:
            obs = self._world_observation(member)
            if obs is None:
                continue
            me = next((e for e in getattr(obs, "entities", {}).values()
                       if getattr(e, "name", None) == obs.agent_id), None)
            if me is None:
                continue
            profiles[member.agent_id] = {
                "is_carrier": bool(getattr(me, "is_carrier", False)),
                "base_locked_while_holding": bool(getattr(me, "base_locked_while_holding", False)),
                "lift_role": getattr(me, "lift_role", None),
            }
        return profiles

    def _lift_rules(self, members):
        for member in members:
            obs = self._world_observation(member)
            rules = getattr(obs, "lift_rules", None) if obs is not None else None
            if rules:
                return dict(rules)
        return None

    TASK_HEADER = "\nYOUR TASK, in the ids it is written in:"

    @classmethod
    def _split_task(cls, view: str) -> Tuple[str, str]:
        """``(view without its YOUR TASK block, that block)``.

        The per-robot view carries the task because a single-robot prompt has
        nowhere else to put it. In a team prompt that put the same ten-node
        route under every one of nine robots (user, 2026-09-13: "repeated").
        The block runs from its header to the next blank-line section
        (``Goal:``) or the end.
        """
        if not view or cls.TASK_HEADER not in view:
            return view, ""
        head, rest = view.split(cls.TASK_HEADER, 1)
        cut = rest.find("\n\nGoal:")
        task, tail = (rest, "") if cut < 0 else (rest[:cut], rest[cut:])
        return head.rstrip("\n") + tail, cls.TASK_HEADER.strip("\n") + task.rstrip("\n")

    def _team_task_block(self, members: List["LLMTeamAgent"]) -> str:
        """The task once, for the whole team: the first member's, since every
        robot of one activity is told the same thing."""
        for member in members:
            _, task = self._split_task(member.symbolic_view or "")
            if task:
                return task.replace("YOUR TASK, in the ids it is written in:", "THE TEAM'S TASK:", 1)
        return ""

    #: The per-robot view's first line and its house line, both of which the
    #: team prompt states once instead of once per robot.
    _STEP_PREFIX = re.compile(r"^Step \d+(?:/\d+)? \| ")
    _HOUSE_PREFIX = "Rooms in this house"

    def _house_line(self, members: List["LLMTeamAgent"]) -> str:
        """The "Rooms in this house" line, once: it is the same for every
        robot, and nine robots repeated it nine times."""
        for member in members:
            for line in (member.symbolic_view or "").splitlines():
                if line.startswith(self._HOUSE_PREFIX):
                    return line
        return ""

    def _member_block(self, member: "LLMTeamAgent") -> str:
        """One robot's section of the team prompt.

        The per-agent observation is reused rather than re-rendered -- it is
        the same text a single-agent topology would send -- minus its YOUR
        TASK block and its house line, which the team prompt states once
        (`_team_task_block`, `_house_line`), and minus the "Step N/M | "
        prefix, which section 6's header already carries.
        """
        lines = [f"=== ROBOT {member.agent_id} ==="]
        if member.symbolic_view:
            view, _ = self._split_task(member.symbolic_view)
            kept = [l for l in view.splitlines() if not l.startswith(self._HOUSE_PREFIX)]
            if kept:
                kept[0] = self._STEP_PREFIX.sub("", kept[0])
            lines.append("\n".join(kept))
        elif member.target_hints:
            lines.append(member.target_hints)
        else:
            lines.append("(no observation yet)")
        return "\n".join(lines)

    def _action_history_block(self, member: "LLMTeamAgent") -> str:
        """Section 9, one robot: what it is doing now and its last plans.

        Per robot, not per team: each robot finished its own plans for its own
        reasons, and a merged list would leave the model to guess which of
        four robots a failure belonged to.
        """
        lines = [f"--- {member.agent_id} ---"]
        plan = member.plan
        if plan is not None:
            lines.append(f"Its current plan: {plan.specification} [{plan.status.value}]")
        history = format_plan_history(getattr(member, "plan_history", []))
        if history:
            lines.append(history)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _render_message(self, message: Dict) -> str:
        return f"From {message.get('sender', 'unknown')}: {message.get('content', '')}"

    def _current_messages_block(self, messages: List[Dict]) -> str:
        """Section 7. Default: the messages that arrived, worded for the mode.
        The board mode puts the shared board here, above any direct message."""
        return current_messages_section(self.COOPERATION_MODE, messages, self._render_message)

    def _plan_closing(self, members: List["LLMTeamAgent"]) -> str:
        return (f"Return exactly {len(members)} plans, one per robot, in each robot's "
                "own ids; `reasoning` says how the work was divided.")

    def _build_team_prompt(self, members: List["LLMTeamAgent"]) -> List[Dict[str, str]]:
        """Sections 6-9, then the closing instruction."""
        anchor = members[0]
        new = list(self._heard)
        user = assemble_user_prompt(
            observations=observations_section(
                anchor.env_step, self.team_name,
                [self._member_block(m) for m in members],
                task_block=self._team_task_block(members),
                goal_instruction=self.goal_instruction,
                reserved_task_observation=getattr(self, "reserved_task_observation", ""),
                house=self._house_line(members),
            ),
            current_messages=self._current_messages_block(new),
            conversation_history=conversation_history_section(
                self._messages_block("## 8. CONVERSATION HISTORY", exclude=new)),
            action_history=action_history_section([self._action_history_block(m) for m in members]),
            closing=self._plan_closing(members),
        )
        return [
            {"role": "system", "content": self._system_prompt(anchor)},
            {"role": "user", "content": user},
        ]

    def _build_interrupt_prompt(
        self, members: List["LLMTeamAgent"], messages: List[Dict]
    ) -> List[Dict[str, str]]:
        """The same sections; the interrupting messages are section 7."""
        anchor = members[0]
        user = assemble_user_prompt(
            observations=observations_section(
                anchor.env_step, self.team_name,
                [self._member_block(m) for m in members],
                task_block=self._team_task_block(members),
                goal_instruction=self.goal_instruction,
                reserved_task_observation=getattr(self, "reserved_task_observation", ""),
                preface="The team was interrupted by a message.",
                house=self._house_line(members),
            ),
            current_messages=self._current_messages_block(list(messages or [])),
            conversation_history=conversation_history_section(
                self._messages_block("## 8. CONVERSATION HISTORY", exclude=messages)),
            action_history=action_history_section([self._action_history_block(m) for m in members]),
            closing=self._interrupt_closing(),
        )
        return [
            {"role": "system", "content": self._system_prompt(anchor)},
            {"role": "user", "content": user},
        ]



class ChainTeamBrain(TeamBrain):
    """Teams speak in order; each broadcasts what it committed to the later ones.

    The ordering mechanism is the predecessor becoming *ready*: whatever it was
    going to say it has said, so nothing more is coming and the world must not be
    held for it. Same rule the per-robot chain uses, lifted a level -- what is
    ordered now is teams, and each message carries a whole team's allocation
    rather than one robot's intention.
    """

    COOPERATION_MODE = "broadcast_chain"

    #: What this round's model call said to the teams behind, until `after_plan`
    #: or `after_interrupt` sends it. A class default so a brain that has not
    #: called yet still answers. Planning and interrupting never overlap for one
    #: team -- a team in I is not at its planning barrier -- so one slot serves.
    _pending_broadcast = ""

    def before_plan(self) -> None:
        self._await_speakers()

    # -- where in the chain this team stands ---------------------------------

    def _chain_position(self) -> tuple:
        """(teams ahead, teams behind), in the chain's own order.

        Not `wait_for` and `send_to`: `wait_for` is only the team immediately
        ahead -- the one whose message this team blocks on -- while *every*
        earlier team's broadcast reaches it and lands in section 7. Naming only
        the predecessor would leave a team told to fit around one plan while
        reading three.
        """
        order = list(getattr(self, "all_teams", None) or {})
        if self.team_name in order:
            index = order.index(self.team_name)
            return order[:index], order[index + 1:]
        return list(self.wait_for), list(self.send_to)

    def _cooperation_extra(self) -> str:
        """Name this team's own upstream and downstream (user, 2026-09-14).

        Section 4 is one text for every team in the mode, so it could only say
        "the teams ahead of you" -- and a team reading a broadcast had to work
        out from the ids whether the sender was ahead of it or behind, which is
        exactly what it cannot see. The names are known at wiring time; this
        puts them in the prompt.
        """
        ahead, behind = self._chain_position()
        lines = []
        if ahead:
            lines.append(f"UPSTREAM (plan before you): {', '.join(ahead)}.")
            lines.append("Their commitments are in section 7: fit your plan around them, "
                         "take what they left, never the leg or cargo they claimed.")
        else:
            lines.append("UPSTREAM: none -- you are the head of the chain.")
            lines.append("Nobody constrains you, and section 7 is empty every round; "
                         "choose the first leg and say so.")
        if behind:
            lines.append(f"DOWNSTREAM (plan after you, on what you say): {', '.join(behind)}.")
            lines.append("Your `broadcast` is all they get from you: name which robot of "
                         "yours holds or goes for which cargo, which leg you take, and "
                         "what you leave to them.")
        else:
            lines.append("DOWNSTREAM: none -- you are the tail.")
            lines.append("Nobody plans around you, so take what the teams ahead left.")
        return "\n".join(lines)

    # -- the response --------------------------------------------------------

    @property
    def plan_response_format(self):
        return LLMChainPlanResponse

    def _call_plan_model(self, prompt: List[Dict[str, str]]):
        return self.llm_client.generate(
            prompt, response_format=self.plan_response_format, temperature=self.temperature
        )

    def _plan_closing(self, members: List["LLMTeamAgent"]) -> str:
        return (super()._plan_closing(members)
                + " `broadcast`: what the teams behind you must know -- which robot of "
                  "yours holds or is going for the cargo, which leg you take this round, "
                  "what you leave to them, in the task's ids, one or two sentences.")

    def _on_plan_response(self, response: Any) -> None:
        """Keep the round's own words; `after_plan` sends them."""
        self._pending_broadcast = (getattr(response, "broadcast", "") or "").strip()

    def after_plan(self) -> None:
        if not self._pending_plans:
            return
        # The model's sentence when it wrote one, the mechanical summary only
        # as a fallback: a round that says nothing is worse than a round that
        # says what it is doing in ids.
        text = getattr(self, "_pending_broadcast", "")
        self._say(text or self._plan_summary(self._pending_plans), "broadcast_chain", interrupts=True)
        self._pending_broadcast = ""

    def before_interrupt_decision(self, messages: List[Dict]) -> List[Dict]:
        """Wait for the team ahead to relay, then decide knowing what it said.

        One message from one team interrupts *every* team behind it -- that is
        the broadcast -- so without this they all decide at once and only the
        first of them decides on anything new. Measured on
        broadcast_chain_agents12_..._064820: team_0 spoke at t=183.28, team_1
        and team_2 both went to I in the same instant, team_2 finished deciding
        at 191.58, and team_1 did not relay until 235.14. team_2's decision was
        44 s older than the information it was supposed to be ordered behind.

        Blocking here is safe in a way it is not on the planning path. Every
        team behind the sender is already in I, so none of them needs the world
        to advance: the team ahead has to reach its own barrier -- its members
        are stopped, so it is already there -- make one model call, and speak.
        A frozen env is exactly what all of them are doing anyway.

        Three exits, and the timeout is the one that should never fire: the
        relay arrives; or the team ahead turns out to have no open round, so no
        relay is coming (it was in R, which the broker does not interrupt); or
        the backstop.
        """
        relayed = self._await_relay()
        return list(messages) + relayed if relayed else messages

    def _interrupt_response_format(self):
        return LLMChainInterruptResponse

    def _call_interrupt_model(self, prompt: List[Dict[str, str]]):
        return self.llm_client.generate(
            prompt, response_format=self._interrupt_response_format(),
            temperature=self.temperature,
        )

    def _interrupt_closing(self) -> str:
        return (super()._interrupt_closing()
                + " `broadcast`: what this changed for your robots and what the teams "
                  "behind you must know, in the task's ids, one or two sentences -- "
                  "also when everyone resumed.")

    def _on_interrupt_response(self, response: Any) -> None:
        self._pending_broadcast = (getattr(response, "broadcast", "") or "").strip()

    def after_interrupt(self, decisions: Dict[str, Any]) -> None:
        """Relay onward, whatever this team decided.

        Exactly once per round and after the decision, so what is said is what
        the robots will actually do. Resume counts: a team that changed nothing
        still has to say so, or the team behind it waits out `before_interrupt_
        decision` for a message that was never coming -- which is the difference
        between an ordering and a stall.

        The words are the model's own, written in the decision call
        (`LLMChainInterruptResponse.broadcast`). `_current_allocation()`, every
        robot's current plan specification joined, is the fallback for a
        response that carried no sentence.
        """
        text = getattr(self, "_pending_broadcast", "")
        self._say(text or self._current_allocation(), "broadcast_chain", interrupts=True)
        self._pending_broadcast = ""

    # -- ordering ----------------------------------------------------------

    def _await_relay(self, timeout: float = CHAIN_RELAY_TIMEOUT) -> List[Dict]:
        """The upstream team's message for this round, waited for if need be."""
        if not self.wait_for or not self.has_open_interrupt_round():
            return []
        upstream = set(self.wait_for)
        # What "this round" means, without a clock. The obvious version -- take
        # messages stamped after the round opened -- cannot work: the broker
        # writes timestamps relative to the run's origin and `_interrupt_opened`
        # is an absolute `time.time()`, so the comparison is a few seconds
        # against 1.7e9 and is false forever. Fingerprinting what was already in
        # hand has no clock in it at all.
        self._collect_heard()
        already = {self._message_key(m) for m in self._heard
                   if m.get("sender") in upstream}
        deadline = time.monotonic() + timeout
        while True:
            self._collect_heard()
            relayed = [m for m in self._heard
                       if m.get("sender") in upstream
                       and self._message_key(m) not in already]
            if relayed:
                return relayed
            if not self._upstream_will_speak():
                if TEAM_VERBOSE:
                    print(f"  [chain {self.team_name}] {self.wait_for} has no open "
                          "round; deciding without a relay")
                return []
            if time.monotonic() >= deadline:
                print(f"  [{self.team_name}] waited {timeout:.0f}s for {self.wait_for} "
                      "to relay; deciding without it")
                return []
            time.sleep(0.05)

    def _upstream_will_speak(self) -> bool:
        """Is a team this one waits on inside an interrupt round of its own?

        The question readiness could not answer. A team ahead that is executing
        is `ready`, and the old ordering read that as "it has already said what
        it was going to say" -- true of the round it announced long ago, not of
        this one.
        """
        return any(
            peer is not None and peer.has_open_interrupt_round()
            for peer in (self.peers.get(name) for name in self.wait_for)
        )

    def _current_allocation(self) -> str:
        """One line per robot, from the plan it holds right now."""
        parts = []
        for name in self.member_ids:
            member = self.members.get(name)
            plan = getattr(member, "plan", None) if member is not None else None
            if plan is not None:
                parts.append(f"{name}: {plan.specification}")
        return f"[{self.team_name}] " + ("; ".join(parts) if parts else "no active plans")


class LeaderTeamBrain(TeamBrain):
    """Asks every follower team what it is doing, then plans for its own robots.

    The request interrupts the follower teams, which is the point: a follower
    that is mid-plan has to answer with where it actually is, not with what it
    intended several hundred ticks ago.
    """

    COOPERATION_MODE = "centralized_leader"
    _assignment_sent = False

    def before_plan(self) -> None:
        """Assign, hear the responses, then plan.

        The leader's first word of a round is the assignment itself -- which
        team takes which leg -- not a request for status (user, 2026-09-13).
        The followers answer the assignment; the leader plans its own robots
        knowing what they said. The wire type stays ``leader_broadcast``: the
        broker treats that name as interrupting, and the interrupt tests pin it.
        """
        self._say(self._compose_assignment(), "leader_broadcast", interrupts=True)
        # Read by followers at their own barrier (`_await_assignment`): False
        # from the end of one round to the send in the next, i.e. exactly while
        # an assignment is still to come.
        self._assignment_sent = True
        self._await_replies()

    def after_plan(self) -> None:
        self._assignment_sent = False

    def _compose_assignment(self) -> str:
        """The round's assignment for the follower teams, written by the model
        from what the leader can see: its own robots' observations, the team
        task, and the record of what was said before. Falls back to a plain
        instruction if the model fails, so the round still opens."""
        members = [self.members[n] for n in self.member_ids if n in self.members]
        followers = list(self.send_to) or ["the other teams"]
        fallback = (f"[{self.team_name}] Assignment: {', '.join(followers)} -- each team takes the "
                    "next uncompleted leg of the route nearest to its robots; reply with what you take.")
        if not members:
            return fallback
        anchor = members[0]
        new = list(self._heard)
        user = assemble_user_prompt(
            observations=observations_section(
                anchor.env_step, self.team_name,
                [self._member_block(m) for m in members],
                task_block=self._team_task_block(members),
                goal_instruction=self.goal_instruction,
                reserved_task_observation=getattr(self, "reserved_task_observation", ""),
                preface=(f"\nYou lead. Before planning your own robots, assign this round's work "
                         f"to the follower teams: {', '.join(followers)}."),
            ),
            current_messages=current_messages_section(self.COOPERATION_MODE, new, self._render_message),
            conversation_history=conversation_history_section(
                self._messages_block("## 8. CONVERSATION HISTORY", exclude=new)),
            action_history=action_history_section([self._action_history_block(m) for m in members]),
            closing=("Write the assignment as one message to all follower teams: for each team, "
                     "which route leg or object it should take this round and which robot kind should "
                     "do it (an arm, a carrier, a drone), using the ids above and the route's node "
                     "names. Say what you will do yourselves in one clause. Three to five sentences. "
                     "No preamble."),
        )
        prompt = [
            {"role": "system", "content": self._system_prompt(anchor)},
            {"role": "user", "content": user},
        ]
        if anchor._should_print_llm_io():
            anchor._print_llm_messages(f"Leader Assignment [{self.team_name}]", prompt)
        text = anchor._generate_text_from_messages(
            messages=prompt, fallback=fallback, label=f"Leader Assignment [{self.team_name}]")
        text = " ".join(str(text).split())
        return text if text.startswith(f"[{self.team_name}]") else f"[{self.team_name}] {text}"

    def _await_replies(self, timeout: float = LEADER_REPLY_TIMEOUT) -> None:
        """Block until every follower team has reported, then plan.

        This replaces `_await_speakers`, whose escape hatch -- give up as soon
        as the teams waited on merely *look* ready -- made "ask, then plan"
        into "ask, then plan without the answer". Measured before this: of
        seven requests the leader sent, three of its plans carried a reply.

        Blocking is safe here and was not before, because the reply no longer
        waits for the follower's next planning barrier: a follower answers from
        `on_interrupt`, while still in I, and a status report costs neither an
        env step nor an LLM call. Every state a follower team can be in can
        answer without the world advancing -- interrupted members answer from
        `on_interrupt`, a team already at its own barrier answers from its
        `before_plan` -- so nothing here is waiting on something that needs the
        world, which is the shape that deadlocks.
        """
        if not self.wait_for:
            return
        deadline = time.monotonic() + timeout
        while True:
            self._collect_heard()
            silent = [team for team in self.wait_for if not self._replied(team)]
            if not silent:
                return
            if time.monotonic() >= deadline:
                print(f"  [{self.team_name}] waited {timeout:.0f}s for {silent}; "
                      "planning without them")
                return
            time.sleep(0.05)

    def _replied(self, team: str) -> bool:
        return any(
            message.get("sender") == team
            and (message.get("metadata") or {}).get("type") == "follower_response"
            for message in self._heard
        )


class FollowerTeamBrain(TeamBrain):
    """Waits for the leader's assignment, answers it for its robots, then plans.

    The answer is written by the model (accept, or say what the team will do
    instead and why), grounded in the assembled status of its robots, which
    goes into the prompt and not onto the wire.
    """

    COOPERATION_MODE = "centralized_follower"

    def on_interrupt(self, messages: List[Dict]) -> None:
        """Answer the leader the moment its request lands.

        This is the path that matters: the request interrupts, so a follower is
        in I within microseconds of being asked, and answering from here means
        the leader gets its reply in the same instant rather than waiting for
        the follower's next planning barrier -- which was hundreds of env steps
        away, and which the leader was blocking the world for.

        Costs nothing that needs the world: the report is assembled from the
        team's own state, so no env step and no LLM call.
        """
        self._answer_leader()

    def before_plan(self) -> None:
        # Still needed, and not a duplicate: a follower team that was already at
        # its own planning barrier when the assignment arrived was in R, which
        # the broker does not interrupt, so `on_interrupt` never fired for it.
        # That is the case the run's first round is always in -- and there the
        # assignment may not have arrived yet, because the leader is still
        # writing it: wait for it while the leader is deciding.
        self._await_assignment()
        self._answer_leader()

    def _await_assignment(self, timeout: float = FOLLOWER_ASSIGNMENT_TIMEOUT) -> None:
        """Block until an unanswered assignment is in hand, while the leader is
        about to assign (`_assignment_still_coming`). Returns at once otherwise:
        nothing is coming."""
        leader = next((self.peers.get(t) for t in self.wait_for if self.peers.get(t) is not None), None)
        deadline = time.monotonic() + timeout
        while True:
            self._collect_heard()
            if self._pending_assignments():
                return
            if not self._assignment_still_coming(leader):
                return
            if time.monotonic() >= deadline:
                print(f"  [{self.team_name}] waited {timeout:.0f}s for {self.wait_for}'s assignment; "
                      "planning without it")
                return
            time.sleep(0.05)

    @staticmethod
    def _assignment_still_coming(leader) -> bool:
        """Is the leader about to assign, so that waiting is worth it?

        Yes when it has not sent this round's assignment and none of its robots
        is executing: a leader whose robots are all in R/W is at (or a thread
        switch away from) its own barrier, which opens with the assignment. At
        step 0 that is every team at once -- a version that tested a flag the
        leader sets only once its own members have all arrived lost that race,
        because a follower can reach its barrier first. A leader with a robot
        still executing is mid-round: waiting for it would freeze the world it
        needs to finish, so no.
        """
        if leader is None or getattr(leader, "_assignment_sent", False):
            return False
        members = list(getattr(leader, "members", {}).values())
        if not members:
            return False
        return not any(getattr(m, "state", None) is AgentState.X for m in members)

    def _pending_assignments(self) -> List[Dict]:
        return [
            message for message in self._heard
            if (message.get("metadata") or {}).get("type") == "leader_broadcast"
            and (message.get("sender"), message.get("timestamp")) not in self._answered
        ]

    def _answer_leader(self) -> None:
        """Report once per request, from whichever path gets there first."""
        pending = self._pending_assignments()
        if not pending:
            return
        for message in pending:
            self._answered.add((message.get("sender"), message.get("timestamp")))
        # Not an interrupt in the other direction: the leader asked for this and
        # is blocked in its own planning barrier waiting for it, so interrupting
        # it is incoherent -- and could not be done uniformly anyway, because
        # the members inside that barrier are in R and R is not interruptible,
        # so the team would split.
        self._say(
            self._compose_response(), "follower_response",
            interrupts=False, expected_reply=True,
        )

    def _was_asked(self) -> bool:
        """Did a leader request actually arrive this round?"""
        return any(
            (message.get("metadata") or {}).get("type") == "leader_broadcast"
            for message in self._heard
        )

    def _compose_response(self) -> str:
        """The reply the leader gets: the team's answer to the assignment,
        written by the model and grounded in facts.

        The leader's message is an assignment (user, 2026-09-13), so the reply
        is a response to it -- we take it / we cannot and here is what we do
        instead -- not a status report. `_status_report` goes into the prompt
        so the model cannot invent a position or a holding; on any failure it is
        what gets sent, so this degrades to a report rather than to silence.
        One call per leader round, which the leader blocks for.
        """
        members = [self.members[n] for n in self.member_ids if n in self.members]
        if not members:
            return self._status_report()
        anchor = members[0]
        assignment = next(
            (str(m.get("content", "")) for m in reversed(self._heard)
             if (m.get("metadata") or {}).get("type") == "leader_broadcast"),
            "Take the next uncompleted leg of the route nearest to you.",
        )
        prompt = [
            {"role": "system", "content": self._system_prompt(anchor)},
            {"role": "user", "content": "\n".join([
                f"=== STEP {anchor.env_step} ===",
                f"\nYour leader assigned TEAM {self.team_name}:",
                f"  {assignment}",
                "\nWhat your robots are actually doing right now:",
                f"  {self._status_report()}",
                "\nReply in two or three sentences, as this team, to the leader: accept the "
                "assignment and say which robot does which part, or say what you will do instead "
                "and why (a robot that cannot lift the cargo, a leg already done, a target another "
                "team holds). Use only the object ids above. No preamble.",
            ])},
        ]
        if anchor._should_print_llm_io():
            anchor._print_llm_messages(f"Team Response [{self.team_name}]", prompt)
        text = anchor._generate_text_from_messages(
            messages=prompt,
            fallback=self._status_report(),
            label=f"Team Response [{self.team_name}]",
        )
        # One model appended a ```yaml block restating its reasoning; the
        # sentences before the fence are the reply.
        return " ".join(str(text).split("```")[0].split())

    def _status_report(self) -> str:
        """Answer the four things the leader asks, for each robot.

        It used to report only ``plan.specification``, which answered none of
        them and was ``wait_for_team(...)`` for most robots most of the time --
        so the leader allocated work having been told, in effect, "we were
        idling". The exchange looked correct in the log and carried nothing.

        Still assembled rather than generated: every field below is already in
        the member's own observation, so this costs no LLM call, which is what
        lets the leader block for the reply.
        """
        parts = []
        for name in self.member_ids:
            member = self.members.get(name)
            if member is None:
                continue
            parts.append(f"{name} {self._member_status(member)}")
        return f"[{self.team_name}] " + "; ".join(parts)

    def _member_status(self, member: "LLMTeamAgent") -> str:
        room, holding = _read_view_header(member.symbolic_view)
        plan = member.plan
        spec = str(plan.specification) if plan is not None else "nothing"
        if spec.startswith("wait_for_team"):
            spec = "idle, waiting for its team"
        bits = []
        if room:
            bits.append(f"in {room}")
        bits.append(f"holding {holding or 'nothing'}")
        target = _first_useful_target(member.symbolic_view)
        if target:
            bits.append(f"nearest target {target}")
        bits.append(f"plan {spec}")
        return ", ".join(bits)


#: Which brain each topology uses for which team. `individual` is the plain
#: TeamBrain: teams never address each other, so the only coordination in the
#: run is the one that happens *inside* each team.
TEAM_BRAIN_ROLES = {
    "individual": "no team talks to any other",
    "broadcast_chain": "teams speak in order, each broadcasting to the later ones",
    "centralized": "the first team leads; the rest report to it",
    "tag": "every team reads and writes one shared task graph, and may notify others to read it",
    "board": "the same, with an unstructured shared message board in place of the graph -- DIG-TAG's ablation",
    "scripted": "no model at all: plans are read from a file, and a robot without one holds position",
}


def reset_notify_budgets(agents: Dict[str, Any]) -> None:
    """After an environment step every team may notify again. A no-op for
    brains without a budget (the modes that never notify)."""
    for brain in {id(a.brain): a.brain for a in agents.values() if hasattr(a, "brain")}.values():
        reset = getattr(brain, "reset_notify_budget", None)
        if reset is not None:
            reset()


def _read_view_header(view: Optional[str]) -> Tuple[str, str]:
    """``(room, holding)`` out of an observation's first two lines.

    The observation opens with
    ``Step 0/2500 | you are agent_4 in empty_room_0 (a empty room)`` and
    ``Holding: nothing``, so this is a header read, not a parse of the body.
    """
    room = holding = ""
    for line in (view or "").split("\n")[:4]:
        if not room:
            match = re.search(r"\byou are \S+ in (\S+)", line)
            if match:
                room = match.group(1)
        if line.startswith("Holding:"):
            value = line.split(":", 1)[1].strip()
            holding = "" if value in ("nothing", "") else value
    return room, holding


def _first_useful_target(view: Optional[str]) -> str:
    """One target worth naming: something in range, else the closest thing.

    A room-listing line reads
    ``  - apple.n.01_2  -> unreachable, navigate_to  [11.6 m away]`` when out of
    range and carries no distance when the robot can already act on it, which is
    exactly the distinction the leader wants reported. Only objects have the
    ``->``: target_hints skips robots, so teammates cannot be reported as
    targets.
    """
    nearest, nearest_distance = "", float("inf")
    for line in (view or "").split("\n"):
        stripped = line.strip()
        if not stripped.startswith("- ") or "->" not in stripped:
            continue
        name = stripped[2:].split("->", 1)[0].strip().split("  ")[0]
        distance = re.search(r"\[([\d.]+) m away\]", stripped)
        if distance is None:
            return f"{name} (in range)"
        value = float(distance.group(1))
        if value < nearest_distance:
            nearest, nearest_distance = name, value
    return f"{nearest} ({nearest_distance:.0f} m away)" if nearest else ""


def _as_plan_response(entry: Any):
    """A ``TeamAgentPlan`` viewed as the single-agent response parse expects."""
    from coop2.cognitive.agent.llm_client import LLMPlanResponse  # noqa: PLC0415

    return LLMPlanResponse(task=entry.task, actions=entry.actions, reasoning=entry.reasoning)


class LLMTeamAgent(BaseLLMAgent):
    """One robot, whose thinking is done by the team's shared brain."""

    def __init__(
        self,
        agent_id: str,
        brain: TeamBrain,
        temperature: float = 0.7,
        verbose: bool = True,
        goal_instruction: str = "",
    ):
        super().__init__(agent_id, brain.llm_client, temperature=temperature, verbose=verbose)
        self.brain = brain
        self.wait_for = []
        self.send_to = []
        self.goal_instruction = goal_instruction
        self.team_agent_ids: List[str] = list(brain.member_ids)
        brain.register(self)

    # -- the two FSM hooks -------------------------------------------------

    def handle_reasoning(self):
        """Take the team's plan, or hold position until the team is complete."""
        # The inbox is *not* cleared here. It belongs to the team: the brain
        # drains every member's buffer into one shared record when the barrier
        # closes, and folds it into the prompt. Clearing here discarded the
        # leader's planning request before the team could read it -- the
        # follower then had nothing to answer and waited out the full timeout
        # for a message that had already arrived and been thrown away.
        plan = self.brain.request_plan(self.agent_id)
        if plan is None:
            plan = self.brain.claim_pending_plan(self.agent_id)
        if plan is None:
            # Teammates are still executing. Staying not-ready here would freeze
            # the world and they would never finish -- see the module docstring.
            # Nothing below may touch the brain: it has already been told this
            # member is holding, and it may be spinning on this member reaching
            # W, so taking its lock here would have both wait on the other.
            if self.verbose:
                print(f"  [{self.agent_id}] holding for the team "
                  f"({TEAM_HOLD_ACTIONS} x {TEAM_HOLD_TICKS} ticks, ended by the recall)")
            plan = self.build_hold_plan()
        self.plan = plan
        return self.plan

    def handle_interrupt(self):
        """Resume or replan, as the brain decided for this robot."""
        messages = self.get_messages(clear_buffer=True)
        decision = self.brain.request_interrupt_decision(self.agent_id, messages)
        if decision is None:
            # Teammates have not reached the barrier yet. Resume for now; the
            # message stays handled because whoever closes the barrier answers
            # for the whole team.
            return
        choice, new_plan = decision
        if choice == InterruptDecision.REPLAN and new_plan is not None:
            self.plan = new_plan

    # -- helpers -----------------------------------------------------------

    def build_hold_plan(self) -> SymbolicPlan:
        """A one-action ``wait`` plan, so idling costs ticks like anything else.

        Built directly rather than through ``parse_plan_response``. That helper
        ends with ``_ensure_task_terminal_action``, which appends an action to
        match the plan's TaskSpecification -- correct for a real goal, wrong
        here: a hold has no goal, and expressing it as ``holding(<self>)`` made
        the helper append ``grasp(<self>)``, i.e. a robot planning to pick
        *itself* up. Seen in a real run before this was fixed.
        """
        return SymbolicPlan(
            specification=f"wait_for_team({self.brain.team_name})",
            # Many, not one: a plan that completes sends its agent back to R,
            # and a holding member has nothing to reason about. See
            # TEAM_HOLD_ACTIONS.
            actions=[
                SymbolicAction(action_type="wait", args={"ticks": TEAM_HOLD_TICKS})
                for _ in range(TEAM_HOLD_ACTIONS)
            ],
            plan_id=self.plan_count + 1,
            agent_id=self.agent_id,
            created_at_step=self.env_step,
        )


def create_llm_team_topology(
    llm_client: LLMClient,
    teams: Dict[str, List[str]],
    topology: str = "individual",
    temperature: float = 0.7,
    verbose: bool = True,
    goal_instruction: str = "",
    notify_budget: int = 1,
    script: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, LLMTeamAgent]:
    """One brain per team, one agent per robot, wired for @topology.

    The topology decides how teams address **each other**; inside every team it
    is always one LLM for all its robots. With teams of one this reproduces the
    old per-robot behaviour of the same topology, which is why there is no
    separate code path for it.

    Args:
        teams: ``{team name: [agent ids]}``, straight from the layout. Order
            matters: it is the chain's speaking order, and the first team leads
            under `centralized`.
        notify_budget: how many times a team may notify others between two
            environment steps, in the modes that have the tool (`tag`, and the
            board with ``NOTIFY_TOOL_ENABLED``).
        script: `scripted` only -- ``{robot id: [plan, ...]}`` for every team,
            from ``scripted_team.load_script``. Each brain reads its own
            members out of it; a robot it does not name holds position.
    """
    if topology not in TEAM_BRAIN_ROLES:
        raise ValueError(f"unknown topology {topology!r}; have {sorted(TEAM_BRAIN_ROLES)}")

    names = list(teams)
    brains: Dict[str, TeamBrain] = {}
    # One shared space for the whole run, whichever kind: the mode *is* the
    # sharing, so it is built here, where every brain is, and not by any one of
    # them. `llm_tag` is the one runtime file that knows dig_tag (its own
    # rule), and `llm_board` is its ablation, so both are imported here.
    tag_graph = board = None
    if topology == "tag":
        from coop2.comm_topology.llm_tag import TagTeamBrain, new_shared_tag  # noqa: PLC0415

        tag_graph = new_shared_tag()
    elif topology == "board":
        from coop2.comm_topology.llm_board import BoardTeamBrain, new_shared_board  # noqa: PLC0415

        board = new_shared_board()
    for index, team_name in enumerate(names):
        extra: Dict[str, Any] = {}
        if topology == "broadcast_chain":
            factory = ChainTeamBrain
        elif topology == "centralized":
            factory = LeaderTeamBrain if index == 0 else FollowerTeamBrain
        elif topology == "board":
            factory = BoardTeamBrain
            extra["board"] = board
            extra["notify_budget"] = notify_budget
        elif topology == "tag":
            factory = TagTeamBrain
            extra["tag"] = tag_graph
            extra["notify_budget"] = notify_budget
        elif topology == "scripted":
            from coop2.comm_topology.scripted_team import ScriptedTeamBrain  # noqa: PLC0415

            factory = ScriptedTeamBrain
            extra["script"] = script or {}
        else:
            factory = TeamBrain
        brains[team_name] = factory(
            team_name=team_name,
            member_ids=list(teams[team_name]),
            llm_client=llm_client,
            temperature=temperature,
            verbose=verbose,
            goal_instruction=goal_instruction,
            **extra,
        )

    agents: Dict[str, LLMTeamAgent] = {}
    for team_name in names:
        for agent_id in teams[team_name]:
            agents[agent_id] = LLMTeamAgent(
                agent_id,
                brains[team_name],
                temperature=temperature,
                verbose=verbose,
                goal_instruction=goal_instruction,
            )

    # Wiring, in team names on both sides: teams are what talk to each other
    # here. Delivery still reaches every robot of an addressed team -- an
    # interrupt has to stop all of it -- but that expansion belongs to the
    # broker, not to the address.
    if topology == "broadcast_chain":
        for index, team_name in enumerate(names):
            brain = brains[team_name]
            brain.wait_for = [names[index - 1]] if index > 0 else []
            brain.send_to = list(names[index + 1:])
    elif topology == "centralized":
        leader, followers = names[0], names[1:]
        brains[leader].wait_for = list(followers)
        brains[leader].send_to = list(followers)
        for name in followers:
            brains[name].wait_for = [leader]
            brains[name].send_to = [leader]

    for brain in brains.values():
        brain.all_agents = agents
        brain.all_teams = {name: list(teams[name]) for name in names}
        # Its own entry included: harmless, and leaving it out would make the
        # map mean something different depending on who is reading it.
        brain.peers = dict(brains)
    if board is not None:
        # The board's notify deliverer: the request is delivered by the brain
        # that made it, which is the one whose budget it spends.
        board.deliverer = lambda record: brains[record.sender]._deliver_notify(record)
    for agent in agents.values():
        # Mirrors of the brain's wiring, in team names. Nothing in the team path
        # reads them -- the brain does all the waiting and sending -- but they
        # are what an inspector reaches for first, and holding a stale agent-id
        # copy of a team-level wiring would be worse than holding none.
        agent.wait_for = list(agent.brain.wait_for)
        agent.send_to = list(agent.brain.send_to)
    return agents
