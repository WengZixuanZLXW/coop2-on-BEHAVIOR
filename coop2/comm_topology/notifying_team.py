"""The round shared by the task-graph mode and the message-board ablation: a
team that acts on a shared space and notifies other teams within a budget.

Ported from HappyEureka/dig-tag-icra `ma_crafter/comm_topology/notify.py`, which
is where their two modes meet. Keeping that seam is the whole point of the
ablation: `tag` and `board` have to differ **only** in the shared space, and two
copies of a round drift the moment one is fixed and the other is not. This file
was extracted from `llm_tag.py` when the board was ported (user, 2026-09-15) --
the board had been written as a copy of it, which is the drift arriving before
the second mode had even run.

Whenever a team plans or is interrupted it observes the shared space in front
of its robots' observations and returns one structured output: the space's part
(what to change or say), `notify` (teams to wake), and the plans. The space's
part takes effect first; then each notification, a message that interrupts its
recipients so they observe and answer the same way; then the plans go to the
robots. A team may notify at most ``notify_budget`` times between two
environment steps -- the runner resets the budgets after every step -- which
bounds the interrupt cascade. A notification that arrives while a team is
inside its planning call is answered before its plans go out.

A subclass names the space. Six seams, upstream's:

    _observe_space()          what the space looks like right now
    _space_section(obs)       how the prompt shows it (section 6's slot)
    _observed(obs)            ids of what it showed, for the round's record
    _record_fields()          the record's space-specific keys, empty
    _apply(response, record)  the space's part of the answer lands
    _summary(record)          one line for the log

plus the class attributes ``SPACE_NAME``, ``MANUAL`` (section 5),
``PLAN_RESPONSE`` / ``INTERRUPT_RESPONSE`` (the two schemas) and
``NOTIFY_TEXT`` (what a notification says). Nothing else about the mode --
the team barrier, the broker, the plan machinery, the prompt's nine sections
-- is touched by either subclass.
"""

from __future__ import annotations

import itertools
import threading
from typing import Any, Dict, List

from coop2.cognitive.agent.llm_client import InterruptDecision
from coop2.comm_topology.llm_team import TeamBrain

__all__ = [
    "DEFAULT_NOTIFY_BUDGET",
    "NOTIFY_MESSAGE_TYPE",
    "NotifyingTeamBrain",
    "rounds_of",
]

#: The message type a notification travels under. The broker interrupts on
#: ``interrupts_execution``; the type is what the record and the prompt
#: heading key on. Shared by both modes, so a notification reads the same.
NOTIFY_MESSAGE_TYPE = "notify"

#: Notifications a team may send between two environment steps (DIG-TAG's
#: default). The runner resets every team's budget after each step.
DEFAULT_NOTIFY_BUDGET = 1

#: One sequence across every team and both modes, so a run's records interleave
#: in the order the rounds actually happened.
_round_sequence = itertools.count()
_round_sequence_lock = threading.Lock()


class NotifyingTeamBrain(TeamBrain):
    """One team on a shared space it may act on and notify others about.

    Nothing here waits for another team: ``wait_for`` and ``send_to`` stay
    empty, as in `individual`. What a team may do is act on the space and
    notify, in every planning and every interrupt call.
    """

    #: What the round is about, in log lines ("task graph", "board").
    SPACE_NAME = "shared space"
    #: Section 5: how to use this space, in the model's terms.
    MANUAL = ""
    #: What a notification says. The recipient reads the space itself, so the
    #: message only has to say that it should.
    NOTIFY_TEXT = "read the shared space"
    #: The two structured outputs: the team plan / the interrupt decisions,
    #: each plus the space's part and `notify`.
    PLAN_RESPONSE: type
    INTERRUPT_RESPONSE: type

    def __init__(self, *args, notify_budget: int = DEFAULT_NOTIFY_BUDGET, **kwargs):
        super().__init__(*args, **kwargs)
        self.notify_budget = int(notify_budget)
        self.notify_left = self.notify_budget
        self.reserved_system_prompt = self.MANUAL
        #: One record per output of this team (`TeamBrain.rounds` is the counter): what it saw, did, sent, dropped.
        self.round_records: List[Dict[str, Any]] = []
        # The ids the prompt being built shows, for the round's record.
        self._observed_now: List[str] = []
        # Keys of the messages the planning prompt quoted, so a notification
        # that arrived during the call can be told from one it already saw.
        self._quoted_in_plan: set = set()
        self._late_round = False

    # -- the seams a mode fills --------------------------------------------

    def _observe_space(self) -> Any:
        raise NotImplementedError

    def _space_section(self, observation: Any) -> str:
        raise NotImplementedError

    def _observed(self, observation: Any) -> List[str]:
        raise NotImplementedError

    def _record_fields(self) -> Dict[str, Any]:
        return {}

    def _apply(self, response: Any, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _summary(self, record: Dict[str, Any]) -> str:
        return ""

    # -- the budget --------------------------------------------------------

    def reset_notify_budget(self) -> None:
        """Called by the runner after every environment step."""
        self.notify_left = self.notify_budget

    # -- the space in the prompt (section 6's reserved slot) ---------------

    def _team_names(self) -> List[str]:
        return list(self.all_teams) if self.all_teams else [self.team_name]

    def _refresh_task_observation(self) -> None:
        observation = self._observe_space()
        self._observed_now = self._observed(observation)
        self.reserved_task_observation = self._space_section(observation)

    def _build_team_prompt(self, members):
        self._refresh_task_observation()
        self._quoted_in_plan = {self._message_key(m) for m in self._heard}
        return super()._build_team_prompt(members)

    def _build_interrupt_prompt(self, members, messages):
        self._refresh_task_observation()
        return super()._build_interrupt_prompt(members, messages)

    # -- the calls ---------------------------------------------------------

    def _call_plan_model(self, prompt):
        return self.llm_client.generate(prompt, response_format=self.PLAN_RESPONSE,
                                        temperature=self.temperature)

    def _call_interrupt_model(self, prompt):
        return self.llm_client.generate(prompt, response_format=self.INTERRUPT_RESPONSE,
                                        temperature=self.temperature)

    def _on_plan_response(self, response: Any) -> None:
        self._round(response, stage="planning")

    def _on_interrupt_response(self, response: Any) -> None:
        self._round(response, stage="late_notify" if self._late_round else "interrupted")

    # -- the round: the space's part lands, then notifications go out -------

    def _round(self, response: Any, stage: str) -> None:
        with _round_sequence_lock:
            sequence = next(_round_sequence)
        record: Dict[str, Any] = {
            "seq": sequence,
            "team": self.team_name,
            "env_step": self._env_step(),
            "stage": stage,
            "observed": list(self._observed_now),
            "reasoning": getattr(response, "reasoning", ""),
            **self._record_fields(),
            "notify": {"requested": [], "sent": [], "dropped": False},
        }
        self.round_records.append(record)
        self._apply(response, record)
        self._notify(getattr(response, "notify", None) or [], record)
        if self.verbose:
            print(f"  [{self.team_name}] {self.SPACE_NAME} ({stage}): {self._summary(record)}"
                  + f", notify {record['notify']['sent'] or '-'}"
                  + (" (dropped: budget spent)" if record["notify"]["dropped"] else ""))

    def _notify(self, requested: List[str], record: Dict[str, Any]) -> None:
        """Wake the named teams, within the budget: one message that interrupts
        each of them; over the budget it is dropped and recorded as dropped."""
        known = set(self._team_names())
        teams = [t for t in dict.fromkeys(requested) if t in known and t != self.team_name]
        record["notify"]["requested"] = teams
        if not teams:
            return
        if self.notify_left <= 0:
            record["notify"]["dropped"] = True
            return
        self.notify_left -= 1
        self._say(f"notification from {self.team_name}: {self.NOTIFY_TEXT}",
                  NOTIFY_MESSAGE_TYPE, interrupts=True, recipients=teams)
        record["notify"]["sent"] = teams

    # -- a notification during the planning call is answered before ready ---

    def after_plan(self) -> None:
        """DIG-TAG's ``_rounds`` loop: notifications that arrived while this
        team was inside its planning call are answered now, before the plans
        go out, rather than at its next barrier.

        The broker does not interrupt a team in R, so such a message only
        landed in the members' inboxes; drained here, it is decided on as an
        interrupt -- the same round, resume-or-replan per robot plus the
        space's part -- and a replanned robot takes the new plan.
        """
        self._collect_heard()
        late = [m for m in self._heard
                if self._message_key(m) not in self._quoted_in_plan
                and (m.get("metadata") or {}).get("type") == NOTIFY_MESSAGE_TYPE]
        if not late:
            return
        if self.verbose:
            print(f"  [{self.team_name}] {len(late)} notification(s) arrived during planning; "
                  "deciding on them before the plans go out")
        self._late_round = True
        try:
            decisions = self._decide_interrupts(late)
        finally:
            self._late_round = False
        for name, (choice, plan) in decisions.items():
            if choice == InterruptDecision.REPLAN and plan is not None:
                self._pending_plans[name] = plan


def rounds_of(agents: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every team's rounds, in the order they happened."""
    brains = {id(a.brain): a.brain for a in agents.values()
              if isinstance(getattr(a, "brain", None), NotifyingTeamBrain)}
    records = [record for brain in brains.values() for record in brain.round_records]
    return sorted(records, key=lambda record: record["seq"])
