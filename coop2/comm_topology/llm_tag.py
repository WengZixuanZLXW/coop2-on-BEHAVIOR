"""The task-graph mode (`tag`): DIG-TAG's Task Activity Graph as the space the
teams coordinate through.

The graph is the reference implementation from HappyEureka/dig-tag-icra,
commit e39c56e ("Letter the T band's marks"; the graph itself is 274882b's,
the renderer letters its marks so labels no longer overlap), vendored unmodified
into ``coop2/dig_tag/`` with its golden tests in ``coop2/dig_tag_tests/``.
This is the one runtime file that imports it -- their rule, kept: the graph
is LLM- and environment-agnostic and nothing else here should learn its
types. ``run_individual`` asks this module for the outputs to save.

**One TAG agent is one team brain** (user, 2026-09-13). DIG-TAG has no team
layer: one LLM per agent observes the graph and answers with actions, agents
to notify, and one plan. Here the unit is the team, so the brain observes
the graph on behalf of all its robots, its ``tag_actions`` are issued under
the team's name, ``notify`` names teams, and the plan part is the team plan
-- one per robot -- exactly as in every other mode. The round is theirs
otherwise, in this order: the actions land on the shared graph; then the
named teams are notified, which interrupts them within a per-step budget;
then the plans go to the robots. A notified team runs the same round from
its interrupt barrier, its answer being resume-or-replan per robot plus the
same two fields. A notification that arrives while this team is inside its
own planning call is answered before the plans go out (DIG-TAG's ``_rounds``
loop): the team decides on it as an interrupt, and a replanned robot takes
the new plan instead.

The prompt is ours, in the fixed order of ``coop2/PROMPTS.md``: the manual is
section 5, the graph fills the reserved slot in section 6 (above the robots'
observations, as O_T precedes O_E in DIG-TAG), the mode's rules are section
4, and a notification is section 7. The graph starts empty, as in the paper;
the team's task is stated in section 6 as for every mode.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from typing import Any, Dict, List, Optional

from coop2.cognitive.agent.llm_client import (
    InterruptDecision,
    LLMTagInterruptResponse,
    LLMTagPlanResponse,
    TagToolCall,
)
from coop2.comm_topology.llm_team import TeamBrain
from coop2.dig_tag.tag import TAG_TOOLS, TAGParallelInterface, observation_with_history, observed

__all__ = [
    "DEFAULT_NOTIFY_BUDGET",
    "HISTORY_SHOWN",
    "NOTIFY_MESSAGE_TYPE",
    "TAG_MANUAL",
    "TagTeamBrain",
    "format_tag_observation",
    "new_shared_tag",
    "save_tag_outputs",
    "shared_tag",
    "tag_call_args",
    "tag_rounds_of",
]

#: The message type a notification travels under. The broker interrupts on
#: ``interrupts_execution``; the type is what the record and the prompt
#: heading key on.
NOTIFY_MESSAGE_TYPE = "notify"

#: How many of a task's history steps the prompt shows (DIG-TAG's value).
HISTORY_SHOWN = 8

#: Notifications a team may send between two environment steps (DIG-TAG's
#: default). The runner resets every team's budget after each step.
DEFAULT_NOTIFY_BUDGET = 1

#: Inches per call in the drawn graph (DIG-TAG's run_condition value).
TAG_FIGURE_PITCH = 0.2

#: Section 5. DIG-TAG's TAG_ROLE, said to a team controller rather than to a
#: robot: the vocabulary and the three-part answer are unchanged.
TAG_MANUAL = """A task has a persistent identity (k1, k2, ...) and a current version (q1,
q2, ...) with a goal, a rule for judging it, and a reported state; the graph
shows each open task's current version, history (which team did what),
relations, and evidence.
tag_actions, applied in order:
- open(goal, rule, state): a new task, fresh identity.
- edit(task, goal, rule): a new version with a new goal or rule.
- update(task, state): a new version with a new reported state.
- split(task, parts): two or more parts, each with goal, rule, state; identity
  null for a fresh one.
- join(tasks, goal, rule, state): versions of distinct tasks into one.
- close(identity): no further work on that task.
- attach(task, payload): evidence or a note on an exact version.
Open what nobody is doing, update the state of what your robots do, attach
what you learned, close what is done, in the task's own ids (cargo, support,
room). Only a task's current version can be continued; acting on an older one
starts a new task. A rejected action is reported back and the others still
apply."""

_round_sequence = itertools.count()
_round_sequence_lock = threading.Lock()


def new_shared_tag() -> TAGParallelInterface:
    """The team's graph, empty at the start of the episode. Every tool,
    observe included, hands back the observation with histories, relations
    and evidence -- DIG-TAG's return policy for its task-graph team."""
    return TAGParallelInterface(returns={tool: observation_with_history for tool in TAG_TOOLS})


def tag_call_args(call: TagToolCall, issuer: str) -> Dict[str, Any]:
    """The arguments the graph's ``step`` takes for this call (DIG-TAG's)."""
    spec = {"goal": call.goal, "rule": call.rule}
    if call.tool == "open":
        return {"spec": spec, "state": call.state}
    if call.tool == "edit":
        return {"task": call.task, "spec": spec}
    if call.tool == "update":
        return {"task": call.task, "state": call.state}
    if call.tool == "split":
        return {"task": call.task, "parts": [
            {"identity": part.identity, "spec": {"goal": part.goal, "rule": part.rule}, "state": part.state}
            for part in call.parts or []
        ]}
    if call.tool == "join":
        return {"tasks": call.tasks or [], "identity": call.identity, "spec": spec, "state": call.state}
    if call.tool == "close":
        return {"identity": call.identity}
    return {"target": call.task, "payload": call.payload, "source": issuer}


def format_tag_observation(tag_observation: Dict[str, Any], team_names: List[str], budget_left: int) -> str:
    """The graph as the prompt shows it (DIG-TAG's ``format_tag_observation``,
    the issuers being teams): each open task's current version, its history,
    its relations, its evidence; then the notify budget."""
    versions = observed(tag_observation)
    histories = tag_observation.get("histories", {})
    relations = tag_observation.get("relations", {})
    evidence = tag_observation.get("evidence", {})
    lines = [f"Open tasks (graph shared by {', '.join(team_names)}):"]
    if versions:
        for version in versions:
            lines.append(
                f"  {version.id} [task {version.identity}] goal: {version.spec.goal}; "
                f"rule: {version.spec.rule}; state: {version.state}"
            )
            steps = histories.get(version.identity, [])
            if steps:
                shown = steps[-HISTORY_SHOWN:]
                told = "; ".join(
                    f"{s['tool']} by {s['issuer']} -> {s['version']}" if s["issuer"] else f"initial {s['version']}"
                    for s in shown
                )
                earlier = len(steps) - len(shown)
                lines.append(f"    history: {told}" + (f" (and {earlier} earlier)" if earlier else ""))
            relation = relations.get(version.identity) or {}
            origin = relation.get("origin")
            if origin:
                came_from = ", ".join(f"{f['version']} of {f['identity']}" for f in origin["from"])
                lines.append(f"    from: {origin['tool']} of {came_from} by {origin['issuer']}")
            for successor in relation.get("successors", []):
                lines.append(f"    into: {successor['tool']} of {successor['from']} by {successor['issuer']} "
                             f"-> {', '.join(successor['identities'])}")
            for phi in evidence.get(version.id, []):
                lines.append(f"    evidence {phi['id']} by {phi['source']}: {phi['payload']}")
    else:
        lines.append("  (none yet)")
    idle = [identity for identity in tag_observation.get("open", [])
            if all(v.identity != identity for v in versions)]
    if idle:
        lines.append(f"Open tasks without a current version: {', '.join(idle)}")
    lines.append(f"Notify budget left until the next environment step: {budget_left}")
    return "\n".join(lines)


class TagTeamBrain(TeamBrain):
    """One team on the shared task graph -- `tag`.

    Nothing here waits for another team: ``wait_for`` and ``send_to`` stay
    empty, as in `individual`. What a team may do is act on the graph and
    notify, in every planning and every interrupt call; see the module
    docstring for the order.
    """

    COOPERATION_MODE = "tag"

    def __init__(self, *args, tag: Optional[TAGParallelInterface] = None,
                 notify_budget: int = DEFAULT_NOTIFY_BUDGET, **kwargs):
        super().__init__(*args, **kwargs)
        #: Shared with every other brain in the run by the factory; a brain
        #: built alone gets a graph of its own so it still works.
        self.tag = tag if tag is not None else new_shared_tag()
        self.notify_budget = int(notify_budget)
        self.notify_left = self.notify_budget
        self.reserved_system_prompt = TAG_MANUAL
        #: One record per output of this team: what it saw, did, sent, dropped.
        self.tag_rounds: List[Dict[str, Any]] = []
        # The version ids the prompt being built shows, for the round's record.
        self._observed_now: List[str] = []
        # Keys of the messages the planning prompt quoted, so a notification
        # that arrived during the call can be told from one it already saw.
        self._quoted_in_plan: set = set()
        self._late_round = False

    # -- the budget --------------------------------------------------------

    def reset_notify_budget(self) -> None:
        """Called by the runner after every environment step."""
        self.notify_left = self.notify_budget

    # -- the graph in the prompt (section 6's reserved slot) ---------------

    def _team_names(self) -> List[str]:
        return list(self.all_teams) if self.all_teams else [self.team_name]

    def _refresh_task_observation(self) -> None:
        observation = self.tag.observe()
        self._observed_now = [version.id for version in observed(observation)]
        self.reserved_task_observation = format_tag_observation(
            observation, self._team_names(), self.notify_left
        )

    def _build_team_prompt(self, members):
        self._refresh_task_observation()
        self._quoted_in_plan = {self._message_key(m) for m in self._heard}
        return super()._build_team_prompt(members)

    def _build_interrupt_prompt(self, members, messages):
        self._refresh_task_observation()
        return super()._build_interrupt_prompt(members, messages)

    def _plan_closing(self, members) -> str:
        return (super()._plan_closing(members)
                + " `tag_actions`: changes the shared graph needs (empty if none); "
                  "`notify`: teams that must read it now (usually empty).")

    def _interrupt_closing(self) -> str:
        return (super()._interrupt_closing()
                + " Also `tag_actions` (empty if none) and `notify` (usually empty).")

    # -- the calls ---------------------------------------------------------

    def _call_plan_model(self, prompt):
        return self.llm_client.generate(prompt, response_format=LLMTagPlanResponse, temperature=self.temperature)

    def _call_interrupt_model(self, prompt):
        return self.llm_client.generate(prompt, response_format=LLMTagInterruptResponse, temperature=self.temperature)

    def _on_plan_response(self, response: Any) -> None:
        self._round(response, stage="planning")

    def _on_interrupt_response(self, response: Any) -> None:
        self._round(response, stage="late_notify" if self._late_round else "interrupted")

    # -- the round: actions land, then notifications go out -----------------

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
            "tag_actions": [],
            "notify": {"requested": [], "sent": [], "dropped": False},
        }
        self.tag_rounds.append(record)
        self._apply(getattr(response, "tag_actions", None) or [], record)
        self._notify(getattr(response, "notify", None) or [], record)
        if self.verbose:
            applied = sum(1 for a in record["tag_actions"] if a["result"] == "applied")
            rejected = len(record["tag_actions"]) - applied
            print(f"  [{self.team_name}] task graph ({stage}): {applied} action(s) applied"
                  + (f", {rejected} rejected" if rejected else "")
                  + f", notify {record['notify']['sent'] or '-'}"
                  + (" (dropped: budget spent)" if record["notify"]["dropped"] else ""))

    def _apply(self, calls: List[TagToolCall], record: Dict[str, Any]) -> None:
        """Each action on the shared graph, in order; a rejected one is noted
        with the graph's reason and the round goes on (DIG-TAG's rule)."""
        for call in calls:
            try:
                args = tag_call_args(call, self.team_name)
                self.tag.step(self.team_name, call.tool, args)
                result = "applied"
            except Exception as error:  # noqa: BLE001 - the graph's rejection is the record
                args = call.model_dump()
                result = f"rejected: {error}"
            record["tag_actions"].append({"tool": call.tool, "args": args, "result": result})

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
        self._say(f"notification from {self.team_name}: read the shared task graph",
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
        graph part -- and a replanned robot takes the new plan.
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


# -- what a run saves ------------------------------------------------------

def shared_tag(agents: Dict[str, Any]) -> TAGParallelInterface:
    """The one task graph a `tag` run shares."""
    graphs = {id(a.brain.tag): a.brain.tag for a in agents.values()
              if isinstance(getattr(a, "brain", None), TagTeamBrain)}
    if len(graphs) != 1:
        raise ValueError("not one task-graph run: expected every team to share one task graph")
    return next(iter(graphs.values()))


def tag_rounds_of(agents: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every team's rounds, in the order they happened."""
    brains = {id(a.brain): a.brain for a in agents.values()
              if isinstance(getattr(a, "brain", None), TagTeamBrain)}
    rounds = [record for brain in brains.values() for record in brain.tag_rounds]
    return sorted(rounds, key=lambda record: record["seq"])


def draw_tag(tag: TAGParallelInterface, output_dir: str) -> Dict[str, str]:
    """The graph drawn as its T band, tag.pdf and tag.png beside tag.json
    (DIG-TAG's ``_draw_tag``). The renderer's fonts are scoped to this
    figure so the run's other plots keep their look."""
    from pathlib import Path  # noqa: PLC0415

    import matplotlib  # noqa: PLC0415

    from coop2.dig_tag.render import canvas, measure_record, render_record  # noqa: PLC0415

    with matplotlib.rc_context():
        canvas.setup()
        size = measure_record(tag, bands=("T",), ids=True, axis=True, pitch=TAG_FIGURE_PITCH)
        fig = canvas.new_figure(size.w + 0.1, size.h + 0.1)
        render_record(canvas.add_panel(fig, 0.05, 0.05, size.w, size.h), tag,
                      bands=("T",), ids=True, axis=True, pitch=TAG_FIGURE_PITCH)
        return canvas.save(fig, "tag", Path(output_dir), preview=True)


def save_tag_outputs(agents: Dict[str, Any], output_dir: str) -> List[str]:
    """Write tag.json, tag_rounds.json and the drawing; return one line per
    file for the runner to print. A drawing that fails costs the picture,
    not the record."""
    tag = shared_tag(agents)
    lines = []
    path = tag.save_json(os.path.join(output_dir, "tag.json"))
    lines.append(f"Task graph saved to {path}")
    rounds_path = os.path.join(output_dir, "tag_rounds.json")
    with open(rounds_path, "w", encoding="utf-8") as handle:
        json.dump(tag_rounds_of(agents), handle, indent=2)
    lines.append(f"Task-graph rounds saved to {rounds_path}")
    try:
        drawn = draw_tag(tag, output_dir)
        lines.append("Task graph drawn: " + ", ".join(str(p) for p in drawn.values()))
    except Exception as error:  # noqa: BLE001 - a picture is not worth the run
        lines.append(f"Task graph not drawn: {type(error).__name__}: {error}")
    return lines
