"""Run a written-down plan instead of asking a model.

The same round as `individual` -- one brain per team, a barrier, one decision
per team -- with the model call replaced by a lookup in a script file. A robot
the script does not name, or one whose entries have all been handed out, holds
position: `TeamBrain` already has exactly that plan (``build_hold_plan``), the
one a team issues to a member it is waiting for, so "no script" costs nothing
new and looks in the logs like any other hold.

Nothing else about the run changes. The environment, the team barrier, the FSM,
the route supervision and every output file are the ones the LLM modes use, so
a scripted episode is comparable with them and records a video the same way.

The script is JSON, one entry per robot, and the n-th entry is handed out on
that team's n-th planning round -- entries are a robot's successive turns, not
alternatives::

    {
      "ridgeback_1": [
        {"task": "holding(notebook.n.01_1)",
         "actions": [{"action_type": "navigate_to", "target": "notebook.n.01_1"},
                     {"action_type": "grasp", "target": "notebook.n.01_1"}]},
        {"task": "standby", "actions": [{"action_type": "wait", "ticks": 600}]}
      ],
      "jackal_1": [
        {"task": "standby", "actions": [{"action_type": "wait", "ticks": 300}]}
      ]
    }

A robot whose whole script is one plan may drop the wrapper and give the action
list directly; ``{"action_type": ...}`` in the first slot is what distinguishes
the two shapes.

Actions are validated against the very Pydantic vocabulary structured output
constrains the model to (``llm_client.LLMAction``), so a script cannot express
an action a model could not, and a typo is a load-time error naming the robot
and the slot rather than a silent no-op in the middle of an episode. An action
may also be written in the ``{"action_type": ..., "args": {...}}`` shape that
``plan_logs.json`` uses, so a turn can be lifted out of a previous run's log
and replayed.
"""

from __future__ import annotations

import json
import typing
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from coop2.cognitive.action.action import SymbolicAction
from coop2.cognitive.agent.cognitive_agent import parse_plan_response
from coop2.cognitive.plan import SymbolicPlan
from coop2.comm_topology.llm_team import TeamBrain

__all__ = ["SCRIPT_IDLE_TICKS", "ScriptedTeamBrain", "NoLLM", "load_script", "script_template"]

#: How long a robot with nothing scripted stands still before asking again.
#:
#: Short, and it has to be. The two obvious alternatives both hang the team,
#: and both were measured doing it:
#:
#: * ``build_hold_plan()`` -- the team's own hold -- is 64 x 600 ticks, and it
#:   is only ever ended by the recall that runs when the barrier closes. Issued
#:   as this round's *plan* nothing recalls it, because the barrier cannot
#:   close while this member is executing: one scripted round in 2500 steps.
#: * No plan at all sends the member down ``claim_pending_plan``'s hold path,
#:   which is recallable -- but whichever member *closes* the barrier is
#:   discarded from ``_awaiting`` first and then takes that same 38 400-tick
#:   hold without ever re-asking, so the next barrier waits for a member that
#:   is not coming. Measured: team_5's second round never happened.
#:
#: A short plan avoids both. The member completes it, asks again, is put back
#: in ``_awaiting``, and takes the recallable hold there -- so it is parked
#: exactly where every other mode's waiting member is parked, and the team's
#: next round costs at most this many ticks of extra latency. The churn this
#: causes is what ``TEAM_HOLD_ACTIONS`` exists to avoid in the LLM modes,
#: where each cycle spends an API call; here a cycle costs nothing.
SCRIPT_IDLE_TICKS = 60


class NoLLM:
    """Stands where the model goes, and is never called.

    The agents read three things off the client (``model``, ``verbose``,
    ``io_recorder``) and only ever *call* it to plan or to decide an interrupt,
    neither of which happens here. Holding this instead of a real client is
    what lets a scripted run work with no credentials at all -- which is half
    the point of having one.
    """

    model = "scripted (no LLM)"
    verbose = False
    io_recorder = None

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"the scripted runner does not call the model (asked for {name!r})"
        )


def _action_models() -> Dict[str, Any]:
    """``{"navigate_to": NavigateToAction, ...}`` from the emittable union.

    Read off ``LLMAction`` rather than listed here: the union is the one place
    that decides what a plan may contain, and a second list would drift from it
    the first time a verb is added.
    """
    from coop2.cognitive.agent.llm_client import LLMAction  # noqa: PLC0415

    models = {}
    for model in typing.get_args(LLMAction):
        literal = model.model_fields["action_type"].default
        models[literal] = model
    return models


def _build_action(raw: Any, where: str):
    """One validated action, from either the plan shape or the log shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: an action must be an object, got {type(raw).__name__}")
    fields = dict(raw)
    # `plan_logs.json` nests everything but the verb under "args"; a script
    # written by hand puts them at the top level. Accept both, so a turn can be
    # copied out of a log unchanged.
    nested = fields.pop("args", None)
    if isinstance(nested, dict):
        fields.update(nested)
    action_type = fields.pop("action_type", None)
    if not action_type:
        raise ValueError(f"{where}: action has no action_type")
    models = _action_models()
    model = models.get(action_type)
    if model is None:
        raise ValueError(
            f"{where}: unknown action_type {action_type!r}; "
            f"the emittable verbs are {sorted(models)}"
        )
    try:
        return model(**fields)
    except Exception as error:  # noqa: BLE001 - re-raised with the robot and slot
        raise ValueError(f"{where}: {action_type} rejected these fields: {error}") from error


def _build_plan(raw: Any, where: str):
    """One entry of a robot's script, as the response the parser expects."""
    from coop2.cognitive.agent.llm_client import LLMPlanResponse  # noqa: PLC0415

    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a plan must be an object, got {type(raw).__name__}")
    actions = raw.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{where}: a plan needs a non-empty 'actions' list")
    return LLMPlanResponse(
        task=str(raw.get("task") or raw.get("specification") or "scripted"),
        actions=[_build_action(a, f"{where} action {i}") for i, a in enumerate(actions)],
        reasoning=str(raw.get("reasoning") or "from the script"),
    )


def load_script(path: str | Path) -> Dict[str, List[Any]]:
    """``{robot id: [plan, ...]}``, every action validated.

    Raises at load time, before the simulator is launched, so a bad script
    costs a second rather than the ten minutes an episode takes.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("a script is an object keyed by robot id")
    # A script may carry a "_comment" or similar for its reader; keys that
    # start with an underscore are not robots.
    script: Dict[str, List[Any]] = {}
    for robot, raw in data.items():
        if robot.startswith("_"):
            continue
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{robot}: expected a non-empty list of plans")
        # One plan given as a bare action list, rather than a list of plans.
        if isinstance(raw[0], dict) and "action_type" in raw[0]:
            raw = [{"task": "scripted", "actions": raw}]
        script[robot] = [_build_plan(p, f"{robot} plan {i}") for i, p in enumerate(raw)]
    return script


def script_template(teams: Dict[str, Iterable[str]]) -> str:
    """A skeleton script for @teams, ready to fill in.

    Every robot gets one short hold, so the file runs as written (nobody moves)
    and each robot can be given real turns one at a time.
    """
    body: Dict[str, Any] = {
        "_comment": "The n-th plan of a robot is handed to it on its team's n-th "
                    "round. Delete a robot to leave it standing still.",
    }
    for team, members in teams.items():
        for robot in members:
            body[robot] = [{
                "task": f"standby ({team})",
                "actions": [{"action_type": "wait", "ticks": 600}],
            }]
    return json.dumps(body, indent=2)


class ScriptedTeamBrain(TeamBrain):
    """A team whose plans are read, not generated.

    Only ``_generate_team_plans`` is replaced. Everything the base class does
    around it -- the barrier, recalling holders, the timeline entry, the round
    counter -- is untouched, which is what keeps a scripted episode structurally
    the same as an `individual` one.
    """

    def __init__(self, *args, script: Dict[str, List[Any]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        #: Keyed by robot id across the whole run; each brain reads only its own
        #: members out of it, so one file describes every team.
        self.script = script or {}
        #: How many entries each robot has already been handed.
        self._handed_out: Counter = Counter()

    def _idle_plan(self, member) -> SymbolicPlan:
        """One short `wait`, for a robot this script has nothing to say about.

        Built directly rather than through ``parse_plan_response``, for the
        reason ``build_hold_plan`` is: that helper appends an action to match
        the plan's stated goal, and standing still has none.
        """
        return SymbolicPlan(
            specification="standby (no script)",
            actions=[SymbolicAction(action_type="wait", args={"ticks": SCRIPT_IDLE_TICKS})],
            plan_id=member.plan_count + 1,
            agent_id=member.agent_id,
            created_at_step=member.env_step,
        )

    def _generate_team_plans(self) -> Dict[str, Any]:
        plans: Dict[str, Any] = {}
        for name in self.member_ids:
            member = self.members.get(name)
            if member is None:
                continue
            entries = self.script.get(name) or []
            index = self._handed_out[name]
            if index >= len(entries):
                # No script, or the script is spent: stand still briefly and
                # ask again. See SCRIPT_IDLE_TICKS for why it is neither the
                # team's own hold nor no plan at all.
                plans[name] = self._idle_plan(member)
                if self.verbose and not entries:
                    print(f"  [{name}] no script; standing still")
                continue
            self._handed_out[name] += 1
            plan = parse_plan_response(
                llm_response=entries[index],
                agent_id=name,
                env_step=member.env_step,
                plan_id=member.plan_count + 1,
            )
            plans[name] = member._finalize_generated_plan(plan)
            if self.verbose:
                print(f"  [{name}] scripted plan {index + 1}/{len(entries)}: "
                      f"{plan.specification}")
        return plans
