"""CPU-only checks on `scripted`: plans read from a file, no model at all.

What the mode promises, and what is checked here:

  1. a robot's n-th entry is handed to it on its team's n-th round;
  2. a robot the script does not name never gets a plan of its own -- it
     holds, which is the same hold every mode issues at a barrier;
  3. a robot whose entries run out goes back to holding;
  4. the actions are the ones a model could emit, validated at load time,
     so a typo names the robot and the slot instead of failing mid-episode;
  5. an action may be written in `plan_logs.json`'s own shape, so a turn can
     be replayed out of a previous run;
  6. the client is never called -- a scripted run needs no credentials;
  7. `--print-template` produces a script that loads.

No Isaac, no model.

Run:
    python feasibility_verify/test_scripted_stubbed.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.comm_topology.llm_team import TEAM_BRAIN_ROLES, create_llm_team_topology
from coop2.comm_topology.scripted_team import SCRIPT_IDLE_TICKS, NoLLM, load_script, script_template


def ok(message: str) -> None:
    print(f"  ok: {message}")


TEAMS = {"team_1": ["ridgeback_1", "jackal_1"], "team_2": ["drone_2"]}

SCRIPT = {
    "_comment": "ignored",
    "ridgeback_1": [
        {"task": "holding(die.n.01_1)",
         "actions": [{"action_type": "navigate_to", "target": "die.n.01_1"},
                     {"action_type": "grasp", "target": "die.n.01_1"}]},
        {"task": "ontop(die.n.01_1, cabinet.n.01_1)",
         "actions": [{"action_type": "place_on_top", "target": "cabinet.n.01_1"}]},
    ],
    # One plan as a bare action list, and in plan_logs.json's nested shape.
    "jackal_1": [{"action_type": "wait", "args": {"ticks": 300}}],
    # drone_2 is deliberately absent: it must never move.
}


def write(data) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    return handle.name


class ExplodingClient(NoLLM):
    """A scripted run must not reach the model by any route."""

    def generate(self, *args, **kwargs):
        raise AssertionError("the scripted mode called the model")

    def generate_team_plan(self, *args, **kwargs):
        raise AssertionError("the scripted mode called the model")


def make_run(script):
    agents = create_llm_team_topology(
        llm_client=ExplodingClient(), teams=TEAMS, topology="scripted",
        verbose=False, script=script,
    )
    for agent in agents.values():
        agent.env_step = 0
    return agents


def brain_of(agents, team):
    return agents[TEAMS[team][0]].brain


def specs(plans):
    return {name: plan.specification for name, plan in plans.items()}


def actions(plan):
    return [(a.action_type, a.args) for a in plan.actions]


def test_1_entries_are_turns():
    """The n-th entry on the n-th round, in order."""
    agents = make_run(load_script(write(SCRIPT)))
    brain = brain_of(agents, "team_1")

    first = brain._generate_team_plans()
    assert first["ridgeback_1"].specification == "holding(die.n.01_1)", first["ridgeback_1"].specification
    assert actions(first["ridgeback_1"])[0] == ("navigate_to", {"target": "die.n.01_1"}), actions(first["ridgeback_1"])
    assert actions(first["ridgeback_1"])[1][0] == "grasp"

    second = brain._generate_team_plans()
    assert second["ridgeback_1"].specification == "ontop(die.n.01_1, cabinet.n.01_1)", second["ridgeback_1"].specification
    ok("a robot's n-th entry is handed to it on its team's n-th round")


def test_2_unscripted_robots_hold():
    """drone_2 is in no entry of the script, so it gets no plan at all.

    It gets one short `wait` -- neither the team's 64 x 600 hold (nothing
    recalls one issued as a plan) nor no plan at all (the member that closes
    the barrier is then dropped from `_awaiting` and holds for 38 400 ticks
    without re-asking). Both were measured hanging a team's next round; see
    SCRIPT_IDLE_TICKS.
    """
    agents = make_run(load_script(write(SCRIPT)))
    brain = brain_of(agents, "team_2")
    for _ in range(3):
        plans = brain._generate_team_plans()
        idle = plans["drone_2"]
        assert idle.specification == "standby (no script)", idle.specification
        assert [(a.action_type, a.args) for a in idle.actions] == [
            ("wait", {"ticks": SCRIPT_IDLE_TICKS})], actions(idle)
    ok("a robot the script does not name stands still and re-asks, every round")


def test_3_a_spent_script_holds():
    """ridgeback_1 has two entries; the third round is a hold."""
    agents = make_run(load_script(write(SCRIPT)))
    brain = brain_of(agents, "team_1")
    brain._generate_team_plans()
    brain._generate_team_plans()
    third = brain._generate_team_plans()
    assert third["ridgeback_1"].specification == "standby (no script)", third["ridgeback_1"].specification
    assert third["jackal_1"].specification == "standby (no script)", third["jackal_1"].specification
    ok("a robot whose entries are spent stands still the same way")


def test_4_actions_are_validated_at_load():
    """A verb no model could emit, and a field it would reject."""
    for bad, needle in [
        ({"r": [{"action_type": "teleport", "target": "x"}]}, "unknown action_type"),
        ({"r": [{"action_type": "grasp"}]}, "grasp rejected"),
        ({"r": [{"action_type": "wait", "ticks": 9999}]}, "wait rejected"),
        ({"r": [{"task": "t", "actions": []}]}, "non-empty 'actions'"),
    ]:
        try:
            load_script(write(bad))
        except ValueError as error:
            assert needle in str(error), f"{needle!r} not in {error}"
            assert "r " in str(error), f"the robot is not named in {error}"
        else:
            raise AssertionError(f"{bad} should not have loaded")
    ok("a bad action is a load-time error naming the robot and the slot")


def test_5_the_log_shape_replays():
    """An action lifted out of plan_logs.json loads unchanged."""
    from_log = {"jackal_1": [{"task": "standby",
                              "actions": [{"action_type": "wait", "args": {"ticks": 300},
                                           "status": "success", "failure_reason": None}]}]}
    script = load_script(write(from_log))
    agents = make_run(script)
    plans = brain_of(agents, "team_1")._generate_team_plans()
    assert actions(plans["jackal_1"]) == [("wait", {"ticks": 300})], actions(plans["jackal_1"])
    ok("an action in plan_logs.json's shape replays, extra keys and all")


def test_6_the_model_is_never_called():
    """Every round of both teams, with a client that raises on any call."""
    agents = make_run(load_script(write(SCRIPT)))
    for team in TEAMS:
        for _ in range(3):
            brain_of(agents, team)._generate_team_plans()
    assert "scripted" in TEAM_BRAIN_ROLES
    ok("no round touches the model; the mode is in the topology registry")


def test_7_the_template_loads():
    """--print-template's output is a valid script that moves nobody."""
    script = load_script(write(json.loads(script_template(TEAMS))))
    assert set(script) == {"ridgeback_1", "jackal_1", "drone_2"}, sorted(script)
    agents = make_run(script)
    plans = brain_of(agents, "team_1")._generate_team_plans()
    assert {a.action_type for a in plans["ridgeback_1"].actions} == {"wait"}
    ok("the printed template loads, and every robot in it stands still")


if __name__ == "__main__":
    print("scripted mode (CPU)")
    test_1_entries_are_turns()
    test_2_unscripted_robots_hold()
    test_3_a_spent_script_holds()
    test_4_actions_are_validated_at_load()
    test_5_the_log_shape_replays()
    test_6_the_model_is_never_called()
    test_7_the_template_loads()
    print("\nALL TESTS PASSED")
