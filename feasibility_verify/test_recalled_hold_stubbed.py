"""CPU-only tests for what a recalled team hold leaves behind.

Two symptoms, read off plan_logs.json of the 12-robot run and reproduced
here: plans whose `action_type` is navigate_to but whose outcome is a 600-tick
WAIT that started under the *previous* plan, and plans of shape
`[wait, place_on_top]` that could only ever fail their precondition.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/zixuanwe/Desktop/BEHAVIOR-1K")
from coop2.cognitive.agent.cognitive_agent import (
    _ensure_task_terminal_action, SymbolicAction,
)
from coop2.cognitive.plan.plan import (
    SymbolicPlan, SymbolicPlanLogger, SymbolicPlanStatus,
)
from coop2.cognitive.plan.plan_env_wrapper import PlanningEnvWrapper


class FakeEngine:
    def __init__(self, active): self.active, self.aborted = active, []
    def has_active(self, agent_id): return agent_id in self.active
    def abort(self, agent_id, retract=False): self.aborted.append(agent_id)


class FakeAgent:
    def __init__(self, plan): self.ready, self.plan, self.plan_history = True, plan, []
    def record_plan_outcome(self, plan, succeeded, reason="", env_step=None, status=None):
        self.plan_history.append({"plan_id": plan.plan_id, "specification": plan.specification,
                                  "succeeded": succeeded, "status": status, "reason": reason})


class FakeTrace:
    def log(self, *a, **k): pass


def _plan(spec, plan_id, actions):
    return SymbolicPlan(specification=spec, actions=actions, plan_id=plan_id,
                        agent_id="agent_1", created_at_step=0)


def _wrapper(logger, agent, engine):
    """The real _sync_committed_plans, with only what it touches wired up."""
    wrapper = object.__new__(PlanningEnvWrapper)
    wrapper.agents = {"agent_1": agent}
    wrapper.logger = logger
    wrapper._current_step = 3288
    wrapper.coop2_trace = FakeTrace()
    wrapper._prev_action_results = {}
    wrapper.symbolic_env = type("Env", (), {
        "env": type("Base", (), {"engine": engine})(), "agent_actions": {},
    })()
    return wrapper


def main() -> int:
    print("test: a recalled hold is filed, and its primitive is aborted")
    logger = SymbolicPlanLogger()
    hold = _plan("wait_for_team(team_0)", 5,
                 [SymbolicAction(action_type="wait", args={"ticks": 600})])
    logger.log_plan_created(hold)
    logger.log_plan_started(hold, 2840)
    # What _recall_holders does: end it by assignment, because needs_new_plan()
    # asks the plan and R with a live plan hangs the run.
    hold.status = SymbolicPlanStatus.INTERRUPTED

    nxt = _plan("ontop(apple.n.01_8, coffee_table.n.01_1)", 6,
                [SymbolicAction(action_type="navigate_to", args={"target": "apple.n.01_8"})])
    engine = FakeEngine(active={"agent_1"})
    _wrapper(logger, FakeAgent(nxt), engine)._sync_committed_plans()

    filed = [p for p in logger.plan_history if p["specification"].startswith("wait_for_team")]
    assert len(filed) == 1, logger.plan_history
    assert filed[0]["end_step"] == 3288, filed[0]
    assert filed[0]["duration"] == 3288 - 2840, filed[0]
    # The symptom: without this the 600-tick WAIT keeps running and the new
    # plan's navigate_to reports the WAIT's outcome instead of moving.
    assert engine.aborted == ["agent_1"], engine.aborted
    print("  ok: end_step stamped, archived once, stale primitive dropped")

    print("\ntest: a plan that ended through the logger is left alone")
    logger = SymbolicPlanLogger()
    done = _plan("ontop(a, t)", 1, [SymbolicAction(action_type="grasp", args={})])
    logger.log_plan_created(done)
    logger.log_plan_started(done, 0)
    logger.log_plan_completed(done, 986, True)
    before = len(logger.plan_history)
    engine = FakeEngine(active={"agent_1"})
    _wrapper(logger, FakeAgent(_plan("ontop(b, t)", 2, [])), engine)._sync_committed_plans()
    assert len(logger.plan_history) == before, logger.plan_history
    assert engine.aborted == [], engine.aborted
    print("  ok: no double filing, and a finished primitive is not aborted")

    print("\ntest: a live plan is still interrupted, not filed as ended")
    logger = SymbolicPlanLogger()
    live = _plan("ontop(a, t)", 1, [SymbolicAction(action_type="grasp", args={})])
    logger.log_plan_created(live)
    logger.log_plan_started(live, 0)
    engine = FakeEngine(active={"agent_1"})
    agent = FakeAgent(_plan("ontop(b, t)", 2, []))
    _wrapper(logger, agent, engine)._sync_committed_plans()
    assert logger.plan_history[0]["status"] == "interrupted"
    assert logger.plan_history[0]["failure_reason"] == "Replanned"
    assert engine.aborted == ["agent_1"]
    # And the agent's own record gets the abandoned plan, marked as such --
    # not as a failure, which it was not, and not omitted, which left a hole in
    # the history the model reads.
    assert [(e["plan_id"], e["status"]) for e in agent.plan_history] == [(1, "replaced")], agent.plan_history
    assert not agent.plan_history[0]["succeeded"]
    print("  ok: the replan path is unchanged, and the replaced plan is an outcome")

    print("\ntest: a recalled hold reaches the agent's record but not its prompt")
    from coop2.cognitive.agent.prompts import format_plan_history
    logger = SymbolicPlanLogger()
    hold = _plan("wait_for_team(team_0)", 5, [SymbolicAction(action_type="wait", args={"ticks": 600})])
    logger.log_plan_created(hold)
    logger.log_plan_started(hold, 2840)
    hold.status = SymbolicPlanStatus.INTERRUPTED
    agent = FakeAgent(_plan("ontop(b, t)", 6, []))
    _wrapper(logger, agent, FakeEngine(active=set()))._sync_committed_plans()
    assert agent.plan_history and agent.plan_history[0]["status"] == "replaced"
    assert format_plan_history(agent.plan_history) == "", format_plan_history(agent.plan_history)
    print("  ok: recorded, hidden -- a hold is not a plan the model made")

    print("\ntest: a wait-only plan keeps its shape")
    actions = [SymbolicAction(action_type="wait", args={"ticks": 600})]
    _ensure_task_terminal_action("ontop(apple.n.01_2, coffee_table.n.01_1)", actions)
    assert [a.action_type for a in actions] == ["wait"], actions
    print("  ok: a deliberate wait is no longer turned into a failed place")

    print("\ntest: the omission this guard is for is still caught")
    actions = [SymbolicAction(action_type="navigate_to", args={"target": "coffee_table.n.01_1"})]
    _ensure_task_terminal_action("ontop(apple.n.01_2, coffee_table.n.01_1)", actions)
    assert [a.action_type for a in actions] == ["navigate_to", "place_on_top"], actions
    empty = []
    _ensure_task_terminal_action("holding(apple.n.01_2)", empty)
    assert [a.action_type for a in empty] == ["grasp"], empty
    mixed = [SymbolicAction(action_type="wait", args={"ticks": 100}),
             SymbolicAction(action_type="navigate_to", args={"target": "apple.n.01_2"})]
    _ensure_task_terminal_action("holding(apple.n.01_2)", mixed)
    assert [a.action_type for a in mixed] == ["wait", "navigate_to", "grasp"], mixed
    print("  ok: navigation-only, empty and wait-then-act plans all get theirs")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
