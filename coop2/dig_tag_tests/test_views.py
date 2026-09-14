"""Golden tests for the views -- the Section 4.3 relations recovered from
one recorded run: L_D and L_T, producers and recipients, provenance,
availability-based ancestry, task versions, lineage, frontier, evidence,
environment transitions, and the per-agent aggregation.
Run with:  python tests/test_views.py"""

import pytest

from dig_tag import views
from dig_tag.interface import DIGTAG
from dig_tag.tag import TaskAction, TaskSpec, Version, current, observed
from helpers import StubEnvironment, spec


def _survey_run() -> DIGTAG:
    """planner splits a root task and hands the parts to two workers; each
    worker searches, updates its part, attaches evidence, and sends the
    observation back; planner joins the drafts and closes."""
    dt = DIGTAG(agents=["planner", "w1", "w2"], environment=StubEnvironment())
    dt.inject_task(Version(spec=spec("survey", "3 sections"), state="todo"), to=["planner"])       # q1; e1
    with dt.open_activation("planner") as h:                                                      # h1
        parts = dt.split(h, "q1", [(None, spec("intro"), "todo"), (None, spec("body"), "todo")])   # q2, q3; e2
        a, b = observed(parts.payload)
        dt.send(h, parts, to=["w1"])
        dt.send(h, parts, to=["w2"])
    for worker, part in (("w1", a), ("w2", b)):                                                   # h2, h3
        with dt.open_activation(worker) as h:
            (hits,) = dt.act(h, "echo", query=part.spec.goal)                                     # e3 / e6
            done = dt.update(h, part, state={"draft": part.spec.goal})                            # q4 / q5; e4 / e7
            dt.attach(h, current(done.payload, part.identity), {"hits": hits.id})                 # e5 / e8
            dt.send(h, done, to=["planner"])
    with dt.open_activation("planner") as h:                                                      # h4
        drafts = [q for q in observed(dt.received(h)[-1].payload) if isinstance(q.state, dict)]
        joined = dt.join(h, drafts, spec("survey"), state="joined")      # q6, a fresh identity: k1 is not an input; e9
        dt.close(h, observed(joined.payload)[0].identity)                                         # e10
    return dt


def test_dig_edges_materialize_L_D():
    dt = _survey_run()
    edges = views.dig_edges(dt)
    inputs = {(e.source, e.target) for e in edges if e.relation == "input"}
    returns = {(e.source, e.target, e.call_index) for e in edges if e.relation == "return"}
    assert ("e1", "h1") in inputs and ("e2", "h2") in inputs and ("e2", "h3") in inputs
    assert ("h1", "e2", 0) in returns                                     # the split returned one event: the observation
    assert ("h2", "e3", 0) in returns and ("h2", "e4", 1) in returns
    assert ("e4", "h4") in inputs and ("e7", "h4") in inputs             # the workers' sends reached the planner
    assert ("e2", "h4") in inputs                                        # and its own earlier return is re-presented
    assert [e.id for e in views.roots(dt)] == ["e1"]
    print("  ok: L_D derives from inputs and returns")


def test_producer_recipients_and_provenance():
    dt = _survey_run()
    e2, e3, e1 = dt.dig.events["e2"], dt.dig.events["e3"], dt.dig.events["e1"]
    activation, call = views.producer(dt, e2)
    assert activation.id == "h1" and call.tool == "split"
    assert views.producer(dt, e1) is None
    assert views.recipients(dt, e2) == ["w1", "w2"] and views.recipients(dt, e1) == []
    assert [h.id for h in views.presented_to(dt, e2)] == ["h2", "h3", "h4"]   # both workers, planner's re-read
    assert [h.id for h in views.presented_to(dt, e1)] == ["h1", "h4"]

    p = views.provenance(dt, e3)
    assert (p.origin, p.agent_id, p.tool) == ("environment", "w1", "echo")
    assert str(views.provenance(dt, e1)) == "root (origin tag)"
    assert views.provenance(dt, e2).agent_id == "planner"
    print("  ok: producers, recipients, and provenance are read off the record")


def test_ancestry_follows_availability():
    dt = _survey_run()
    e4 = dt.dig.events["e4"]          # what w1's update handed back
    ancestors = [e.id for e in views.ancestry(dt, e4)]
    assert ancestors == ["e2", "e3", "e1"]   # Avail(h2, 1), then e2's own ancestry: the root
    final = dt.dig.events["e9"]
    assert set(e.id for e in views.ancestry(dt, final)) == {"e1", "e2", "e3", "e4", "e6", "e7"}
    assert views.ancestry(dt, dt.dig.events["e1"]) == []
    assert [e.id for e in views.dependents(dt, dt.dig.events["e3"])] == ["e4", "e5", "e9", "e10"]
    h2 = dt.dig.activations["h2"]
    assert [e.id for e in views.available(dt, h2, 0)] == ["e2"]
    assert [e.id for e in views.available(dt, h2)] == ["e2", "e3", "e4", "e5"]
    print("  ok: ancestry is the availability-based over-approximation")


def test_tag_edges_versions_lineage_and_frontier():
    dt = _survey_run()
    edges = views.tag_edges(dt)
    assert {(e.source, e.target) for e in edges if e.relation == "consume"} == {
        ("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4"), ("q5", "a4"),      # a5, the CLOSE, consumes nothing
    }
    assert {(e.source, e.target) for e in edges if e.relation == "return"} == {
        ("a1", "q2"), ("a1", "q3"), ("a2", "q4"), ("a3", "q5"), ("a4", "q6"),
    }
    assert [q.id for q in views.versions(dt, "k1")] == ["q1"]
    assert [q.id for q in views.versions(dt, "k2")] == ["q2", "q4"]
    assert [q.id for q in views.versions(dt, "k4")] == ["q6"]        # the join introduced k4
    q6 = dt.tag.tasks["q6"]
    assert [q.id for q in views.lineage(dt, q6)] == ["q4", "q5", "q2", "q3", "q1"]
    assert [q.id for q in views.descendants(dt, dt.tag.tasks["q1"])] == ["q2", "q3", "q4", "q5", "q6"]
    assert [q.id for q in views.frontier(dt)] == ["q6"]   # CLOSE consumed nothing: q6 stays on the frontier
    observation = views.build_observation(dt)              # the task configuration: K_t and F_t
    # closing the join closes k2 and k3 it joined, and then k1, every part of whose split is closed
    assert observation.closed_identities == ["k1", "k2", "k3", "k4"] and observation.open_identities == []
    assert [q.id for q in observation.frontier] == ["q6"] and observation.open_frontier == []
    assert views.producing_action(dt, q6).label is TaskAction.Label.JOIN
    assert views.producing_action(dt, dt.tag.tasks["q1"]) is None
    assert [a.id for a in views.consuming_actions(dt, dt.tag.tasks["q2"])] == ["a2"]
    a4 = dt.tag.actions["a4"]
    assert [q.id for q in views.inputs(dt, a4)] == ["q4", "q5"] and [q.id for q in views.outputs(dt, a4)] == ["q6"]
    assert [(a.id if a else None, q.id) for a, q in views.history(dt, "k2")] == [("a1", "q2"), ("a2", "q4")]
    assert [(a, q.id) for a, q in views.history(dt, "k1")] == [(None, "q1")]        # an initial version
    assert views.relations(dt, "k2")["origin"] == {
        "tool": "split", "issuer": "h1", "from": [{"version": "q1", "identity": "k1"}],
    }
    assert [s["identities"] for s in views.relations(dt, "k1")["successors"]] == [["k2", "k3"]]   # the split
    assert [s["identities"] for s in views.relations(dt, "k2")["successors"]] == [["k4"]]         # the join
    assert views.relations(dt, "k4")["origin"]["from"] == [
        {"version": "q4", "identity": "k2"}, {"version": "q5", "identity": "k3"},
    ]
    print("  ok: L_T, versions, lineage, histories, relations, and the frontier derive from the actions")


def test_frontier_is_the_latest_task_state():
    dt = DIGTAG(agents=["a"])
    with dt.open_activation("a") as h:
        dt.open_task(h, spec("x"), state=1)
        dt.update(h, "q1", state=2)
        dt.open_task(h, spec("y"), state=1)
    assert [q.id for q in views.frontier(dt)] == ["q2", "q3"]        # the update and the second task
    with dt.open_activation("a") as h:
        dt.close(h, dt.tag.tasks["q2"])                              # CLOSE(k1): the version stands for its identity
        dt.edit(h, "q1", spec("x branched"))                         # q1 is consumed: the edit branches, closed or not
    assert dt.tag.tasks["q4"].identity == "k3"
    assert [q.id for q in views.frontier(dt)] == ["q2", "q3", "q4"]  # closing left the frontier unchanged
    observation = views.build_observation(dt)
    assert observation.closed_identities == ["k1"] and observation.open_identities == ["k2", "k3"]   # a branch goes on
    assert [q.id for q in observation.open_frontier] == ["q3", "q4"]
    print("  ok: the frontier is every version no action consumed; CLOSE leaves it unchanged")


def test_closure_reaches_back_through_joins_and_splits():
    dt = DIGTAG(agents=["a"])
    with dt.open_activation("a") as h:
        dt.open_task(h, spec("root"), state=0)                                      # k1
        dt.split(h, "q1", [(None, spec("p1"), 0), (None, spec("p2"), 0)])          # k2, k3
        dt.update(h, "q2", state=1)                                                 # q4 (k2)
        dt.close(h, "k2")
    seen = views.build_observation(dt)
    assert seen.closed_identities == ["k2"] and seen.open_identities == ["k1", "k3"]   # one part closed: k1 stays
    with dt.open_activation("a") as h:
        dt.join(h, ["q4", "q3"], spec("joined"), state=0)                           # q5 (k4): a closed input is fine
        dt.update(h, "q5", state=1)                                                 # q6 (k4)
        dt.close(h, "k4")
    seen = views.build_observation(dt)
    assert seen.closed_identities == ["k1", "k2", "k3", "k4"] and seen.open_identities == []
    print("  ok: a closed join closes what it joined; a split's input closes once all its parts are")


def test_evidence_stays_on_its_exact_version():
    dt = _survey_run()
    q4, q5, q6 = (dt.tag.tasks[i] for i in ("q4", "q5", "q6"))
    assert [phi.payload for phi in views.evidence_for(dt, q4)] == [{"hits": "e3"}]
    assert [phi.payload for phi in views.evidence_for(dt, q5)] == [{"hits": "e6"}]
    assert views.evidence_for(dt, q6) == []                # the successor inherits nothing

    dt2 = DIGTAG(agents=["a"])
    with dt2.open_activation("a") as h:
        dt2.open_task(h, TaskSpec("count", rule=lambda goal, state: len(state)), state=[1, 2, 3])
        dt2.open_task(h, spec("count", "3 items"), state=[1])
    assert views.evaluate(dt2, dt2.tag.tasks["q1"]) == 3
    with pytest.raises(TypeError, match="non-callable rule"):
        views.evaluate(dt2, dt2.tag.tasks["q2"])
    print("  ok: evidence targets exact versions; evaluate is opt-in over callable rules")


def test_environment_transitions_in_order():
    dt = _survey_run()
    steps = views.transitions(dt)
    assert [str(t) for t in steps] == ["w1@h2: echo -> [e3]", "w2@h3: echo -> [e6]"]
    assert [c.args["query"] for c in views.environment_calls(dt)] == ["intro", "body"]
    assert [e.id for e in views.feedback(dt, steps[0].call)] == ["e3"]
    with pytest.raises(ValueError, match="not an environment tool"):
        views.feedback(dt, dt.dig.activations["h1"].calls[0])
    print("  ok: environment calls and feedback form the transition sequence")


def test_agent_view_aggregates_activations():
    dt = _survey_run()
    interaction = views.interaction_graph(dt)
    assert interaction.agents == ["planner", "w1", "w2"]
    assert interaction.edge_set == {("planner", "w1"), ("planner", "w2"), ("w1", "planner"), ("w2", "planner")}
    assert {(e.source, e.target): e.event_ids for e in interaction.edges}[("planner", "w1")] == ["e2"]
    assert interaction.successors("planner") == ["w1", "w2"] and interaction.predecessors("planner") == ["w1", "w2"]
    assert views.interaction_edges(dt) == interaction.edge_set

    flow = views.flow_graph(dt)
    assert flow.edge_set == {
        ("planner", "w1"), ("planner", "w2"), ("w1", "planner"), ("w2", "planner"),
        ("planner", "planner"),                                  # h4 re-read its own e2
    }
    assert {(e.source, e.target): e.event_ids for e in flow.edges}[("planner", "planner")] == ["e2"]
    print("  ok: the agent view aggregates sends (interaction) and provenance (flow)")


if __name__ == "__main__":
    from helpers import run_golden

    run_golden(globals(), "views golden tests")
