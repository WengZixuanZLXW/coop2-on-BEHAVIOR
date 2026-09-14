"""Golden tests for the coupled transition -- the cases of `DIGTAG.issue`,
D and T as separate records, availability across the coupling, and
validate-before-mutate. The interface is the one write path into D, T, and Z.
Run with:  python tests/test_transition.py"""

import pytest

from dig_tag.dig import ENVIRONMENT_ORIGIN, DIGEvent
from dig_tag.interface import DIGTAG, TAG_ORIGIN
from dig_tag.tag import observed
from helpers import StubEnvironment, spec


def test_send_makes_an_existing_event_available_without_copying():
    dt = DIGTAG(agents=["alice", "bob", "carol"])
    root = dt.inject(DIGEvent(payload={"text": "start"}), to=["alice"])
    with dt.open_activation("alice") as h:
        dt.send(h, root, to=["bob", "carol"])
        (call,) = h.calls
        assert call.tool == "send" and call.returned == []             # (send, {}) in U_h
        assert call.args == {"event": "e1", "to": ["bob", "carol"]}
    assert dt.inbox("bob") == [root] and dt.inbox("bob")[0] is root      # no copy
    assert len(dt.dig.events) == 1                                        # no new event
    with dt.open_activation("bob") as h_bob:
        assert h_bob.inputs == [root]                                     # e -> h'_j at open
        assert h_bob.calls == []                                          # no second call
    assert root.producer is None                                          # no producing edge
    print("  ok: send records (send, {}) and routes the existing event to J")


def test_send_requires_availability_and_recipients_in_I():
    dt = DIGTAG(agents=["alice", "bob"])
    unseen = dt.inject(DIGEvent(), to=["bob"])
    mine = dt.inject(DIGEvent(), to=["alice"])
    with dt.open_activation("alice") as h:
        with pytest.raises(ValueError, match="cannot name e1"):
            dt.send(h, unseen, to=["bob"])
        with pytest.raises(ValueError, match="unknown agent id"):
            dt.send(h, mine, to=["zed"])
        with pytest.raises(ValueError, match="non-empty list of agent ids"):
            dt.send(h, mine, to=[])
        assert h.calls == []                            # nothing was recorded
    with pytest.raises(ValueError, match="unknown agent id"):
        dt.inject(DIGEvent(), to=["zed"])
    print("  ok: a send names only Avail(h, r) and J subset of I")


def test_task_call_updates_D_and_T_together():
    dt = DIGTAG(agents=["alice"])
    with dt.open_activation("alice") as h:
        handed = dt.open_task(h, spec("write", "3 pages"), state="todo")
        (call,) = h.calls
        assert call.tool == "open" and call.returned == [handed]          # D (+)_h (u, {e}): what T handed back
        assert handed.id == "e1" and handed.origin == TAG_ORIGIN and handed.producer == DIGEvent.Producer("h1", 0)
        (q1,) = observed(handed.payload)                                  # the observation: the frontier
        assert q1.id == "q1" and q1.identity == "k1" and handed.payload["open"] == ["k1"]
        assert "q1" not in dt.dig.events and dt.tag.tasks["q1"].spec.rule == "3 pages"   # D and T: separate records
        (action,) = dt.tag.actions.values()                               # T (+) (a, Q)
        assert action.issuer == "h1" and action.call is call
        assert action.input_ids == [] and action.output_ids == ["q1"]

        done = dt.update(h, "q1", state="done")                           # a task reference need only exist
        assert done.producer.call_index == 1 and dt.tag.tasks["q2"].identity == "k1"
        assert dt.tag.actions["a2"].input_ids == ["q1"] and dt.tag.actions["a2"].output_ids == ["q2"]
    print("  ok: a task call appends (a, Q) to T and (u, {e}) to D, e carrying what T handed back")


def test_environment_call_inserts_feedback_with_environment_origin():
    env = StubEnvironment()
    dt = DIGTAG(agents=["alice"], environment=env)
    with dt.open_activation("alice") as h:
        (feedback,) = dt.act(h, "echo", q="x")
        assert feedback.payload == {"q": "x"} and feedback.origin == ENVIRONMENT_ORIGIN
        assert feedback.producer == DIGEvent.Producer("h1", 0)
        assert h.calls[0].tool == "echo" and h.calls[0].args == {"q": "x"}
        assert env.calls == [h.calls[0]]

        (relayed,) = dt.act(h, "relay", text="hi")
        assert relayed.origin == "alice"                     # the environment declared it
        assert len(dt.tag.tasks) == 0 and len(dt.tag.actions) == 0   # T unchanged

        with pytest.raises(ValueError, match="unknown tool"):
            dt.act(h, "teleport")
        with pytest.raises(RuntimeError, match="rejected"):
            dt.act(h, "fail")
        assert len(h.calls) == 2 and len(dt.dig.events) == 2   # a rejected call leaves no trace
    print("  ok: an environment call appends (u, E_Z); Z is the environment's business")


def test_attach_adds_evidence_and_no_transition():
    dt = DIGTAG(agents=["alice", "bob"])
    with dt.open_activation("alice") as h:
        opened = dt.open_task(h, spec("write"), state="todo")
        handed = dt.attach(h, "q1", {"check": "ok"})
        phi = dt.tag.evidence["phi1"]
        assert phi.target == "q1" and phi.source == "h1"
        assert h.calls[1].tool == "attach" and h.calls[1].returned == [handed]   # (attach, {e}) in D
        assert len(dt.tag.actions) == 1                                          # no action node
        dt.attach(h, "q1", {"by": "reviewer"}, source="reviewer")
        assert dt.tag.evidence["phi2"].source == "reviewer"
        dt.send(h, opened, to=["bob"])
    with dt.open_activation("bob") as h_bob:
        dt.attach(h_bob, "q1", "seen")
        assert dt.tag.evidence["phi3"].source == "h2"
    assert [e.target for e in dt.tag.evidence.values()] == ["q1", "q1", "q1"]
    print("  ok: attach adds phi to T and records (attach, {e}) in D")


def test_attach_target_must_exist():
    dt = DIGTAG(agents=["alice", "bob"])
    with dt.open_activation("alice") as h:
        dt.open_task(h, spec("write"))
    with dt.open_activation("bob") as h_bob:
        dt.attach(h_bob, "q1", "x")                   # a task reference: it exists, though bob never received it
        phi = dt.tag.evidence["phi1"]
        assert phi.target == "q1" and phi.source == "h2" and h_bob.inputs == []
        with pytest.raises(ValueError, match="not in the TAG"):
            dt.attach(h_bob, "q9", "x")
        dt.attach(h_bob, "q1", "x")                   # the same triple: Phi unchanged, the call recorded
        assert len(dt.tag.evidence) == 1
        assert [c.tool for c in h_bob.calls] == ["attach", "attach"]
    print("  ok: attach names an existing exact version; a repeated triple leaves Phi unchanged")


def test_a_rejected_task_call_leaves_no_partial_state():
    dt = DIGTAG(agents=["alice"])
    with dt.open_activation("alice") as h:
        dt.open_task(h, spec("survey"), state="todo")
        before = (len(dt.dig.events), len(dt.tag.tasks), len(dt.tag.actions), len(h.calls))
        with pytest.raises(ValueError, match="at least two"):
            dt.split(h, "q1", [(None, spec("only"), "todo")])
        with pytest.raises(ValueError, match="at least two"):
            dt.join(h, ["q1"], spec("j"))
        with pytest.raises(ValueError, match="not in the TAG"):
            dt.update(h, "q9", state="x")
        assert (len(dt.dig.events), len(dt.tag.tasks), len(dt.tag.actions), len(h.calls)) == before
        assert dt.tag.identities == {"k1"}
    print("  ok: validation happens before any mutation")


def test_context_accumulates_across_an_agents_activations():
    dt = DIGTAG(agents=["planner", "worker"])
    with dt.open_activation("planner") as h1:
        opened = dt.open_task(h1, spec("survey"), state="todo")
        parts = dt.split(h1, "q1", [(None, spec("intro"), "todo"), (None, spec("body"), "todo")])
        dt.send(h1, parts, to=["worker"])
    with dt.open_activation("worker") as h2:
        assert h2.inputs == [parts]
        done_a = dt.update(h2, "q2", state="done")
        dt.send(h2, done_a, to=["planner"])
    with dt.open_activation("planner") as h3:
        assert [e.id for e in h3.inputs] == ["e1", "e2", "e3"]          # own returns, then the arrival
        dt.send(h3, opened, to=["worker"])                               # returned in h1, still in reach
    with dt.open_activation("worker") as h4:
        assert [e.id for e in h4.inputs] == ["e2", "e3", "e1"]          # parts, own return done_a, then the arrival
        dt.join(h4, ["q4", "q3"], spec("joined"), identity="k2")         # q4 came from an EARLIER activation
        assert dt.tag.tasks["q5"].identity == "k2" and dt.tag.actions["a4"].input_ids == ["q4", "q3"]
        dt.close(h4, "k1")                       # CLOSE names an identity: it need only exist
        assert h4.calls[-1].args == {"identity": "k1"} and dt.tag.actions["a5"].input_ids == []
    print("  ok: what an agent received or returned stays available to its later activations")


def test_own_returns_do_not_wake_the_agent():
    dt = DIGTAG(agents=["planner", "worker"])
    root = dt.inject(DIGEvent(), to=["planner"])
    with dt.open_activation("planner") as h1:
        handed = dt.open_task(h1, spec("survey"))
    assert dt.pending("planner") == []             # producing something is not an arrival
    assert dt.inbox("planner") == [root]
    with dt.open_activation("planner") as h2:
        assert h2.inputs == [root, handed]         # yet it is in the next context
        assert dt.received(h2) == [root]           # and is told apart from what arrived
        dt.close(h2, "k1")
    print("  ok: own returns enter the next context without counting as pending or received")


if __name__ == "__main__":
    from helpers import run_golden

    run_golden(globals(), "transition golden tests")
