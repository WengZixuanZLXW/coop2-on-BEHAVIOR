"""Golden tests for the TAG representation -- the six task-tool signatures
(fresh tokens, identity rules, m >= 2, immutable history, branching from
a consumed version), evidence attachment to an exact version, and the record.
Run with:  python tests/test_tag.py"""

import pytest

from dig_tag.base import ToolCall
from dig_tag.tag import Evidence, TAGGraph, TaskAction, Version
from helpers import spec


def _realize(tag: TAGGraph, call: ToolCall, index: int = 0) -> TaskAction:
    """Realize one call: build its outputs, stamp the call, record the
    action with them; the graph mints the versions' ids."""
    outputs = tag.transition(call)
    call.index = index
    return tag.record(TaskAction(issuer="h1", call=call, outputs=outputs))


def test_open_mints_a_fresh_identity_and_returns_one_node():
    tag = TAGGraph()
    action = _realize(tag, ToolCall("open", {"spec": spec("write").to_dict(), "state": "todo"}))

    (q,) = action.outputs
    assert action.id == "a1" and action.label is TaskAction.Label.OPEN
    assert action.input_ids == [] and action.output_ids == ["q1"]
    assert q.identity == "k1" and q.spec.goal == "write" and q.state == "todo"
    assert tag.tasks == {"q1": q} and tag.identities == {"k1"}
    print("  ok: OPEN: {} -> q with a fresh identity")


def test_edit_and_update_return_new_versions_and_leave_history_alone():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec("write", "3 pages").to_dict(), "state": "todo"}))
    q1 = tag.tasks["q1"]

    edited = _realize(tag, ToolCall("edit", {"task": "q1", "spec": spec("write more", "5 pages").to_dict()}), 1)
    (q2,) = edited.outputs
    assert q2 is not q1 and q2.identity == "k1"
    assert q2.spec.goal == "write more" and q2.state == "todo"     # state carried over
    assert q1.spec.goal == "write" and q1.spec.rule == "3 pages"   # history untouched

    updated = _realize(tag, ToolCall("update", {"task": "q2", "state": "done"}), 2)
    (q3,) = updated.outputs
    assert q3.identity == "k1" and q3.spec == q2.spec and q3.state == "done"
    assert q2.state == "todo"
    assert updated.input_ids == ["q2"] and updated.output_ids == ["q3"]
    assert tag.identities == {"k1"} and len(tag.tasks) == 3

    with pytest.raises(ValueError, match="`state` is required"):
        tag.transition(ToolCall("update", {"task": "q3"}))
    with pytest.raises(ValueError, match="`spec` must be a spec mapping"):
        tag.transition(ToolCall("edit", {"task": "q3", "spec": "x"}))
    print("  ok: EDIT / UPDATE return fresh versions with the same identity; old nodes are immutable")


def test_split_requires_two_parts_and_applies_the_identity_rules():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec("survey").to_dict(), "state": "todo"}))
    _realize(tag, ToolCall("open", {"spec": spec("other").to_dict(), "state": "todo"}), 1)
    assert tag.identities == {"k1", "k2"}

    parts = [
        {"identity": "k1", "spec": spec("intro").to_dict(), "state": "todo"},   # keeps the input identity
        {"identity": None, "spec": spec("body").to_dict(), "state": "todo"},    # minted
        {"identity": "mine", "spec": spec("end").to_dict(), "state": "todo"},   # introduced
    ]
    with pytest.raises(ValueError, match="at least two"):
        tag.transition(ToolCall("split", {"task": "q1", "parts": parts[:1]}))
    with pytest.raises(ValueError, match="in use by another task"):
        tag.transition(ToolCall("split", {"task": "q1", "parts": [
            {"identity": "k2", "spec": spec().to_dict()},
            {"identity": None, "spec": spec().to_dict()},
        ]}))
    with pytest.raises(ValueError, match="distinct identities"):
        tag.transition(ToolCall("split", {"task": "q1", "parts": [
            {"identity": "k1", "spec": spec().to_dict()},          # two parts cannot both continue k1
            {"identity": "k1", "spec": spec().to_dict()},
        ]}))
    action = _realize(tag, ToolCall("split", {"task": "q1", "parts": parts}), 2)
    assert action.label is TaskAction.Label.SPLIT and action.input_ids == ["q1"]
    assert [q.identity for q in action.outputs] == ["k1", "k3", "mine"]
    assert [q.spec.goal for q in action.outputs] == ["intro", "body", "end"]
    assert len(tag.tasks) == 5
    print("  ok: SPLIT: q -> {q_j}, m >= 2; one continues the input identity at most, the rest are new")


def test_join_requires_two_inputs_and_applies_the_identity_rules():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec("a").to_dict(), "state": 1}))
    _realize(tag, ToolCall("open", {"spec": spec("b").to_dict(), "state": 2}), 1)
    _realize(tag, ToolCall("open", {"spec": spec("c").to_dict(), "state": 3}), 2)

    joined = _realize(tag, ToolCall("join", {
        "tasks": ["q1", "q2"], "identity": "k1", "spec": spec("ab").to_dict(), "state": 12,
    }), 3)
    (q4,) = joined.outputs
    assert joined.input_ids == ["q1", "q2"] and q4.identity == "k1" and q4.state == 12

    fresh = _realize(tag, ToolCall("join", {
        "tasks": ["q2", "q3"], "identity": None, "spec": spec("bc").to_dict(), "state": 23,
    }), 4)
    assert fresh.outputs[0].identity == "k4"    # minted
    # q2 was consumed twice: explicit branching, no mutation
    assert tag.tasks["q2"].state == 2

    with pytest.raises(ValueError, match="at least two"):
        tag.transition(ToolCall("join", {"tasks": ["q1"], "spec": spec().to_dict()}))
    with pytest.raises(ValueError, match="distinct task identities"):
        tag.transition(ToolCall("join", {"tasks": ["q2", "q2"], "spec": spec().to_dict()}))
    _realize(tag, ToolCall("open", {"spec": spec("d").to_dict(), "state": 4}), 5)     # q6: k5
    _realize(tag, ToolCall("update", {"task": "q6", "state": 44}), 6)                 # q7: a later version of k5
    with pytest.raises(ValueError, match="distinct task identities"):               # two versions of one identity
        tag.transition(ToolCall("join", {"tasks": ["q6", "q7"], "spec": spec().to_dict()}))
    with pytest.raises(ValueError, match="in use by another task"):
        tag.transition(ToolCall("join", {"tasks": ["q1", "q2"], "identity": "k3", "spec": spec().to_dict()}))
    print("  ok: JOIN: {q_j} -> q', m >= 2; an input may be consumed by several actions")


def test_a_consumed_version_branches():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec("a").to_dict(), "state": 1}))              # q1: k1
    _realize(tag, ToolCall("update", {"task": "q1", "state": 2}), 1)                        # q2 continues k1
    (q3,) = _realize(tag, ToolCall("update", {"task": "q1", "state": 3}), 2).outputs         # q1 is consumed: q3 branches
    assert q3.identity == "k2" and tag.tasks["q2"].identity == "k1"
    assert not tag.on_frontier("q1") and tag.on_frontier("q2") and tag.on_frontier("q3")
    (q4,) = _realize(tag, ToolCall("edit", {"task": "q1", "spec": spec("b").to_dict()}), 3).outputs
    assert q4.identity == "k3"                                                              # an edit branches the same way
    with pytest.raises(ValueError, match="not its frontier version"):                       # a part cannot continue k1 from q1
        tag.transition(ToolCall("split", {"task": "q1", "parts": [
            {"identity": "k1", "spec": spec().to_dict()}, {"identity": None, "spec": spec().to_dict()},
        ]}))
    parts = _realize(tag, ToolCall("split", {"task": "q1", "parts": [
        {"identity": None, "spec": spec().to_dict()}, {"identity": None, "spec": spec().to_dict()},
    ]}), 4)
    assert [q.identity for q in parts.outputs] == ["k4", "k5"]                              # every part fresh
    with pytest.raises(ValueError, match="not its frontier version"):                       # a join cannot continue k1 from q1
        tag.transition(ToolCall("join", {"tasks": ["q1", "q3"], "identity": "k1", "spec": spec().to_dict()}))
    (q7,) = _realize(tag, ToolCall("join", {"tasks": ["q1", "q3"], "identity": "k2", "spec": spec().to_dict()}), 5).outputs
    assert q7.identity == "k2"                                                              # q3 is k2's frontier version
    (q8,) = _realize(tag, ToolCall("join", {"tasks": ["q1", "q4"], "identity": None, "spec": spec().to_dict()}), 6).outputs
    assert q8.identity == "k6"                                                              # or a fresh identity
    heads = [q for q in tag.tasks.values() if tag.on_frontier(q.id)]
    assert [q.id for q in heads] == ["q2", "q5", "q6", "q7", "q8"]
    assert len({q.identity for q in heads}) == len(heads)                                   # one current version per identity
    print("  ok: an action continues an identity only from its frontier version; a consumed version branches")


def test_close_names_an_identity_and_consumes_nothing():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec().to_dict()}))
    action = _realize(tag, ToolCall("close", {"identity": "k1"}), 1)
    assert action.label is TaskAction.Label.CLOSE
    assert action.input_ids == [] and action.outputs == []          # Q_in = Q_out = {}
    assert len(tag.tasks) == 1 and len(tag.actions) == 2
    with pytest.raises(ValueError, match="not in the TAG"):
        tag.transition(ToolCall("close", {"identity": "k9"}))
    with pytest.raises(ValueError, match="requires `identity`"):
        tag.transition(ToolCall("close", {"task": "q1"}))
    print("  ok: CLOSE(k): Q_in = Q_out = {}, k must be in use")


def test_transition_rejects_unknown_inputs_and_non_task_tools():
    tag = TAGGraph()
    with pytest.raises(ValueError, match="not in the TAG"):
        tag.transition(ToolCall("update", {"task": "q9", "state": 1}))
    with pytest.raises(ValueError, match="not a task tool"):
        tag.transition(ToolCall("send", {"event": "e1", "to": []}))
    with pytest.raises(ValueError, match="already recorded"):
        tag.record(tag.actions["a1"] if tag.actions else _realize(tag, ToolCall("open", {"spec": spec().to_dict()})))
    print("  ok: unknown inputs and non-task tools are rejected before anything changes")


def test_attach_targets_an_exact_version_only():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec().to_dict(), "state": "todo"}))
    phi = tag.attach(Evidence(source="h1", payload={"check": "ok"}, target="q1"))
    assert phi.id == "phi1" and phi.created_at is not None and tag.evidence == {"phi1": phi}
    assert len(tag.actions) == 1      # no action node for an attachment

    _realize(tag, ToolCall("update", {"task": "q1", "state": "done"}), 1)
    assert [e.target for e in tag.evidence.values()] == ["q1"]   # the successor q2 inherits nothing
    with pytest.raises(ValueError, match="not in the TAG"):
        tag.attach(Evidence(source="h1", payload=None, target="q9"))
    with pytest.raises(ValueError, match="already recorded"):
        tag.attach(phi)
    again = tag.attach(Evidence(source="h1", payload={"check": "ok"}, target="q1"))
    assert again is phi and list(tag.evidence) == ["phi1"]          # Phi is a set of triples
    other = tag.attach(Evidence(source="h2", payload={"check": "ok"}, target="q1"))
    assert other.id == "phi2"                                        # a different source: a new triple
    print("  ok: attach adds to Phi only, on one exact version; a repeated triple leaves Phi unchanged")


def test_record_root_admits_an_injected_task_without_an_action():
    tag = TAGGraph()
    root = tag.record_root(Version(spec=spec("root"), state="todo"))
    assert root.id == "q1" and root.identity == "k1" and tag.tasks == {"q1": root} and tag.actions == {}
    given = tag.record_root(Version(spec=spec(), id="q7", identity="mine"))       # ids given are kept
    assert given.id == "q7" and tag.identities == {"k1", "mine"}
    with pytest.raises(ValueError, match="already in the TAG"):
        tag.record_root(root)
    print("  ok: an injected root task enters Q with no action")


def test_the_record_round_trips_on_its_own():
    tag = TAGGraph()
    _realize(tag, ToolCall("open", {"spec": spec("a").to_dict(), "state": 1}))
    _realize(tag, ToolCall("update", {"task": "q1", "state": 2}), 1)
    tag.attach(Evidence(source="h1", payload={"check": "ok"}, target="q2"))
    data = tag.to_dict()
    assert set(data) == {"versions", "actions", "evidence", "counters", "calls"}
    assert data["actions"]["a2"] == {"id": "a2", "issuer": "h1", "call_index": 1, "outputs": ["q2"]}
    back = TAGGraph.from_dict(data)
    assert list(back.tasks) == ["q1", "q2"] and back.actions["a2"].outputs == [back.tasks["q2"]]
    assert not back.on_frontier("q1") and back.on_frontier("q2")
    assert back.evidence["phi1"].target == "q2"
    (q3,) = _realize(back, ToolCall("open", {"spec": spec("b").to_dict()}), 2).outputs
    assert (q3.id, q3.identity) == ("q3", "k2")                                   # the counters continue
    assert "calls" not in tag.to_dict(calls=False) and "versions" in tag.to_dict(calls=False)   # in DIG-TAG the calls are D's
    print("  ok: a standalone TAG record carries its versions and calls and round-trips")


if __name__ == "__main__":
    from helpers import run_golden

    run_golden(globals(), "TAG golden tests")
