"""Golden tests for the DIG representation -- the four funnels (root, open,
append, make_available), availability, the cumulative inbox, and the
record. The graph stores; it answers nothing beyond its own state.
Run with:  python tests/test_dig.py"""

import pytest

from dig_tag.base import ToolCall
from dig_tag.dig import ENVIRONMENT_ORIGIN, EXTERNAL_ORIGIN, DIGEvent, DIGGraph


def test_root_events_have_no_producer_and_reach_inboxes():
    dig = DIGGraph()
    root = dig.record_root(DIGEvent(payload={"text": "start"}), to=["alice", "bob"])

    assert root.id == "e1" and root.is_root and root.producer is None
    assert root.origin == EXTERNAL_ORIGIN
    assert root.created_at is not None
    assert dig.inbox_events("alice") == [root] and dig.inbox_events("bob") == [root]
    assert dig.pending("alice") == [root]

    declared = dig.record_root(DIGEvent(origin="user"), to=["alice"])
    assert declared.origin == "user"   # a declared origin is kept
    with pytest.raises(ValueError, match="already recorded"):
        dig.record_root(root, to=["bob"])
    print("  ok: a root event has no producer, a declared origin, and reaches the named inboxes")


def test_open_fixes_inputs_to_the_whole_inbox():
    dig = DIGGraph()
    e1 = dig.record_root(DIGEvent(), to=["alice"])
    h1 = dig.open("alice")

    assert h1.id == "h1" and h1.is_open and h1.inputs == [e1]
    assert dig.pending("alice") == []   # presented now
    e2 = dig.record_root(DIGEvent(), to=["alice"])   # arrives while h1 is open
    assert h1.inputs == [e1]            # E_h was fixed at open
    assert dig.pending("alice") == [e2]
    dig.end(h1)
    h2 = dig.open("alice")
    assert h2.inputs == [e1, e2]        # the inbox accumulates: e1 is presented again
    assert dig.pending("alice") == []
    print("  ok: E_h is the agent's whole inbox at open; later arrivals wait for the next activation")


def test_own_returns_join_the_next_context_but_are_not_pending():
    dig = DIGGraph()
    e1 = dig.record_root(DIGEvent(), to=["alice"])
    h1 = dig.open("alice")
    out = DIGEvent()
    dig.append(h1, ToolCall("say"), returned=[out], origin="alice")
    dig.end(h1)
    assert dig.pending("alice") == []                    # producing is not an arrival
    assert dig.context("alice") == [e1, out]             # but the return is in reach
    h2 = dig.open("alice")
    assert h2.inputs == [e1, out]
    late = dig.record_root(DIGEvent(), to=["alice"])
    assert dig.pending("alice") == [late]
    print("  ok: own returns enter the next activation's inputs without waking the agent")


def test_one_open_activation_per_agent_but_overlap_across_agents():
    dig = DIGGraph()
    h_alice = dig.open("alice")
    h_bob = dig.open("bob")             # overlaps in time with alice's
    assert h_alice.is_open and h_bob.is_open
    assert dig.open_activation_of("alice") is h_alice
    with pytest.raises(ValueError, match="already has an open activation"):
        dig.open("alice")
    dig.end(h_alice)
    assert dig.open_activation_of("alice") is None
    assert dig.open("alice").id == "h3"
    with pytest.raises(TypeError, match="agent_id must be a non-empty str"):
        dig.open("")
    print("  ok: activations overlap across agents; an agent fires one at a time")


def test_append_inserts_fresh_returns_with_this_call_as_producer():
    dig = DIGGraph()
    h = dig.open("alice")
    feedback = DIGEvent(payload={"hits": 3})
    more = DIGEvent(payload={"hits": 4})
    call = dig.append(
        h,
        ToolCall("search", {"q": "x"}),
        returned=[feedback, more],
        origin=ENVIRONMENT_ORIGIN,
    )

    assert call.index == 0 and call.at is not None and call.returned == [feedback, more]
    assert feedback.id == "e1" and more.id == "e2"          # one store, ids in order
    assert dig.events["e1"] is feedback and dig.events["e2"] is more
    assert feedback.producer == DIGEvent.Producer("h1", 0) == more.producer
    assert feedback.origin == ENVIRONMENT_ORIGIN
    assert h.calls == [call] and h.available_ids() == ["e1", "e2"]

    declared = DIGEvent(origin="alice")
    dig.append(h, ToolCall("note"), returned=[declared], origin=ENVIRONMENT_ORIGIN)
    assert declared.origin == "alice"                        # a declared origin is kept
    assert declared.producer.call_index == 1

    with pytest.raises(ValueError, match="already recorded"):
        dig.append(h, ToolCall("search"), returned=[feedback], origin="x")
    with pytest.raises(ValueError, match="already recorded"):
        dig.append(h, call, returned=[], origin="x")
    dig.end(h)
    with pytest.raises(ValueError, match="is closed"):
        dig.append(h, ToolCall("search"), returned=[], origin="x")
    with pytest.raises(ValueError, match="not recorded"):
        dig.append(DIGGraph().open("alice"), ToolCall("search"), returned=[], origin="x")
    print("  ok: append realizes one call-return pair; returns are fresh, producer = (h, r)")


def test_make_available_is_send_without_copying():
    dig = DIGGraph()
    h = dig.open("alice")
    event = DIGEvent()
    dig.append(h, ToolCall("say"), returned=[event], origin="alice")

    assert dig.make_available(event, ["bob", "carol"]) == ["bob", "carol"]
    assert dig.inbox_events("bob") == [event] and dig.inbox_events("bob")[0] is event   # the same node
    assert dig.make_available(event, ["bob"]) == []                        # already available
    assert dig.inbox_events("bob") == [event] and list(dig.inbox["bob"]) == ["e1"]
    with pytest.raises(ValueError, match="not recorded"):
        dig.make_available(DIGEvent(), ["bob"])
    with pytest.raises(TypeError, match="recipient must be"):
        dig.make_available(event, [""])
    print("  ok: make_available puts the existing event in inboxes without copying")


def test_expect_available_enforces_avail():
    dig = DIGGraph()
    seen = dig.record_root(DIGEvent(), to=["alice"])
    unseen = dig.record_root(DIGEvent(), to=["bob"])
    h = dig.open("alice")
    produced = DIGEvent()
    dig.append(h, ToolCall("say"), returned=[produced], origin="alice")

    dig.expect_available(h, [seen.id, produced.id])   # inputs + earlier returns
    with pytest.raises(ValueError, match="cannot name e2"):
        dig.expect_available(h, [unseen.id])
    print("  ok: a call may name only Avail(h, r)")


def test_end_stamps_and_keeps_what_was_issued():
    dig = DIGGraph()
    h = dig.open("alice")
    dig.append(h, ToolCall("say"), returned=[DIGEvent()], origin="alice")
    dig.end(h, error=RuntimeError("boom"))

    assert not h.is_open and h.ended_at is not None
    assert h.error == {"type": "RuntimeError", "message": "boom"}
    assert len(h.calls) == 1 and "e1" in dig.events    # the record is prefix-realized
    with pytest.raises(ValueError, match="is closed"):
        dig.end(h)
    print("  ok: ending stamps t' and keeps everything issued, error or not")


def test_record_round_trip():
    dig = DIGGraph()
    dig.record_root(DIGEvent(payload={"text": "root"}), to=["alice"])
    h = dig.open("alice")
    out = DIGEvent(payload={"x": 1})
    dig.append(h, ToolCall("say", {"text": "hi"}), returned=[out], origin="alice")
    dig.make_available(out, ["bob"])
    dig.end(h)

    data = dig.to_dict()
    assert set(data) == {"created_at", "events", "activations", "inbox", "routed", "counters"}
    assert {agent: list(arrivals) for agent, arrivals in data["inbox"].items()} == {"alice": ["e1"], "bob": ["e2"]}
    import json
    json.dumps(data)

    back = DIGGraph.from_dict(data)
    assert back.events["e1"].payload == {"text": "root"} and back.events["e1"].is_root
    assert back.activations["h1"].inputs[0] is back.events["e1"]
    assert back.activations["h1"].calls[0].returned[0] is back.events["e2"]
    assert back.inbox_events("bob")[0] is back.events["e2"]
    assert back.pending("bob") == [back.events["e2"]]
    assert back.open("carol").id == "h2"   # counters continue
    print("  ok: the DIG record round-trips with one object per event")


if __name__ == "__main__":
    from helpers import run_golden

    run_golden(globals(), "DIG golden tests")
