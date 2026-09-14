"""Golden tests for the DIG's record types and the contracts each
validates on its own: DIGEvent with its two provenances, and DIGActivation
with its availability.
Run with:  python tests/test_dig_nodes.py"""

import pytest

from dig_tag.base import ToolCall
from dig_tag.dig import ENVIRONMENT_ORIGIN, DIGActivation, DIGEvent
from helpers import run_golden  # first: it puts the repo root on sys.path for direct runs


def test_event_validates_its_own_contract():
    event = DIGEvent(payload={"text": "hi"}, origin="alice")
    assert not event.recorded and not event.is_root

    with pytest.raises(TypeError, match="payload must be a dict"):
        DIGEvent(payload="hi")
    with pytest.raises(TypeError, match="origin must be a str"):
        DIGEvent(origin=3)
    with pytest.raises(TypeError, match="producer must be a DIGEvent.Producer"):
        DIGEvent(producer=("h1", 0))
    with pytest.raises(TypeError, match="call_index"):
        DIGEvent.Producer("h1", -1)
    print("  ok: DIGEvent validates payload, origin, and producer")


def test_event_two_provenances_and_record_round_trip():
    produced = DIGEvent(
        payload={"n": 1},
        origin=ENVIRONMENT_ORIGIN,           # declared: the environment
        id="e3",
        producer=DIGEvent.Producer("h1", 2),  # structural: call 2 of h1
        created_at=1.5,
        metadata={"note": "x"},
    )
    assert produced.recorded and not produced.is_root
    assert str(produced.producer) == "h1:2"
    root = DIGEvent(id="e1", origin="external")
    assert root.is_root

    data = produced.to_dict()
    assert data["producer"] == {"activation_id": "h1", "call_index": 2}
    back = DIGEvent.from_dict(data)
    assert type(back) is DIGEvent
    assert back.id == "e3" and back.origin == ENVIRONMENT_ORIGIN
    assert back.producer == DIGEvent.Producer("h1", 2)
    assert back.payload == {"n": 1} and back.metadata == {"note": "x"}
    print("  ok: producer (structural) and origin (declared) are two fields; record round-trips")


def test_activation_availability_is_inputs_plus_earlier_returns():
    e1, e2 = DIGEvent(id="e1"), DIGEvent(id="e2")
    r1, r2 = DIGEvent(id="e3"), DIGEvent(id="e4")
    h = DIGActivation(
        agent_id="alice",
        inputs=[e1, e2],
        calls=[
            ToolCall("search", index=0, returned=[r1]),
            ToolCall("send", {"event": "e3", "to": ["bob"]}, index=1, returned=[]),
            ToolCall("search", index=2, returned=[r2, e1]),   # e1 passes through: no duplicate
        ],
        id="h1",
        started_at=0.0,
    )
    assert h.is_open
    assert h.input_ids == ["e1", "e2"]
    assert h.available_ids(before=0) == ["e1", "e2"]          # Avail(h, 0) = E_h
    assert h.available_ids(before=1) == ["e1", "e2", "e3"]    # plus E_0
    assert h.available_ids(before=2) == ["e1", "e2", "e3"]    # send returned nothing
    assert h.available_ids() == ["e1", "e2", "e3", "e4"]      # the next call sees everything
    assert [e.id for e in h.returned] == ["e3", "e4", "e1"]

    h.ended_at = 1.0
    assert not h.is_open
    assert h.error is None
    h.mark_error(RuntimeError("boom"))
    assert h.error == {"type": "RuntimeError", "message": "boom"}

    with pytest.raises(TypeError, match="agent_id must be a non-empty str"):
        DIGActivation(agent_id="")
    with pytest.raises(TypeError, match="inputs must contain DIGEvent"):
        DIGActivation(agent_id="a", inputs=["e1"])
    with pytest.raises(TypeError, match="calls must contain ToolCall"):
        DIGActivation(agent_id="a", calls=[{}])
    print("  ok: Avail(h, r) is E_h plus the returns of calls before r")


if __name__ == "__main__":
    run_golden(globals(), "DIG node golden tests")
