"""Golden tests for what both graphs are built on -- the tool call and the
vocabulary the packages declare, and sequential ids.
Run with:  python tests/test_base.py"""

import pytest

from dig_tag.base import Minter, ToolCall
from dig_tag.dig import SEND_TOOL, DIGEvent
from dig_tag.tag import ATTACH_TOOL, TASK_TOOLS
from helpers import run_golden  # first: it puts the repo root on sys.path for direct runs


def test_tool_call_family_derives_from_the_declared_vocabulary():
    """The DIG declares send, the TAG declares the six task tools and
    attach; a call to any other name is a call on the environment."""
    assert TASK_TOOLS == ("open", "edit", "update", "split", "join", "close")
    assert (SEND_TOOL, ATTACH_TOOL) == ("send", "attach")
    for tool in TASK_TOOLS:
        assert ToolCall(tool).label is ToolCall.Label.TASK
    for tool in (SEND_TOOL, ATTACH_TOOL):
        assert ToolCall(tool).label is ToolCall.Label.INFORMATION
    assert ToolCall("search", {"q": "x"}).label is ToolCall.Label.ENVIRONMENT

    call = ToolCall("search", {"q": "x"})
    assert not call.recorded and call.returned_ids == []
    with pytest.raises(TypeError, match="tool must be a non-empty str"):
        ToolCall("")
    with pytest.raises(TypeError, match="returned must contain"):
        ToolCall("search", returned=["e1"])
    print("  ok: the tool family derives from the names the packages declare")


def test_tool_call_record_round_trip_relinks_returned_nodes():
    e1 = DIGEvent(id="e1")
    call = ToolCall("search", {"q": "x"}, index=0, returned=[e1], at=2.0)
    data = call.to_dict()
    assert data == {"tool": "search", "args": {"q": "x"}, "index": 0, "returned": ["e1"], "at": 2.0}
    back = ToolCall.from_dict(data, {"e1": e1})
    assert back.returned[0] is e1 and back.index == 0
    with pytest.raises(ValueError, match="unknown"):
        ToolCall.from_dict(data, {})
    print("  ok: a call's returned nodes are referenced by id and re-linked")


def test_minter_counts_on_and_skips_taken_ids():
    minter = Minter("e")
    assert minter.next({}) == "e1" and minter.next({}) == "e2"
    assert minter.next({"e3", "e4"}) == "e5"              # a loaded record may have spent those
    assert minter.counter == 5
    assert Minter("q", 7).next({}) == "q8"                 # continues from the recorded counter
    print("  ok: ids are minted in sequence, past any id already taken")


if __name__ == "__main__":
    run_golden(globals(), "base golden tests")
