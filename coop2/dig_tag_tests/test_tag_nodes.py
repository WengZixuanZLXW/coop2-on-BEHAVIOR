"""Golden tests for the TAG's record types and the contracts each
validates on its own: TaskSpec, Version, Evidence targeting an exact
version, and the TaskAction derived from a call.
Run with:  python tests/test_tag_nodes.py"""

import pytest

from dig_tag.base import ToolCall
from dig_tag.tag import Evidence, TaskAction, TaskSpec, Version, close_target, task_input_ids
from helpers import run_golden, spec  # first: it puts the repo root on sys.path for direct runs


def test_version_carries_identity_spec_and_state():
    version = Version(spec=spec("write"), state="todo", identity="k1", id="q1")
    assert version.recorded and version.spec.goal == "write" and version.spec.rule is None
    assert not Version(spec=spec()).recorded

    with pytest.raises(TypeError, match="spec must be a TaskSpec"):
        Version(state="todo")
    with pytest.raises(TypeError, match="identity must be a non-empty str"):
        Version(spec=spec(), identity="")
    with pytest.raises(Exception):
        version.spec.goal = "changed"   # TaskSpec is frozen

    data = version.to_dict()
    assert data["identity"] == "k1" and data["spec"] == {"goal": "write", "rule": None}
    back = Version.from_dict(data)
    assert back.identity == "k1" and back.spec == TaskSpec("write") and back.state == "todo"
    assert Version(spec=spec()) != Version(spec=spec())   # each version is a fresh token
    print("  ok: Version carries identity, spec, and state; the record round-trips")


def test_evidence_targets_an_exact_recorded_version():
    version = Version(spec=spec(), id="q2")
    phi = Evidence(source="alice", payload={"score": 1}, target=version)
    assert phi.target == "q2"   # reduced to the exact version's id
    assert Evidence(source="h1", payload=None, target="q2").target == "q2"

    with pytest.raises(ValueError, match="RECORDED version"):
        Evidence(source="alice", payload=None, target=Version(spec=spec()))
    with pytest.raises(TypeError, match="target must be a version id"):
        Evidence(source="alice", payload=None, target=7)
    with pytest.raises(TypeError, match="source must be a non-empty str"):
        Evidence(source="", payload=None, target="q2")

    back = Evidence.from_dict(phi.to_dict())
    assert back.source == "alice" and back.target == "q2" and back.payload == {"score": 1}
    print("  ok: Evidence targets one exact task version by id")


def test_task_action_derives_inputs_and_outputs_from_its_call():
    q2 = Version(spec=spec(), id="q2", identity="k1")
    call = ToolCall("update", {"task": "q1", "state": "done"}, index=0)
    action = TaskAction(issuer="h1", call=call, id="a1", outputs=[q2])
    assert action.label is TaskAction.Label.UPDATE
    assert action.input_ids == ["q1"] and action.output_ids == ["q2"]      # Q_in from the args, Q_out stored
    assert action.outputs == [q2] and call.returned == []                   # the call's returns are D's business
    assert action.to_dict() == {"id": "a1", "issuer": "h1", "call_index": 0, "outputs": ["q2"]}
    with pytest.raises(TypeError, match="outputs must contain Version"):
        TaskAction(issuer="h1", call=call, outputs=["q2"])

    assert task_input_ids(ToolCall("open", {})) == []
    assert task_input_ids(ToolCall("join", {"tasks": ["q1", "q2"]})) == ["q1", "q2"]
    assert task_input_ids(ToolCall("close", {"identity": "k1"})) == []
    assert close_target(ToolCall("close", {"identity": "k1"})) == "k1"
    with pytest.raises(ValueError, match="requires `identity`"):
        close_target(ToolCall("close", {"task": "q1"}))
    with pytest.raises(ValueError, match="requires `task`"):
        task_input_ids(ToolCall("edit", {}))
    with pytest.raises(ValueError, match="requires `tasks`"):
        task_input_ids(ToolCall("join", {"tasks": "q1"}))
    with pytest.raises(ValueError, match="must be a task tool call"):
        TaskAction(issuer="h1", call=ToolCall("send", {}))
    with pytest.raises(TypeError, match="issuer must be a non-empty str"):
        TaskAction(issuer="", call=call)
    print("  ok: TaskAction derives Q_in from args and Q_out from the returns")


if __name__ == "__main__":
    run_golden(globals(), "TAG node golden tests")
