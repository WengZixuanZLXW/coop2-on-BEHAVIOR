"""Golden tests for the DIGTAG interface -- construction, the two root
funnels, the firing lifecycle, the tool methods end to end with what they
hand back, the return policies, subscribe, and the record round trip.
Run with:  python tests/test_interface.py"""

import pytest

from dig_tag import views
from dig_tag.base import ToolCall
from dig_tag.dig import DIGEvent, Environment, NullEnvironment
from dig_tag.interface import DIGTAG, SCHEMA, TAG_ORIGIN
from dig_tag.tag import Version, current, observed, produced, returns_for
from helpers import StubEnvironment, spec


def test_construction_registers_agents_and_checks_the_environment():
    dt = DIGTAG(agents=["a", "b"])
    assert dt.agents == ["a", "b"] and isinstance(dt.environment, NullEnvironment)
    dt.add_agent("c")
    dt.add_agent("a")                       # idempotent
    assert dt.agents == ["a", "b", "c"]
    with pytest.raises(TypeError, match="agent id must be a non-empty str"):
        dt.add_agent("")
    with pytest.raises(ValueError, match="unknown agent id"):
        dt.pending("zed")

    class Clashing(Environment):
        tools = frozenset({"open", "search"})

        def step(self, activation, call):
            return []

    with pytest.raises(ValueError, match="collide with the interface"):
        DIGTAG(environment=Clashing())
    with pytest.raises(TypeError, match="expects a Environment"):
        DIGTAG(environment=object())
    print("  ok: agents are I; environment tool names must not be reserved")


def test_the_root_funnels():
    dt = DIGTAG(agents=["planner"])
    root = dt.inject_task(Version(spec=spec("survey"), state="todo"), to=["planner"])
    assert dt.tag.tasks["q1"].identity == "k1" and dt.tag.actions == {}            # T: the version, no action
    assert root.id == "e1" and root.is_root and root.origin == TAG_ORIGIN         # D: the observation, a root event
    assert root.payload["open"] == ["k1"] and [q.id for q in observed(root.payload)] == ["q1"]
    assert "q1" not in dt.dig.events and dt.pending("planner") == [root]

    note = dt.inject(DIGEvent(payload={"text": "hint"}), to=["planner"], origin="user")
    assert note.origin == "user" and note.id == "e2" and len(dt.tag.tasks) == 1
    print("  ok: inject_task admits a version to T and hands the observation to D; inject is D's own funnel")


def test_open_activation_is_the_firing_lifecycle():
    dt = DIGTAG(agents=["alice"])
    updates = []
    dt.subscribe(lambda: updates.append(len(dt.dig.activations)))
    dt.inject(DIGEvent(), to=["alice"])

    with dt.open_activation("alice") as h:
        assert h.id == "h1" and h.is_open and dt.dig.activations["h1"] is h   # in H at open
        assert h.inputs == dt.inbox("alice")
        dt.open_task(h, spec("x"))
    assert not h.is_open and h.ended_at is not None and h.error is None
    assert updates == [0, 1, 1, 1]      # inject, open, issue, end

    with pytest.raises(RuntimeError, match="boom"):
        with dt.open_activation("alice") as h2:
            dt.open_task(h2, spec("y"))
            raise RuntimeError("boom")
    assert not h2.is_open and h2.error["message"] == "boom"
    assert len(h2.calls) == 1 and "q2" in dt.tag.tasks     # what was issued stays
    assert dt.dig.open_activation_of("alice") is None       # a retry can open again
    with pytest.raises(ValueError, match="unknown agent id"):
        with dt.open_activation("zed"):
            pass
    print("  ok: open_activation enters H on entry, closes on exit, keeps issued calls on error")


def test_open_and_end_interleave_across_agents():
    dt = DIGTAG(agents=["a", "b"])
    root = dt.inject(DIGEvent(), to=["a", "b"])
    ha = dt.open("a")                       # a superstep: both start on the same inputs
    hb = dt.open("b")
    assert ha.is_open and hb.is_open and ha.inputs == [root] and hb.inputs == [root]
    ea = dt.open_task(ha, spec("from a"))
    dt.send(ha, ea, to=["b"])
    dt.end(ha)
    assert hb.inputs == [root]              # b's inputs were fixed when it opened
    assert dt.pending("b") == [ea]          # the send waits for b's next activation
    dt.end(hb, error=RuntimeError("late"))
    assert not hb.is_open and hb.error["message"] == "late"
    with pytest.raises(ValueError, match="is closed"):
        dt.end(hb)
    print("  ok: open/end is the explicit lifecycle; open_activation wraps it")


def test_issue_rejects_reused_calls_and_closed_activations():
    dt = DIGTAG(agents=["alice"], environment=StubEnvironment())
    with dt.open_activation("alice") as h:
        call = ToolCall("echo", {"n": 1})
        dt.issue(h, call)
        with pytest.raises(ValueError, match="one occurrence"):
            dt.issue(h, call)
    with pytest.raises(ValueError, match="is closed"):
        dt.issue(h, ToolCall("echo"))
    with pytest.raises(TypeError, match="expects a ToolCall"):
        with dt.open_activation("alice") as h2:
            dt.issue(h2, {"tool": "echo"})
    print("  ok: a ToolCall is one occurrence; closed activations issue nothing")


def test_tool_methods_end_to_end():
    dt = DIGTAG(agents=["planner", "worker", "verifier"], environment=StubEnvironment())
    dt.inject_task(Version(spec=spec("survey", "3 sections"), state="todo"), to=["planner"])

    with dt.open_activation("planner") as h:
        edited = dt.edit(h, "q1", spec("survey", "2 sections"))            # what D recorded: the observation
        (q2,) = observed(edited.payload)
        assert q2.id == "q2" and q2.identity == "k1" and q2.spec.rule == "2 sections"
        parts = dt.split(h, q2, [
            ("k1", spec("intro"), "todo"),      # keeps the identity
            (None, spec("body"), "todo"),       # minted
        ])
        assert [q.id for q in observed(parts.payload)] == ["q3", "q4"]
        dt.send(h, parts, to=["worker"])
    assert [dt.tag.tasks[q].identity for q in ("q2", "q3", "q4")] == ["k1", "k1", "k2"]

    with dt.open_activation("worker") as h:
        assert h.inputs == [parts]
        intro, body = observed(h.inputs[0].payload)
        (hits,) = dt.act(h, "echo", query="intro")
        done_intro = dt.update(h, intro, state={"draft": "..."})
        dt.attach(h, current(done_intro.payload, "k1"), {"hits": hits.id})
        done_body = dt.update(h, body, state={"draft": "..."})
        dt.send(h, done_body, to=["verifier"])                             # the latest observation: both drafts

    with dt.open_activation("verifier") as h:
        drafts = observed(h.inputs[0].payload)
        assert [q.id for q in drafts] == ["q5", "q6"]
        dt.join(h, drafts, spec("survey"), state="verified", identity="k1")
        dt.close(h, "k1")

    assert list(dt.tag.tasks) == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    labels = [action.label.value for action in dt.tag.actions.values()]
    assert labels == ["edit", "split", "update", "update", "join", "close"]
    assert [call.tool for call in dt.dig.activations["h2"].calls] == ["echo", "update", "attach", "update", "send"]
    assert {e.origin for e in dt.dig.events.values()} == {TAG_ORIGIN, "environment"}
    assert all(len(c.returned) == 1 for h in dt.dig.activations.values() for c in h.calls if c.tool != "send")
    assert dt.tag.evidence["phi1"].target == "q5" and not set(dt.tag.tasks) & set(dt.dig.events)

    with pytest.raises(TypeError, match="split part must be"):
        with dt.open_activation("planner") as h:
            dt.split(h, "q1", [spec("a"), spec("b")])
    with pytest.raises(TypeError, match="task spec must be a TaskSpec"):
        with dt.open_activation("planner") as h:
            dt.open_task(h, {"goal": "x"})
    with pytest.raises(ValueError, match="must be a recorded event"):
        with dt.open_activation("planner") as h:
            dt.send(h, DIGEvent(), to=["worker"])
    print("  ok: the tool methods realize the task tools and U_phi end to end; D holds what they handed back")


def test_returns_are_the_tools_information():
    """Every TAG tool hands back the observation unless a return policy
    says otherwise for it; observe hands it back without changing T."""
    dt = DIGTAG(agents=["a"])
    with dt.open_activation("a") as h:
        opened = dt.open_task(h, spec("x"), state=0)
        assert opened.payload == {"open": ["k1"], "frontier": [dt.tag.tasks["q1"].to_dict()]}
        seen = dt.observe(h)
        assert seen.payload == opened.payload and seen.origin == TAG_ORIGIN
        assert h.calls[1].tool == "observe" and len(dt.tag.actions) == 1     # recorded in D, nothing in T
        closed = dt.close(h, "k1")
        assert closed.payload == {"open": [], "frontier": []}

    custom = DIGTAG(agents=["a"], returns={"open": produced, "attach": produced})
    with custom.open_activation("a") as h:
        opened = custom.open_task(h, spec("y"), state=1)
        assert opened.payload == {"versions": [custom.tag.tasks["q1"].to_dict()]}
        assert custom.update(h, "q1", state=2).payload["frontier"][0]["id"] == "q2"   # the others as by default
        assert custom.attach(h, "q2", {"n": 1}).payload == {"evidence": custom.tag.evidence["phi1"].to_dict()}
        assert custom.close(h, "k1").payload == {"open": [], "frontier": []}
    with pytest.raises(ValueError, match="no such TAG tool"):
        returns_for({"teleport": produced})
    print("  ok: a tool's return is a policy: the observation by default, anything from T by override")


def test_record_round_trip(tmp_path):
    dt = DIGTAG(agents=["planner", "worker"], environment=StubEnvironment())
    dt.inject_task(Version(spec=spec("survey"), state="todo"), to=["planner"])
    with dt.open_activation("planner") as h:
        parts = dt.split(h, "q1", [(None, spec("a"), 1), (None, spec("b"), 2)])
        dt.act(h, "echo", n=1)
        dt.attach(h, "q2", {"why": "x"})
        dt.send(h, parts, to=["worker"])
    with dt.open_activation("worker") as h:
        dt.update(h, "q2", state="done")

    data = dt.to_dict()
    assert data["schema"] == SCHEMA and data["agents"] == ["planner", "worker"]
    assert data["environment_tools"] == ["echo", "fail", "relay"]
    assert set(data["tag"]) == {"versions", "actions", "evidence", "counters"}   # T's calls are D's
    path = dt.save_json(tmp_path / "run" / "dig_tag.json")
    back = DIGTAG.load_json(path)

    assert back.agents == dt.agents and sorted(back.environment.tools) == ["echo", "fail", "relay"]
    assert list(back.dig.events) == list(dt.dig.events) and list(back.tag.tasks) == list(dt.tag.tasks)
    assert back.tag.tasks["q2"].identity == "k2" and back.tag.tasks["q2"].spec.goal == "a"
    assert back.tag.actions["a1"].call is back.dig.activations["h1"].calls[0]
    assert back.tag.actions["a1"].output_ids == ["q2", "q3"]
    assert back.tag.evidence["phi1"].target == "q2"
    assert back.inbox("worker") == [back.dig.events[parts.id]]
    assert back.to_dict() == data
    with pytest.raises(ValueError, match="finished run"):
        with back.open_activation("planner") as h:
            back.act(h, "echo")               # a loaded record realizes nothing
    with pytest.raises(ValueError, match="unsupported schema"):
        DIGTAG.from_dict({"schema": "other"})
    print("  ok: the DT record round-trips through JSON")


def test_route_presents_without_a_send():
    dt = DIGTAG(agents=["alice", "bob"])
    with dt.open_activation("alice") as h:
        e = dt.open_task(h, spec("x"))
    assert dt.route(e, to=["bob"], by="runtime") == ["bob"]
    assert dt.route(e, to=["bob"], by="runtime") == []            # already there
    with dt.open_activation("bob") as h_bob:
        assert h_bob.inputs == [e] and views.delivery(dt, h_bob, e) == "runtime"
        assert h_bob.calls == []
    assert views.interaction_edges(dt) == set()                      # no send: not an interaction
    assert ("alice", "bob") in views.flow_graph(dt).edge_set         # but a presentation
    with pytest.raises(ValueError, match="not in the DIG"):
        dt.route("e9", to=["bob"])
    print("  ok: route makes an event available on the runtime's behalf; sends alone draw interaction edges")


if __name__ == "__main__":
    from helpers import run_golden

    run_golden(globals(), "interface golden tests")
