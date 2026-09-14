"""Golden tests for the record renderer: a TAG on its own draws its T
band, a DIG on its own D and Z, and the timeline places a TAG's moments.
Run with:  python tests/test_render.py"""

import pytest

from helpers import run_golden, spec, StubEnvironment  # first: it puts the repo root on sys.path for direct runs

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dig_tag.base import ToolCall  # noqa: E402
from dig_tag.dig import DIGParallelInterface, DIGEvent  # noqa: E402
from dig_tag.render import ActionTimeline, measure_record, render_record  # noqa: E402
from dig_tag.render.canvas import add_panel, new_figure, save  # noqa: E402
from dig_tag.tag import TAGParallelInterface, TAGGraph, TaskAction, Version, observed  # noqa: E402


def _tag() -> TAGParallelInterface:
    tag = TAGParallelInterface()
    (q1,) = observed(tag.open("planner", spec("survey"), state="todo"))
    a, b = observed(tag.split("planner", q1, [(q1.identity, spec("intro"), "todo"), (None, spec("body"), "todo")]))
    tag.attach("w1", a, {"hits": 3})
    tag.update("w1", a, "drafted")
    tag.update("w2", a, "drafted too")            # a consumed version: branches
    tag.close("planner", b)
    return tag


def test_a_tag_alone_draws_its_t_band(tmp_path):
    tag = _tag()
    size = measure_record(tag)
    assert list(size.bands) == ["T"] and size.h > 3 * 0.2               # three identity lanes
    fig = new_figure(size.w + 0.1, size.h + 0.1)
    render_record(add_panel(fig, 0.05, 0.05, size.w, size.h), tag)
    out = save(fig, "tag", tmp_path, preview=True)
    assert (tmp_path / "tag.pdf").stat().st_size > 1000 and (tmp_path / "tag.png").exists()
    with pytest.raises(ValueError, match="no D band"):
        measure_record(tag, bands=("D",))
    with pytest.raises(TypeError, match="a record to draw"):
        measure_record(object())
    print(f"  ok: a TAG on its own draws its T band ({out['pdf']})")


def test_a_dig_alone_draws_d_and_z(tmp_path):
    dt = DIGParallelInterface(agents=["alice", "bob"], environment=StubEnvironment())
    dt.inject(DIGEvent(payload={"text": "start"}), to=["alice"])
    with dt.open_activation("alice") as h:
        (feedback,) = dt.act(h, "echo", q="x")
        dt.send(h, feedback, to=["bob"])
    with dt.open_activation("bob") as h:
        dt.act(h, "echo", q="y")
    size = measure_record(dt)
    assert list(size.bands) == ["Z", "D"]                                # from the bottom: no T to offer
    fig = new_figure(size.w + 0.1, size.h + 0.1)
    render_record(add_panel(fig, 0.05, 0.05, size.w, size.h), dt, context=True)
    save(fig, "dig", tmp_path)
    assert (tmp_path / "dig.pdf").stat().st_size > 1000
    with pytest.raises(ValueError, match="no T band"):
        render_record(add_panel(new_figure(1, 1), 0, 0, 1, 1), dt, bands=("T",))
    print("  ok: a DIG draws D and Z alone")


def test_the_timeline_places_a_tags_moments_in_order():
    tag = _tag()
    line = ActionTimeline(tag.tag)
    actions = list(tag.tag.actions.values())
    xs = [line.action(a) for a in actions]
    assert xs == sorted(xs) and xs[0] == 1.0 and not line.root_slot        # one slot per moment, no root
    (phi,) = tag.tag.evidence.values()
    (attach_x, phi_drawn, agent) = line.attachments()[0]
    assert phi_drawn is phi and agent is None and xs[1] < attach_x < xs[2]    # the attach between the split and the update
    assert line.extent == max(xs)
    clocked = ActionTimeline(tag.tag, clock=1.0)
    assert clocked.action(actions[-1]) >= clocked.action(actions[0])         # wall-clock order agrees
    burst = TAGParallelInterface()
    burst.open("a", spec("x"), state=0)
    burst.update("a", "q1", 1)
    burst.update("a", "q1", 2)
    for action in burst.tag.actions.values():
        action.call.at = 100.0                                              # three actions in the same instant
    bx = [ActionTimeline(burst.tag, clock=1.0).action(a) for a in burst.tag.actions.values()]
    assert bx == [bx[0]] * 3                                                # time is strict: one instant, one x
    t0 = actions[0].call.at
    fixed = ActionTimeline(tag.tag, clock=0.5, origin=t0 - 1.0, horizon=t0 + 9.0)
    assert fixed.at(t0) == 0.5 + 2.0 and fixed.extent == 0.5 + 20.0 and fixed.span == 10.0   # the axis is the horizon's
    graph = TAGGraph()
    root = graph.record_root(Version(spec=spec("root")))
    assert ActionTimeline(graph).root_slot
    call = ToolCall("update", {"task": root.id, "state": 1})
    call.index, call.at, call.returned = 0, 1.0, graph.transition(call)
    graph.record(TaskAction(issuer="a", call=call))
    assert ActionTimeline(graph).action(graph.actions["a1"]) == 1.5          # the root slot before it
    print("  ok: the timeline places a TAG's actions and attachments in the order they happened")


if __name__ == "__main__":
    run_golden(globals(), "render golden tests")
