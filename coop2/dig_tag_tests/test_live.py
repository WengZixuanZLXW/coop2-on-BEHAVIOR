"""Golden tests for the live figure: the axis that starts at five seconds
and doubles when outgrown, the fit when the run is over, and the three
ways a run drives it -- threads, asyncio, and inline on the main thread.
Run with:  python tests/test_live.py"""

import threading

import pytest

from helpers import run_golden, spec, StubEnvironment  # first: it puts the repo root on sys.path for direct runs

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dig_tag.dig import DIGEvent  # noqa: E402
from dig_tag.render import LiveFigure, measure_record  # noqa: E402
from dig_tag.tag import TAGParallelInterface  # noqa: E402


def test_a_live_axis_starts_at_five_seconds_and_doubles_capped():
    tag = TAGParallelInterface()
    live = LiveFigure(tag, width=6, height=3)
    assert live.span == 5.0 and live.refresh() and not live.refresh()       # nothing to draw until time passes
    live.origin -= 120.0                                                    # the run is two minutes in
    assert live.refresh() and live.span == 140.0                            # 5, 10, 20, 40, 80, then +60 (the cap)
    assert not live.refresh()                                               # stable until outgrown again
    live.origin -= 30.0
    assert live.refresh() and live.span == 200.0
    logical = LiveFigure(tag, logical=True, width=6, height=3)
    logical.origin -= 120.0
    assert logical.refresh() and not logical.refresh() and logical.span == 5.0   # the paper's order: no axis to grow
    live.close()
    logical.close()
    print("  ok: the live axis starts at 5 s and doubles by at most a minute when outgrown")


def test_a_finished_run_fits_its_axis():
    tag = TAGParallelInterface()
    live = LiveFigure(tag, width=6, height=3)
    live.origin -= 12.0                                                     # the figure was made 12 s ago
    tag.open("planner", spec("job"), state=0)
    tag.update("w1", "q1", 1)
    assert live.refresh() and live.span == 20.0                             # grown past 12 s: 5, 10, 20
    live.finish()
    assert live.finished and 12.0 <= live.span < 12.5 and not live.refresh()   # fitted to the run, then still
    live.origin -= 100.0
    assert not live.refresh()                                               # a finished axis no longer grows
    tag.update("w2", "q1", 2)                                               # a late change draws on the fitted axis
    assert live.refresh() and 112.0 <= live.span < 112.5
    logical = LiveFigure(tag, logical=True, width=6, height=3)
    logical.watch(until=lambda: True)                                       # until: the run is over
    assert logical.finished and logical.span == 5.0                         # nothing to fit in the paper's order
    live.close()
    logical.close()
    print("  ok: when the run is over the axis is fitted to it and stops growing")


def test_an_inline_live_figure_follows_a_synchronous_run():
    """A run that is synchronous on the main thread has no loop to refresh
    from: inline, every change redraws on the spot."""
    from dig_tag.interface import DIGTAG

    dt = DIGTAG(agents=["alice", "bob"], environment=StubEnvironment())
    live = LiveFigure(dt, capture=True, inline=True, width=6, height=3)
    assert live.frames == []
    dt.inject(DIGEvent(payload={"text": "start"}), to=["alice"])           # one change, one frame
    with dt.open_activation("alice") as h:
        (feedback,) = dt.act(h, "echo", q="x")
        dt.send(h, feedback, to=["bob"])
    assert len(live.frames) == 5 and not live.dirty                        # inject, open, act, send, end
    live.close()
    print("  ok: inline, a synchronous run on the main thread draws itself at every change")


def test_a_live_figure_follows_an_asyncio_run():
    """The DIG's runtime and the figure share the main thread: the figure
    follows from inside the event loop, and a run in progress is drawn
    with its open activations reaching to now."""
    import asyncio

    from dig_tag.dig import AsyncAgent
    from dig_tag.interface import DIGTAG
    from dig_tag.tag import Version, observed

    dt = DIGTAG(environment=StubEnvironment())
    stop = asyncio.Event()

    async def planner(h, dt):                     # hands the job to the worker; closes it after two drafts
        (job,) = observed(dt.received(h)[-1].payload)
        await asyncio.sleep(0.03)
        if job.state == "drafted twice":
            dt.close(h, job.identity)
            stop.set()
            return
        dt.send(h, dt.received(h)[-1], to=["worker"])

    async def worker(h, dt):                      # acts on Z, drafts, and reports back
        (job,) = observed(dt.received(h)[-1].payload)
        await asyncio.sleep(0.03)
        dt.act(h, "echo", q="work")
        drafted = dt.update(h, job, state="drafted twice" if job.state == "drafted" else "drafted")
        dt.send(h, drafted, to=["planner"])

    agents = [AsyncAgent("planner", dt, planner), AsyncAgent("worker", dt, worker)]
    live = LiveFigure(dt, capture=True, width=8, height=4)
    seen_open = []

    async def main():
        dt.inject_task(Version(spec=spec("job"), state="todo"), to=["planner"])
        team = asyncio.ensure_future(asyncio.gather(*(agent.run(stop) for agent in agents)))
        watching = asyncio.create_task(live.watch_async(interval=0.005, until=team.done))
        while not team.done():
            seen_open.append(any(h.is_open for h in dt.dig.activations.values()))
            await asyncio.sleep(0.005)
        await team
        await watching

    asyncio.run(main())
    assert live.finished and any(seen_open) and len(live.frames) >= 3
    assert len(dt.dig.activations) == 5 and list(measure_record(dt).bands) == ["Z", "D", "T"]
    assert dt.tag.tasks["q3"].state == "drafted twice"
    live.close()
    print(f"  ok: the live figure follows an asyncio run from inside its loop ({len(live.frames)} frames)")


def test_a_live_figure_follows_a_threaded_tag(tmp_path):
    tag = TAGParallelInterface()
    live = LiveFigure(tag, capture=True, width=6, height=3)
    assert live.refresh() and not live.refresh()                            # drawn once, then nothing new
    tag.open("planner", spec("job"), state=0)
    assert live.dirty and live.refresh() and len(live.frames) == 2

    def work(name: str) -> None:
        for _ in range(3):
            tag.update(name, "q1", {"by": name})

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(1, 7)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):                    # the main thread draws while they act
        live.refresh()
    for thread in threads:
        thread.join()
    live.refresh()
    assert len(tag.tag.actions) == 19 and not live.dirty
    assert live.frames[-1].shape[2] == 4 and len(live.frames) >= 3

    def fits(frame, w_px, h_px):                                            # the record fills the window one way
        h, w = frame.shape[:2]
        return w <= w_px + 1 and h <= h_px + 1 and (abs(w - w_px) <= 1 or abs(h - h_px) <= 1)

    small = live.frames[-1]
    assert fits(small, 600, 300)
    live.fig.set_size_inches(12, 6)                                         # the window is resized: the image follows
    assert live.refresh(force=True) and fits(live.frames[-1], 1200, 600)
    assert live.frames[-1].shape[1] > small.shape[1]
    path = live.save(tmp_path / "live.gif")
    assert path.stat().st_size > 1000
    live.close()
    print(f"  ok: the live figure redraws on the main thread while six threads act ({len(live.frames)} frames)")


if __name__ == "__main__":
    run_golden(globals(), "live figure golden tests")
