"""Golden tests for the asynchronously activated agent -- wake on send,
one activation per batch, overlap across agents, arrivals during an open
activation, stop, and the error path.
Run with:  python tests/test_agent.py"""

import asyncio

import pytest

from dig_tag.dig import AsyncAgent, DIGEvent
from dig_tag.tag import observed
from dig_tag.interface import DIGTAG
from helpers import spec


def _run(coro):
    return asyncio.run(coro)


def test_agent_wakes_on_send_and_fires_once_per_batch():
    dt = DIGTAG()
    seen = []

    def planner(h, dt):
        seen.append([e.id for e in h.inputs])
        task = dt.open_task(h, spec("work"))
        dt.send(h, task, to=["worker"])

    def worker(h, dt):
        seen.append([e.id for e in h.inputs])
        (task,) = observed(h.inputs[-1].payload)
        done = dt.update(h, task, state="done")
        dt.send(h, done, to=["planner"])

    async def main():
        stop = asyncio.Event()
        agents = [AsyncAgent("planner", dt, planner), AsyncAgent("worker", dt, worker)]
        assert dt.agents == ["planner", "worker"]

        def finish():
            if len(dt.dig.activations) == 3 and not any(h.is_open for h in dt.dig.activations.values()):
                stop.set()
        dt.subscribe(finish)
        dt.inject(DIGEvent(payload={"go": True}), to=["planner"])
        await asyncio.wait_for(asyncio.gather(*(a.run(stop) for a in agents)), timeout=5)

    _run(main())
    assert seen == [["e1"], ["e2"], ["e1", "e2", "e3"]]   # planner, worker, planner again: context accumulates
    assert [h.agent_id for h in dt.dig.activations.values()] == ["planner", "worker", "planner"]
    print("  ok: an agent wakes when something is pending and fires once per batch")


def test_activations_overlap_across_agents():
    dt = DIGTAG()

    async def slow(h, dt):
        await asyncio.sleep(0.05)
        dt.open_task(h, spec(h.agent_id))

    async def main():
        stop = asyncio.Event()
        agents = [AsyncAgent(name, dt, slow) for name in ("a", "b")]
        dt.subscribe(lambda: stop.set() if len(dt.dig.activations) == 2 and not any(
            h.is_open for h in dt.dig.activations.values()) else None)
        dt.inject(DIGEvent(), to=["a", "b"])
        await asyncio.wait_for(asyncio.gather(*(x.run(stop) for x in agents)), timeout=5)

    _run(main())
    ha, hb = dt.dig.activations.values()
    assert ha.started_at < hb.ended_at and hb.started_at < ha.ended_at   # the spans overlap
    assert ha.agent_id == "a" and hb.agent_id == "b"
    print("  ok: two agents' activations overlap in time")


def test_arrival_during_an_open_activation_waits_for_the_next_one():
    dt = DIGTAG()
    batches = []

    async def listener(h, dt):
        batches.append([e.id for e in h.inputs])
        await asyncio.sleep(0.05)

    async def main():
        stop = asyncio.Event()
        agent = AsyncAgent("listener", dt, listener)
        dt.inject(DIGEvent(), to=["listener"])
        task = asyncio.ensure_future(agent.run(stop))
        await asyncio.sleep(0.01)                  # h1 is open now
        assert agent.active
        dt.inject(DIGEvent(), to=["listener"])     # arrives mid-activation
        assert dt.pending("listener") and dt.dig.activations["h1"].inputs[-1].id == "e1"
        await asyncio.sleep(0.12)                  # h1 ends; h2 fires with the whole inbox
        stop.set()
        agent.wake()
        await asyncio.wait_for(task, timeout=5)

    _run(main())
    assert batches == [["e1"], ["e1", "e2"]]
    print("  ok: an event sent during an open activation enters the next one")


def test_pending_before_run_fires_without_a_new_update():
    dt = DIGTAG(agents=["a"])
    dt.inject(DIGEvent(), to=["a"])          # pending before the agent exists
    fired = []

    async def main():
        stop = asyncio.Event()
        agent = AsyncAgent("a", dt, lambda h, dt: fired.append(h.id))
        dt.subscribe(lambda: stop.set() if fired else None)
        await asyncio.wait_for(agent.run(stop), timeout=5)

    _run(main())
    assert fired == ["h1"]
    print("  ok: run() fires on what is already pending")


def test_behavior_error_closes_the_activation_and_propagates():
    dt = DIGTAG()

    def broken(h, dt):
        dt.open_task(h, spec("half done"))
        raise RuntimeError("model call failed")

    async def main():
        agent = AsyncAgent("a", dt, broken)
        dt.inject(DIGEvent(), to=["a"])
        with pytest.raises(RuntimeError, match="model call failed"):
            await asyncio.wait_for(agent.run(asyncio.Event()), timeout=5)

    _run(main())
    (h,) = dt.dig.activations.values()
    assert not h.is_open and h.error["type"] == "RuntimeError"
    assert len(h.calls) == 1 and "q1" in dt.tag.tasks    # prefix-realized
    with pytest.raises(TypeError, match="behavior must be callable"):
        AsyncAgent("b", dt, None)
    print("  ok: a failing behavior closes its activation with the error noted and re-raises")


def test_stop_ends_an_idle_loop():
    dt = DIGTAG()

    async def main():
        stop = asyncio.Event()
        agent = AsyncAgent("a", dt, lambda h, dt: None)
        task = asyncio.ensure_future(agent.run(stop))
        await asyncio.sleep(0.01)
        assert not task.done()
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    _run(main())
    assert dt.dig.activations == {}
    print("  ok: stop ends an idle loop")


if __name__ == "__main__":
    from helpers import run_golden

    run_golden(globals(), "agent golden tests")
