"""Golden tests for the DIG interface on its own -- D and Z without T:
inject, activations, send, environment calls, and a record with no task
graph in it.
Run with:  python tests/test_dig_interface.py"""

import pytest

from dig_tag.base import ToolCall
from dig_tag.dig import DIGParallelInterface, ENVIRONMENT_ORIGIN, DIGEvent
from helpers import run_golden, StubEnvironment  # first: it puts the repo root on sys.path for direct runs


def test_the_dig_stands_alone():
    env = StubEnvironment()
    dt = DIGParallelInterface(agents=["alice", "bob"], environment=env)
    assert dt.reserved_tools == {"send"}
    root = dt.inject(DIGEvent(payload={"text": "start"}), to=["alice"])
    with dt.open_activation("alice") as h:
        assert h.inputs == [root]
        (feedback,) = dt.act(h, "echo", q="x")
        assert feedback.origin == ENVIRONMENT_ORIGIN and feedback.producer.activation_id == h.id
        dt.send(h, feedback, to=["bob"])
        with pytest.raises(ValueError, match="unknown tool"):
            dt.issue(h, ToolCall("open", {"spec": {}}))      # a task tool is not in the DIG's vocabulary
        assert len(h.calls) == 2
    assert dt.pending("bob") == [feedback] and env.calls[0].tool == "echo"

    data = dt.to_dict()
    assert data["schema"] == "dig.v2" and "tag" not in data
    back = DIGParallelInterface.from_dict(data)
    assert back.inbox("bob") == [back.dig.events[feedback.id]] and back.to_dict() == data
    print("  ok: D and Z work with no T: inject, send, environment calls, the record")


def test_threads_act_through_the_interface_at_once():
    """Parallel: agents on threads issue through one interface; every write
    is one critical section, so the record is whole and each activation's
    calls are its own."""
    import threading

    dt = DIGParallelInterface(agents=[f"a{i}" for i in range(6)], environment=StubEnvironment())
    root = dt.inject(DIGEvent(payload={"text": "start"}), to=dt.agents)

    def work(agent_id: str) -> None:
        for _ in range(5):
            with dt.open_activation(agent_id) as h:
                (feedback,) = dt.act(h, "echo", by=agent_id)
                dt.send(h, feedback, to=[a for a in dt.agents if a != agent_id])

    threads = [threading.Thread(target=work, args=(agent_id,)) for agent_id in dt.agents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    activations = list(dt.dig.activations.values())
    assert len(activations) == 30 and all(not h.is_open and h.error is None for h in activations)
    assert all([c.tool for c in h.calls] == ["echo", "send"] for h in activations)
    assert all(h.calls[0].returned[0].payload["by"] == h.agent_id for h in activations)   # no call crossed activations
    assert len(dt.dig.events) == 31 and root.id in dt.dig.events
    assert all(len(dt.inbox(agent_id)) == 1 + 25 for agent_id in dt.agents)             # the root, and everyone else's sends
    with dt.hold():
        assert len(dt.dig.activations) == 30
    print("  ok: six threads issued through one interface; the record is whole")


def test_a_slow_tool_holds_no_other_agent():
    """The world's step runs outside the lock: six agents each calling a
    tool that takes a fifth of a second finish together, not in turn."""
    import threading
    import time

    from dig_tag.dig import Environment

    class SlowWorld(Environment):
        tools = frozenset({"wait"})

        def step(self, activation, call):
            time.sleep(0.2)
            return [DIGEvent(payload={"waited": activation.agent_id})]

    dt = DIGParallelInterface(agents=[f"a{i}" for i in range(6)], environment=SlowWorld())

    def work(agent_id: str) -> None:
        with dt.open_activation(agent_id) as h:
            dt.act(h, "wait")

    threads = [threading.Thread(target=work, args=(agent_id,)) for agent_id in dt.agents]
    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.time() - start
    assert elapsed < 0.6, elapsed                                          # six in turn would take 1.2 s
    assert len(dt.dig.events) == 6 and all(len(h.calls) == 1 for h in dt.dig.activations.values())
    print(f"  ok: six agents waited on the world at once ({elapsed:.2f} s for six 0.2 s tools)")


if __name__ == "__main__":
    run_golden(globals(), "DIG interface golden tests")
