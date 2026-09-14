"""Golden tests for the TAG interface on its own: the TAG's tools as an
action space over one shared graph, what a step hands back (the
observation, or a policy's own return), steps atomic under threads, a
consumed version branching, and the record.
Run with:  python tests/test_tag_interface.py"""

import threading

import pytest

from helpers import run_golden, spec  # first: it puts the repo root on sys.path for direct runs

from dig_tag.tag import observation_with_history, TAG_TOOLS  # noqa: E402
from dig_tag.tag import TAGParallelInterface, current, observed, produced
from dig_tag.tag.views import build_observation


def frontier_ids(handed) -> list:
    return [q.id for q in observed(handed)]


def test_agents_act_on_one_graph_and_are_handed_its_frontier():
    tag = TAGParallelInterface()
    assert tag.tools == {"open", "edit", "update", "split", "join", "close", "attach", "observe"}
    changes = []
    tag.subscribe(lambda: changes.append(len(tag.tag.tasks)))

    seen = tag.open("planner", spec("survey"), state={"owner": "planner"})
    assert seen == {"open": ["k1"], "frontier": [tag.tag.tasks["q1"].to_dict()]}        # what every tool hands back
    (q1,) = observed(seen)
    seen = tag.split("planner", q1, [(q1.identity, spec("intro"), {"owner": "w1"}), (None, spec("body"), {"owner": "w2"})])
    assert frontier_ids(seen) == ["q2", "q3"] and seen["open"] == ["k1", "k2"]
    seen = tag.update("w1", current(seen, "k1"), {"owner": "w1", "status": "drafted"})
    assert frontier_ids(seen) == ["q3", "q4"]
    seen = tag.update("w2", "q3", {"owner": "w2", "status": "drafted"})
    assert frontier_ids(seen) == ["q4", "q5"]
    seen = tag.attach("w1", "q4", {"hits": 3})
    phi = tag.tag.evidence["phi1"]
    assert phi.target == "q4" and phi.source == "w1" and frontier_ids(seen) == ["q4", "q5"]   # attach hands the same back
    seen = tag.join("planner", ["q4", "q5"], spec("survey"), state={"status": "joined"}, identity="k1")
    assert frontier_ids(seen) == ["q6"]                                                 # k2 is open but has no current version
    seen = tag.close("planner", "k1")
    assert seen == {"open": [], "frontier": []} and tag.observe() == seen           # closing the join closed k2 too
    assert build_observation(tag.tag).closed_identities == ["k1", "k2"]
    assert [a.issuer for a in tag.tag.actions.values()] == ["planner", "planner", "w1", "w2", "planner", "planner"]
    assert [a.call.index for a in tag.tag.actions.values()] == [0, 1, 2, 3, 5, 6]    # one sequence; the attach was step 4
    assert tag.step("w2", "observe") == seen and len(tag.tag.actions) == 6             # observe: a step, no action
    assert changes and changes[-1] == 6                                               # the hook saw every change
    print("  ok: the TAG's tools act on one shared graph; every step hands back the frontier of the open tasks")


def test_returns_are_a_policy():
    tag = TAGParallelInterface(returns={"open": produced, "attach": produced, "close": produced})
    assert tag.open("a", spec("job"), state=0) == {"versions": [tag.tag.tasks["q1"].to_dict()]}
    assert frontier_ids(tag.update("a", "q1", 1)) == ["q2"]                            # the others as by default
    assert tag.attach("a", "q2", "x") == {"evidence": tag.tag.evidence["phi1"].to_dict()}
    assert tag.close("a", "k1") == {}
    with pytest.raises(ValueError, match="no such TAG tool"):
        TAGParallelInterface(returns={"teleport": produced})
    print("  ok: a tool hands back the observation unless its return policy says otherwise")


def test_a_rejected_step_changes_nothing():
    tag = TAGParallelInterface()
    tag.open("a", spec("job"), state=0)
    with pytest.raises(ValueError, match="unknown tool"):
        tag.step("a", "send", {"event": "e1", "to": ["b"]})
    with pytest.raises(ValueError, match="not in the TAG"):
        tag.step("a", "update", {"task": "q9", "state": 1})
    with pytest.raises(ValueError, match="requires `target`"):
        tag.step("a", "attach", {"payload": 1})
    assert list(tag.tag.tasks) == ["q1"] and frontier_ids(tag.observe()) == ["q1"]
    assert len(tag.tag.actions) == 1 and tag.tag.evidence == {}
    seen = tag.step("a", "update", {"task": "q1", "state": 1})                        # a raw call steps like the tool methods
    assert frontier_ids(seen) == ["q2"] and tag.tag.tasks["q2"].identity == "k1"
    assert tag.step("a", "attach", {"target": "q2", "payload": 1}) == seen
    print("  ok: a rejected step leaves the graph as it was; a raw call steps like the tool methods")


def test_concurrent_updates_of_one_version_branch():
    tag = TAGParallelInterface()
    tag.open("planner", spec("job"), state=0)
    results = {}

    def work(name: str) -> None:
        results[name] = tag.update(name, "q1", {"by": name})

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(1, 9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    versions = [q for q in tag.tag.tasks.values() if q.id != "q1"]
    assert len(versions) == 8 and len({q.id for q in versions}) == 8                    # ids unique under the lock
    assert sum(q.identity == "k1" for q in versions) == 1                               # one continues the identity
    assert len({q.identity for q in versions}) == 8                                     # the others branched
    assert len(frontier_ids(tag.observe())) == 8 and not tag.tag.on_frontier("q1")
    assert len(tag.tag.actions) == 9 and all(len(observed(seen)) >= 1 for seen in results.values())
    print("  ok: eight agents updating one version leave eight tasks, never two versions of one")


def test_the_record_round_trips(tmp_path):
    tag = TAGParallelInterface()
    tag.open("planner", spec("job"), state=0)
    tag.attach("w1", "q1", {"note": "x"})
    tag.update("w1", "q1", state=1)
    path = tag.save_json(tmp_path / "tag.json")
    back = TAGParallelInterface.load_json(path)
    assert back.to_dict() == tag.to_dict() and frontier_ids(back.observe()) == ["q2"]
    assert back.tag.evidence["phi1"].source == "w1" and back.tag.actions["a2"].output_ids == ["q2"]
    seen = back.step("w2", "update", {"task": "q2", "state": 2})                       # the loaded TAG continues
    assert frontier_ids(seen) == ["q3"] and back.tag.tasks["q3"].identity == "k1"
    assert back.tag.actions["a3"].call.index == 3
    with pytest.raises(ValueError, match="unsupported schema"):
        TAGParallelInterface.from_dict({"schema": "other"})
    print("  ok: the TAG's record round-trips through JSON and continues")


if __name__ == "__main__":
    run_golden(globals(), "TAG interface golden tests")


def test_a_return_policy_can_hand_back_histories_relations_and_evidence():
    tag = TAGParallelInterface(returns={tool: observation_with_history for tool in TAG_TOOLS})
    (q1,) = observed(tag.open("planner", spec("survey"), state="todo"))
    handed = tag.split("planner", q1, [(None, spec("intro"), "todo"), (None, spec("body"), "todo")])
    tag.attach("w1", "q2", {"hits": 3})
    handed = tag.update("w1", "q2", "drafted")
    assert handed["open"] == ["k1", "k2", "k3"] and [q["id"] for q in handed["frontier"]] == ["q3", "q4"]   # k1: open, no current version
    assert handed["histories"]["k2"] == [
        {"version": "q2", "tool": "split", "issuer": "planner", "index": 1},
        {"version": "q4", "tool": "update", "issuer": "w1", "index": 3},
    ]
    assert handed["relations"]["k2"]["origin"]["from"] == [{"version": "q1", "identity": "k1"}]
    assert handed["evidence"] == {"q3": [], "q4": []}                     # phi1 is on q2, not its successor
    assert tag.observe() == handed                                          # observe follows the policy too
    print("  ok: a return policy hands back histories, relations, and evidence; observe follows it")
