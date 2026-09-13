# Route supervision: ordered sub-goals for COOHAVIOR tasks

A COOHAVIOR task is not "put the box on the bed". It is "carry the box through
these nine supports, in this order, then onto the bed", and a robot that puts
the box straight on the bed has failed it. Our port so far carries only the
final state, because that is all BDDL can say -- which is why `v4_s1_v4_hl`
ends at env_step 0 (PORTING_COOHAVIOR.md, "the goal holds at t=0"). This
document is the plan for the two capabilities that close that gap:

1. a **route file** next to each `problem0.bddl` that states the ordered
   sub-goals, roles and load rules BDDL cannot;
2. a **supervision layer** that checks the sub-goals are met correctly and in
   order, decides `terminated`, tells the agent where it is on the route, and
   scores it.

Written 2026-09-12 from reading COOHAVIOR's own code, not its README. File
references into COOHAVIOR are under `/home/zixuanwe/Desktop/COOHAVIOR/`.
Read `coop2/PORTING_COOHAVIOR.md` first for how a task's BDDL, scene and robot
layout are produced; this document starts where that one stops.

## 1. What a COOHAVIOR task actually is

**The authority is `behavior_style_task/tasks.modified.json`, not the BDDL.**
COOHAVIOR says so itself: its `bddl_scope.external_execution_constraints`
lists what the BDDL cannot carry -- "four checkpoint order", "payload capacity
by robot role", "box mass", "movable-arm base lock while gripping",
"push/pull direction", "exact spawn pose". Its BDDL goal is the final `ontop`
and nothing else.

Twelve tasks = 3 scenes x {LL, LH, HL, HH}. Two independent axes:

| letter | L | H |
|---|---|---|
| first: parallelism | 1 route, 1 box | 5 routes, 5 boxes |
| second: payload | 8 g; the drone may lift it | 20 g; only Jackal carrier + Ridgeback arm are load-bearing |

**A route is the ordered sub-task list.** Its nodes are
spawn -> checkpoint C1..CN -> destination, each checkpoint carrying
`sequence`, `relation: ONTOP`, `bddl_support` (a BDDL instance such as
`shelf.n.01_3`), `room_instance`, and the runtime object it binds to. Every
node means one BDDL predicate: `ontop(<cargo>, <support>)`.

| | routes | nodes per route | nodes total | S1 in words |
|---|---|---|---|---|
| LL, LH | 1 | 9 checkpoints + destination | 10 | one box visits M5 -> M3 -> M2 -> M1 -> M4 -> back to M5's bed |
| HL, HH | 5 | 1 checkpoint + destination | 10 | each station's box goes one station on |

All three scenes have exactly this shape (S2 stations are G1..G5, S3 are
H1..H5); only the supports differ. Measured over all twelve entries, not
assumed.

**`packages[].depends_on` is documentation, not a mechanism.** A grep of every
Python file in `behavior_style_task/` and `custom_task_suite/` finds it read in
two places, both `validate_*.py` scripts asserting its shape
(`validate_ground_elevated_v4.py:25`). No runtime or evaluation code consumes
it. In L\* the five packages P1 -> P2 -> ... -> P5 all sit on the one route
and are the same box's five legs (`leg_count: 5`); in H\* every `depends_on`
is empty. **The only order that is enforced is checkpoint `sequence`.** Our
route file must not encode order twice.

`required_robot_roles` is likewise only asserted, but its meaning matters:
LL/HL need `push_car` alone; LH/HH need `pull_car + push_car + robot_arm`
together, which is what makes those tasks impossible for one robot.

**COOHAVIOR's BDDL and its route file disagree on where LL ends.**
`custom_task_suite/push_pull_arm_v4/bddl/S1-V4-LL.bddl` says
`ontop(box, floor.n.01_target_1)` with the target floor `inroom bedroom`. The
route entry says the destination is station M5, `childs_room_0`, fixture
`bed.n.01_1`, and the prompt handed to the model in a real run
(`runs/s1_ll_run_001/task_definition_goal_prompt.md`) says the same. The
route file is what they run; the BDDL is stale. Our port of LL/LH ends on a
bedroom floor/cabinet for the same reason -- it was read off their BDDL. The
route file fixes the destination; see section 5.1.

## 2. How COOHAVIOR checks the sub-goals

Three parts, all under `behavior_style_task/`.

**(a) `evaluation/benchmark_recorder.py` is the ledger.** Per route an ordered
node list `[C1, ..., C9, <route>-D]`; per box a `next_index`.
`record_box_at_node(box, node)` classifies the visit:

| visit is | recorded as | advances progress |
|---|---|---|
| the node at `next_index` | `completed` | yes, `next_index += 1` |
| a node already visited | `duplicate` | no |
| anything else | `out_of_order` | no |

A route is complete when one of its boxes reaches `-D`; `task_complete` is
every route complete. `summary()` exposes `correct_checkpoint_count`,
`correct_destination_count`, `duplicate_visit_count`,
`out_of_order_visit_count` and `task_complete`.

**(b) The trigger is an action, not a state.** `PLACE_AT_CHECKPOINT` runs
`robot_tools/task_actions.py:1994 place_at()`: Lula IK reaches the support ->
the marker's AABB holds no other payload (`task_point_occupancy`, line 483)
-> suction releases -> **the box root is teleported to the marker** ->
`_record_marker_completion` reads the marker prim's `checkpoint_id` attribute
and calls the ledger. Nothing re-reads `ontop` afterwards. Placing on the
wrong support is recorded `out_of_order`; being near a marker never counts.

**(c) `robot_tools/isaac_stage_tools.py:344 next_task_targets()` is the
"what is next" query** behind the model's `GET_NEXT_TARGET`: for each route,
the first node not in the completed set. L\* returns one target, H\* up to
five. `coop/runner.py:244` puts the ledger's summary into every observation
as `runtime_evaluation`, and `runner.py:526` ends the run on `task_complete`.

Dependency constraints beyond order are refused at the action level
(`coop/executor.py:653 precheck` and the api): base locked while holding;
Jackal may not GRASP or PLACE; Jackal may not `NAVIGATE_TO` a checkpoint
unless `carrying_box`; Jackal may not park at the box; **a held box within
0.55 m of the current unfinished node makes `PLACE_ON_JACKAL` refuse**
(`executor.py:559` -- it stops the box being shuttled between arm and
carrier without ever being placed, which is exactly how both real runs
ended); the drone may `RELEASE` only within 0.12 m horizontally of the named
marker; the drone's payload gate is 10 g.

Scoring: `Y_task = completed_nodes / required_nodes` where required =
checkpoints + destinations (`coop/compute_metrics.py:343`); `S_team` is one
point per in-order node, credited to the placing robot's team.

**This layer has never got past node 0 in a real run.** `runs/s1_ll_run_001`
(1000 s) and `runs/s1_lh_run_001` (300 s) both score 0 of 10 nodes, timed
out. The ledger is exercised only by `evaluation/test_benchmark_recorder.py`.

## 3. What we have to build on

| piece | where | what it gives us |
|---|---|---|
| macro-step boundary | `coop2/behavior_env/coop_env.py:894` -- `task_tracker.step()` is called when any primitive terminates | the moment to check sub-goals; the world model is rebuilt right after |
| predicate evaluation | `world_state.py:682 relation_holds(token, a, b)` | one `ontop` question, answered against the live scene |
| task container | `cooperative_tasks.py:408 BehaviorTaskState`, `:466 CoopTaskTracker` | goals are already BDDL triples; the tracker already snapshots per macro-step |
| task injection | `coop_env.py:70,104,341 task_specs` | a constructor parameter nobody populates |
| goal in the prompt | `coop_env.py:743 _goal_terms` -> `symbolic_view.render_goal_terms` -> the `YOUR TASK` block at `symbolic_view.py:603` | the place the route state goes |
| observation plumbing | `world_state.py:732 observation_for(goal_status=, goal_terms=)` | two fields already carried per agent |
| termination | `coop_env.py:917 terminated_all = self._goal_reached()`, `:971` | `check_goal` is currently the sole authority |
| post-episode | `plan_log_saver.py:44-47, 98-99` read `task_tracker.get_history()` / `capability_history`; `run_individual.py:340 compute_all_metrics` | the output path that must not break (see "recurring defect class" in CLAUDE.md) |
| role constraints | `base_locked_while_holding`, `carrier`, `NO_ARM`, `BASE_LOCKED` in `symbolic_contention.py` and `target_hints` | most of the action gates already exist; the mass gate does not |

The two-side rule from the carrying work holds here too: **the engine refuses,
and the listing does not offer what the engine would refuse.** Anything this
layer forbids has to be withheld from `target_hints` as well, or an agent
burns a plan and an LLM round trip learning it.

## 4. Design decisions

**Detect by state, not by action.** COOHAVIOR records a node when its
placement tool succeeds on the right marker. We evaluate the node's predicate
after every primitive terminates. Three reasons:

* our symbolic `place_on_top` teleports and welds, and `OnTop` is reliable
  once the object is kept awake (CLAUDE.md, "the bug that made the activity
  look unsolvable") -- the predicate *is* the ground truth here;
* it covers every path that puts cargo on a support -- `place_on_top`,
  a drone's `release`, `unload_from` onto a support if that ever exists --
  without instrumenting each verb;
* it is the same instrument `check_goal` uses, so the two cannot disagree
  about whether a box is on a shelf.

The cost is that a state check can only see the box where it is *now*: a box
that was placed on C2 and picked up again before the next macro-step is not
seen. Under our engine that cannot happen -- a placement terminates a
primitive, and the check runs at that boundary before anything else is
issued -- but it is the assumption to re-verify if the engine ever aligns
differently.

**Out-of-order placements are recorded, not punished.** COOHAVIOR's choice,
kept: a box put on C4 while C2 is next is an `out_of_order` event, and
progress resumes the moment C2 is satisfied. Punishing it would conflate an
exploratory mistake with an inability, and what we measure is the difference
between cooperation topologies, not that. Open for the user to overrule
(section 8).

**The route file is a sidecar, and the BDDL is unchanged.** BDDL keeps the
final state and the object scope, so the sampler, the cached instance,
`check_goal` and `render_goal_terms` all keep working. A BDDL-flavoured
s-expression was considered and rejected: `bddl3`'s parser will not accept a
new section, so we would be writing a parser for a file only we read, and
JSON gets a CPU test for free.

**`check_goal` stops deciding `terminated` for a routed task.** With a route
file present, `terminated` = every route complete. `check_goal` is kept as a
consistency check and printed when it disagrees with the tracker at the end
-- the BDDL final state should be true exactly when the last destination is
reached, and if it is not, one of the two files is wrong. HL stops ending at
env_step 0 because the trivial BDDL goal no longer decides anything.

## 5. The pieces

### 5.1 The route file: `bddl3/bddl/activity_definitions/<activity>/route.json`

Discovered by the activity name, next to `problem0.bddl`; no new flag. An
activity without one behaves exactly as today.

```json
{
  "activity": "v4_s1_v4_ll",
  "source": "COOHAVIOR tasks.modified.json S1-V4-LL, route S1-M5",
  "policy": "serial",
  "routes": [
    {
      "id": "S1-M5",
      "cargo": "die.n.01_1",
      "nodes": [
        {"id": "C1", "goal": ["ontop", "die.n.01_1", "cabinet.n.01_1"]},
        {"id": "C2", "goal": ["ontop", "die.n.01_1", "bookcase.n.01_1"]},
        {"id": "C3", "goal": ["ontop", "die.n.01_1", "electric_refrigerator.n.01_1"]},
        {"id": "C4", "goal": ["ontop", "die.n.01_1", "armchair.n.01_1"]},
        {"id": "C5", "goal": ["ontop", "die.n.01_1", "breakfast_table.n.01_1"]},
        {"id": "C6", "goal": ["ontop", "die.n.01_1", "coffee_table.n.01_1"]},
        {"id": "C7", "goal": ["ontop", "die.n.01_1", "cabinet.n.01_2"]},
        {"id": "C8", "goal": ["ontop", "die.n.01_1", "cabinet.n.01_3"]},
        {"id": "C9", "goal": ["ontop", "die.n.01_1", "bed.n.01_2"]},
        {"id": "D",  "goal": ["ontop", "die.n.01_1", "bed.n.01_1"]}
      ]
    }
  ],
  "lift": {
    "die.n.01": ["arm", "drone"],
    "notebook.n.01": ["arm"]
  }
}
```

Rules, each of which is a CPU test:

* **A node's `goal` is a BDDL predicate triple** -- the same shape as
  `BehaviorTaskState.goal` and as a line of an activity's `:goal`. Nothing
  else (no marker ids, no coordinates, no runtime names). `relation_holds`
  evaluates it as it is.
* **Every id in a node must be in the BDDL's `:objects`**, and the loader
  refuses otherwise. COOHAVIOR's own checkpoints carry
  `"bddl_mapping_status": "add this object to BDDL :objects / :init"` for the
  same reason: a support the sampler did not bind is a support the scene may
  not have. This means the LL/LH BDDL grows from 3 objects to ~13. Their
  `inroom` lines go in `:init` -- that is where the room of a node comes from
  for the prompt; the route file does not repeat it.
* **The last node is the destination**, and its goal must equal the BDDL
  `:goal` clause for that cargo. Checked at load; a mismatch is the LL
  BDDL-vs-route disagreement from section 1 and is refused, not warned.
* **No `depends_on`, no `sequence` field**: order is list order, and the
  five "packages" of L\* are the nodes.
* `policy` is `serial` (L\*: one route) or `parallel` (H\*: routes are
  independent, each has its own cargo). It is descriptive -- the tracker
  treats every route independently either way -- and is there so the prompt
  can say which it is.
* `lift` maps a cargo synset to the roles allowed to grasp it (section 5.4).
  Roles are ours: `arm` is a robot with a hand, `drone` is a robot whose
  layout says so, `carrier` is a robot with no arm. Left out means anyone
  with a hand.
* Support instance numbers are ours, not COOHAVIOR's. Renumber to
  `<synset>_<int>` per PORTING_COOHAVIOR.md step 1, and pin each to its
  runtime object through the sampler whitelist as today.

Synset names must be ours too, and the check has to be against the
taxonomy *and* the scene, because COOHAVIOR's names were wrong in three
different ways on S1 alone (found writing the LL file, 2026-09-12):

* `fridge.n.01` is not a synset here; ours is `electric_refrigerator.n.01`.
* `bottom_cabinet.n.01` (their C7, C8) is not a synset either -- the category
  `bottom_cabinet` belongs to `cabinet.n.01`, so C1/C7/C8 are three
  `cabinet.n.01` instances separated by `inroom`.
* Their `shelf_owvfik_0` (C2, and a node of every other S1 task) does not exist
  in our Merom_1_int under that name or category. The **same model**, `owvfik`,
  is there at the **same pose** (-0.17, 4.90) as `bookcase_owvfik_0` -- an older
  asset naming -- so the node is `bookcase.n.01_1`. A route file that said
  `shelf.n.01_1` would have passed the loader (the BDDL declared it) and failed
  at the sampler, or worse, sampled a shelf that is not the object COOHAVIOR
  meant.
* `ottoman.n.01` -- the destination of S2's G1 -> G2 route -- does not exist
  here at all and needs a substitute support when S2 is ported.

Every support must also be something `place_on_top` is offered on: all nine of
LL's are receptacles (`is_receptacle`), the fridge by `fillable`/`openable`
rather than as a surface. Check each new one before writing the file, not
after the sampler fails.

### 5.2 `RouteTracker` -- L1d, `coop2/behavior_env/route_tracker.py`

Pure Python over the world model; no OmniGibson import, so
`feasibility_verify/test_route_tracker_stubbed.py` can drive it with a stub
`relation_holds`.

```python
class RouteTracker:
    def __init__(self, world, spec: RouteSpec): ...
    def step(self, env_step: int) -> list[RouteEvent]:
        """Called at every macro-step boundary, after world.step()."""
    def next_node(self, route_id) -> RouteNode | None
    def progress(self) -> dict          # for the observation and for metrics
    def complete(self) -> bool          # every route's last node completed
    def history(self) -> list[RouteEvent]
```

`step()` per route: evaluate the goal of the node at `next_index`. True ->
emit `completed`, advance, then re-evaluate (a single placement may satisfy
nothing further, but the loop costs nothing and handles a node whose support
is also the next node's). Also evaluate every *later* node once; any that
holds while a nearer one does not -> `out_of_order`, recorded with which node
was expected. Do not evaluate earlier nodes -- a box sitting where it already
scored is not a duplicate visit under state detection; it is just still there.

Each `RouteEvent` carries `env_step`, `route_id`, `node_id`, `status`, the
predicate, and `by` -- the agent whose primitive terminated on this
macro-step and whose held-object set lost the cargo, when there is exactly
one such agent; otherwise `None`. That is the credit `S_team` needs, and
"unknown" is an honest value for it.

Evaluate at most `routes x remaining nodes` predicates per macro-step -- ten
for any of the twelve tasks. `relation_holds` was measured at one question
per call precisely so this is cheap; do not build the fact set.

### 5.3 Wiring into `coop_env`

* `_build` (`coop_env.py:341`): after the tracker, `self.route_tracker =
  RouteTracker(self.world, spec)` when `route.json` exists for
  `bddl_activity`, else `None`. Load through the same helper the BDDL uses to
  find its directory.
* `step` (`coop_env.py:894`): right after `self.world.step()` in the refresh
  branch, `events = self.route_tracker.step(env_step)`; print each event on
  one line (`[route] S1-M5 C2 completed at env_step 1830 by agent_0`), the
  same style as `[goal]` and `[nav]`.
* `_goal_reached` (`coop_env.py:971`): if `route_tracker` is set, return
  `route_tracker.complete()`; on completion also call `_bddl_goal_reached()`
  and print a `[route] BDDL check_goal disagrees` line if it is False.
  `goal_reached_at` is set either way so the rest of the code is unchanged.
* `_build_info`: `goal_terms` becomes the route rendering (5.5) when a
  tracker exists; `goal_status` carries `progress()` so the follower's report
  to a leader can say how far its route is.
* Post-episode: write `route_progress.json` (events + final `progress()`)
  beside `plan_logs.json`. Extend `test_cooperative_tasks_stubbed.py`'s
  post-episode test to run this saver, because that is the code path that
  has cost a GPU run per bug three times.

### 5.4 The lift gate -- `symbolic_contention.py` + `target_hints`

The one COOHAVIOR constraint we have no word for yet, already flagged in
PORTING_COOHAVIOR.md ("The weight is a capability constraint, and it is not
enforced yet"). With `lift` in the route file:

* `_require_may_lift(obj)` next to `_require_arm`: raise `PRE_CONDITION` with
  `reason_code: CANNOT_LIFT` when the cargo's synset is in `lift` and the
  robot's role is not listed. Role comes from the layout: `drone` if the
  layout says so, `carrier` if no arm, else `arm`.
* `target_hints`: do not offer `grasp` on such an object to such a robot, and
  say why on the object's line (`-> navigate_to  [too heavy for you]`), the
  way the base-lock absence is explained today.
* CPU test in `test_carrier_view_stubbed.py`'s style: a drone is offered
  `grasp(die)` and not `grasp(notebook)`; an arm is offered both.

### 5.5 What the agent is shown

Replace the `YOUR TASK` block, for a routed task, with the route in order,
marked. The list of supports is the *task*, not the world -- COOHAVIOR gives
the model all ten nodes in its task prompt -- and each support's room is in
the BDDL `:init` already, so stating it discloses nothing the activity did
not (the same argument that let `render_goal_terms` say "floor.n.01_2 is in
the bedroom"). The room listing stays strictly local.

```
YOUR TASK -- route S1-M5, carry die.n.01_1 through these in order:
  done   C1  ontop(die.n.01_1, cabinet.n.01_5)         [childs_room]
  NEXT   C2  ontop(die.n.01_1, shelf.n.01_3)           [kitchen]
         C3  ontop(die.n.01_1, electric_refrigerator.n.01_1)   [kitchen]
         ...
         D   ontop(die.n.01_1, bed.n.01_1)             [childs_room]
  A support skipped does not count; C2 must hold before C3 is credited.
```

Five routes in H\* print five blocks; a completed route prints one line. This
is what makes COOHAVIOR's `GET_NEXT_TARGET` unnecessary: the model is told
where it is instead of having to ask.

`ENV_DESCRIPTION` and `TEAM_ENV_DESCRIPTION` each get one sentence saying
what a route is and that progress is judged on the support the cargo rests
on; `test_team_prompt_stubbed.py` already fails if a shared rule leaves one
of the two.

### 5.6 Metrics

`compute_task_success_metrics` (`coop2/cognitive/compute_metrics.py:479`)
reads `route_progress.json` when present and adds COOHAVIOR's two numbers
under the same names, so a table can hold both projects' runs:

* `Y_task` = completed nodes / required nodes (all checkpoints plus all
  destinations; ten for every V4 task);
* `S_team` = one point per `completed` event, credited to `by`'s team,
  `unattributed` otherwise; plus `out_of_order` count.

`Y_plan`, failure attribution and tokens are unchanged and still the first
things to read (see the memory note on ignoring constraint metrics).

## 6. Order of work, each step verified before the next

1. **Done 2026-09-12.** `coop2/behavior_env/route_spec.py` (`load_route_spec`,
   `parse_route_spec`, 13 rules in `test_route_spec_stubbed.py`), `route.json`
   for `v4_s1_v4_ll`, its BDDL grown to 13 objects with the goal corrected to
   `bed.n.01_1`, `verify_definition` and the parser both pass, the sampler
   whitelist pins the nine supports by model. **A BDDL grown past its cached
   instance is silently unwinnable until re-sampled**: `behavior_task.py`
   skips instances the template never bound, the scope entry stays None, and
   `evaluate_bddl_predicate` returns False for None -- so `check_goal` can
   never fire and the run reads as agent failure. Re-sample is part of this
   step, not optional.
2. **Done 2026-09-12.** `coop2/behavior_env/route_tracker.py`, driven through
   the real loader by `test_route_tracker_stubbed.py` (11 checks: in-order
   completion, out-of-order recorded and uncredited then recovery, a scored
   node never re-evaluated, two parallel routes, credit only when exactly one
   terminated agent let go, a bare world with no inventories, the evaluation
   budget, serialisation). Decisions taken by the user the same day: out of
   order does not count and is not punished; `Y_task` counts nodes; the prompt
   shows the whole route; when route and BDDL disagree the route is the task
   and the BDDL is corrected (the loader's messages say so).
3. **Done 2026-09-12.** `coop_env` builds a `RouteTracker` when
   `load_route_spec` finds a file; `step()` judges the routes after
   `world.step()` and before `_build_info()`, so the prompt shows the node just
   credited; `_goal_reached()` is `route_tracker.complete()` when there is one,
   with `check_goal` run as a cross-check and a `[route] BDDL check_goal
   disagrees` line if the two part. `render_route_block` (L1b) puts the whole
   route under YOUR TASK, marked done/NEXT with each support's room;
   `save_route_progress` writes `route_progress.json`; `compute_route_metrics`
   adds `Y_task`, node counts, `out_of_order_visits`, `route_complete` and
   `S_team` (credited by team via `team_timeline.json`, `unattributed` kept
   apart) -- all CPU-tested on a fabricated run dir before any GPU run.
   `inspect_scene --view agent_0` on LL renders the ten-node block from the
   real instance. One thing the suite caught: a CPU test builds the env with
   `object.__new__`, so a new attribute set in `__init__` does not exist there
   -- every read of `route_tracker` goes through `getattr(..., None)`.
4. One GPU episode of LL, individual topology, the V4 layout. Acceptance: a
   `[route] ... completed` line per node in order, `terminated` on `D`,
   `check_goal` agrees, `route_progress.json` written, `Y_task` in
   `coop2_metrics.json`. Budget it: ten nodes at roughly navigate + grasp +
   navigate + place each is ~5000 ticks before any handoff, so `--steps 8000`
   and `--time-limit-seconds 0`.
5. **Done 2026-09-12.** The lift gate (5.4): `RobotSpec.drone` (a layout
   fact, `"drone": true` on every Crazyflie entry; never inferred from the
   model name), `carrier.lift_role` computing carrier / drone / arm in one
   place, `coop_env._build` handing `route_spec.lift` to the world and every
   controller as `lift_rules`, `_require_may_lift` in `_grasp` and
   `unload_from` (unloading puts the cargo in the hand, so it is a lift)
   raising `CANNOT_LIFT`, and `target_hints` withholding the verb with
   `blocked [too heavy for you]`. `ReasonCode.CANNOT_LIFT` terminates the
   plan like `TOO_FAR`. Both system prompts say what the mark means. CPU:
   `test_carrier_view_stubbed` 8b/8c, `test_team_config_stubbed`. LH's
   episode is still step 4's successor.
6. **Done 2026-09-12 (out of order with 4 and 5, at the user's request).**
   HL's `route.json`: five two-node routes chaining through shared supports
   (M5's destination is M3's checkpoint). Its BDDL goals are the destinations
   now, `allow_trivial_goal` is removed, and at load all five are unsatisfied.
   The run-level acceptance -- no longer ending at env_step 0 -- is step 4's
   episode.
7. **S1 done 2026-09-12**: LH (LL's ten nodes, notebook, `lift` arm-only) and
   HH (HL's five routes, notebook) alongside HL; all four BDDLs grown to the
   same 13-object shape and re-sampled, every node bound, goal false at load.
   Two things learned writing them, both now CPU-checked in
   `test_route_spec_stubbed.py`: a `*` anywhere in a definition -- a comment
   included -- is a wildcard to `_strip_wildcards` and crashes
   `Environment.__init__` (two GPU runs to find); and COOHAVIOR's spawn
   annotations (`spawn_nodes.bddl_support` = a fixture) disagree with its own
   `initial_relation` and staged box prims (the floor) -- the prim is what
   spawns, so boxes start on floors and no node is true at load. **S2/S3's
   eight files remain**, per scene once those are ported.

## 7. Traps, known before starting

* **Supports have to be awake too.** `OnTop` is a contact query and a slept
  actor reports none. `_keep_task_objects_awake` covers what is in
  `task.object_scope`; adding the supports to `:objects` puts them in scope,
  which is one more reason 5.1 insists on it.
* **A floor node collapses.** Merom_1_int has one floor object per room, so
  a checkpoint whose support is "the kitchen floor" and a spawn on that same
  floor are the same predicate. Every V4 checkpoint is a fixture, so this
  bites only if a route file is written with floors; the loader should warn
  when two consecutive nodes share a goal.
* **State detection sees the box where it is now.** Verified safe under
  the current engine (a placement terminates a primitive, and the check runs
  at that boundary); re-verify if primitives ever stop aligning with
  macro-steps.
* **`check_goal` may be true before the route is complete** -- HL at t=0 is
  the extreme case, and any route that revisits its start is another. That
  is why `terminated` must not be `check_goal` alone once a route file
  exists, and why the disagreement is printed rather than resolved silently.
* **A node's cargo is a BDDL instance, not a name pattern.** COOHAVIOR binds
  a box to its route by regex on `box_s1_m5`. Ours is the `cargo` field, and
  a node whose goal names a different object than its route's cargo is a
  load error.
* **`by` is inference.** Two robots ending primitives on the same tick with
  the cargo changing hands leaves the credit `None`. Do not guess; report
  `unattributed` in `S_team`.
* **Post-episode code fails only after a full GPU run.** Every saver and
  metric reader added here gets a CPU test that runs it on fabricated logs
  first (CLAUDE.md, "the recurring defect class in this port").

## 8. Decisions left to the user

1. **Out-of-order: record only (proposed) or penalise.** Section 4 argues for
   recording only.
2. **Show the whole route or only the next node.** Proposed: the whole route,
   marked, as COOHAVIOR does; the room listing stays local either way.
3. **Whether `Y_task` counts nodes (10) or legs (5) for L\*.** Nodes match
   COOHAVIOR's own metric and are proposed.
