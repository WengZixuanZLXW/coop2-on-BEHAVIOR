# coop2 — COOP² multi-agent cooperation on BEHAVIOR-1K

Port of `coop2-llm-mas/ma_crafter`'s LLM multi-agent cooperation stack onto
BEHAVIOR-1K. Goal: run **individual / broadcast_chain / centralized** topologies
here with minimal changes to the COOP² code, then add **decentralized**.
Repair is explicitly **not** ported.

**Read `coop2/PORTING_PLAN.md` before designing anything.** It is the canonical
design doc: seven-layer structure, the full environment-contract checklist that
COOP²'s upper layers require, the BDDL multi-agent audit, and the trap list.
This file is only the operational summary.

## Layers

| layer | package | status |
|---|---|---|
| L6 experiment (runners, grid, metrics) | `coop2/experiment/` | **all three runners GPU-verified against a real LLM**; `inspect_scene.py` loads a scene without an episode |
| L5 comm_topology (individual/chain/centralized) | `coop2/comm_topology/` | **all three exercised at 9 and 12 robots**; `llm_team.py` makes a *team* the unit of every topology |
| L4 cognitive/agent (FSM, memory, broker, prompts, LLM) | `coop2/cognitive/agent/` | **BDDL vocabulary aligned, FSM GPU-verified**; prompt scoped to the activity's objects, with plan history |
| L3 cognitive/plan (plan lifecycle, PlanningEnvWrapper) | `coop2/cognitive/plan/` | **GPU-verified driving the facade** (2026-09-06) |
| L2 cognitive/action (symbolic action → primitive) | `coop2/cognitive/action/behavior_action.py` | **rewritten, GPU-verified** (M5) |
| L1a world model | `coop2/behavior_env/world_state.py` | **done, CPU-tested + GPU-verified** (M4) |
| L1b text observation + target_hints | `coop2/behavior_env/symbolic_view.py` | **done, CPU-tested + GPU-verified** (M4) |
| L1c primitive execution engine | `coop2/behavior_env/primitive_engine.py` | **done, GPU-verified** 2026-09-05 |
| L1d task tracker | `coop2/behavior_env/cooperative_tasks.py` | **done, M6 PASSED** 2026-09-07 |
| L1 symbolic nav fix + contention | `coop2/behavior_env/symbolic_{navigation,contention}.py` | **done, GPU-verified** 2026-09-05 |
| L1 placement (scene-derived poses) | `coop2/behavior_env/placement.py` | **done, GPU-verified** |
| L1 facade | `coop2/behavior_env/coop_env.py` | **done, GPU-verified** (M5) |
| L0 N-robot env config + startup ritual | `coop2/behavior_env/env_setup.py` | **done, GPU-verified** 2026-09-05 |
| — repair shim, viz stub | `coop2/_repair_shim/`, `coop2/cognitive/viz/` | **done** (no-op by design) |
| BDDL tasks + cached instances | `bddl3/.../{coop_two_apples_pomaria,coop_nine_apples_hall,v4_s1_v4_ll}/`, `feasibility_verify/sample_*.py` | **three activities, all GPU-verified and all solved**; two apples solves, v4_s1_v4_ll at env_step 504, nine apples at 4434 (2026-09-12) |
| Robots from outside BEHAVIOR | `coop2/robot_configs/`, `coop2/omnigibson_definitions/fix_*.py` | **Ridgeback+UR5, Jackal and Crazyflie imported from upstream URDFs** (2026-09-11); URDF colours restored by `fix_robot_visual_materials.py`; the Crazyflie holds altitude (driven z joint owned by `arm_0`, `base_footprint_link_name: world`, navigate keeps world z) (2026-09-12) |
| COOHAVIOR S1 family | `bddl3/.../v4_s1_v4_{ll,lh,hl,hh}/` (each with `route.json`), `feasibility_verify/sample_v4_s1_task.py`, `coop2/team_layouts/s1/sets_{1..5}.json` (shared spawn in living_room_0, drones hovering at 1.2 m over their Jackal; generator `make_s1_shared_spawn_layouts.py`) | **all four routed, BDDLs corrected to the routes, re-sampled with every node bound and goal false at load** (2026-09-12); cargo is `die.n.01` (8 g, arm or drone) or `notebook.n.01` (20 g, arm only) per `tasks.modified.json`; HL's goals are its routes' destinations, so it no longer ends at env_step 0 |
| M9 wiring (BehaviorTask + `check_goal` termination) | `coop2/behavior_env/coop_env.py` | **done**, commits `e4d28a98`…`5fda3d28` |

## Where we are (2026-09-12)

**All three BDDL activities now reach their goals.** `coop_nine_apples_hall`
was the one that never had -- 6 of 9 apples in 8000 steps was its ceiling --
and `individual`, twelve robots as three teams of four, solved it at env_step
**4434** on 2026-09-12. Twelve robots as three teams of four, and nine as three
teams of three, run all three topologies without stalling.

**In progress: route supervision.** A COOHAVIOR task is an *ordered* sequence
of `ontop` sub-goals on one box, and BDDL can only state the last one -- which
is why `v4_s1_v4_hl` ends at env_step 0. `coop2/ROUTE_SUPERVISION_PLAN.md` is
the design; steps 1-3 are done (the `route.json` sidecar, its loader, LL's
route, the `RouteTracker`, and the wiring that lets it decide `terminated` --
see "The route file" and "Route supervision, wired" below), and the S1 part of
steps 6-7: all four S1 tasks have route files and corrected, re-sampled BDDLs
("Four route files, four re-samples"), and step 5, the lift gate ("The lift
gate"). Step 4, the LL episode against the route, has run twice: two of ten
nodes, then five of ten after the fixes ("Step 4's first episode", "The
acceptance run"); termination on `D` is still untested. Not started: S2/S3's
eight route files. Read the plan before touching termination or the `YOUR
TASK` block.

**Two lines of work are open**, and neither is a milestone from
PORTING_PLAN.md section 7:

* **What the agent is shown.** The prompt is now scoped to the activity's own
  objects, carries the agent's plan history, and states the task in the ids it
  is written in. See "What the agent is shown" below -- including the change
  that fixed one task by breaking the premise the benchmark rests on, and how it
  was undone.
* **Heterogeneous robots.** V4's Ridgeback+UR5 and Crazyflie are imported from
  their own upstream URDFs, which is the first real exercise of the robot-model
  registry. See "Third activity: v4_s1_v4_ll".

**Anything measured before 2026-09-11 is not comparable to anything measured
after**, on four counts, each enough on its own: the travel charge halved
(60 -> 30 ticks/m), every robot but two was shown to the others under another
robot's name, the prompt's object listing changed shape, and the listing is now
scoped to the activity. Re-run rather than compare across that line. Every
results table in this file that predates it has been removed for that reason;
what was learned *from* those runs is kept in the sections that explain it.

Still on the plan and not started: **M7 step 3** (>=3 seeds per topology for the
metrics table) and **M8** (decentralized topology). Note
`build_results_table.py` cannot read these runs: it skips any folder without
`team_score.json`, which the runners do not write while `team_score.enabled` is
False -- post-episode code that only runs after a full GPU episode, the same
defect class as the rest of that file.

## M9 is wired, and the scene it runs in

`coop_env` loads OmniGibson's `BehaviorTask` from the cached instance when
`bddl_activity` is set, and `compiled_task.check_goal` is the **only** authority
over `terminated`. M1-M6 and M9 passed; acceptance criteria are in
PORTING_PLAN.md section 7.

The two-apple task is on `Pomaria_1_int`/`living_room_0` because that is where
the two-armchair + coffee-table layout exists. The earlier target was
`house_single_floor`; `Rs_int` was measured unusable, 96.2 % of sampled base
poses reject.

*(The 2026-09-08 and 2026-09-09 per-topology result tables were here. They were
measured at 60 ticks/m, on the one-floor template, before the team layer and
before the naming and prompt fixes -- four independent reasons they cannot be
compared with anything current, so they are gone rather than misleading. The
findings they produced are in the sections below, which is where they were
load-bearing.)*

## The bug that made the activity look unsolvable (2026-09-08)

`OnTop` is `Touching` and `Touching` is a contact-report query, and **a sleeping
PhysX actor emits no contact reports**. Every OmniGibson object is constructed
with `DEFAULT_SLEEP_THRESHOLD = 5e-05` (`entity_prim.py`). An apple placed on the
coffee table falls the sampler's 2 cm z-offset, comes to rest, and is slept --
after which it is still sitting on the table and `OnTop` reads False forever.

Our own placement path is what makes it arrive so fast: `_place_with_predicate`
does release -> `set_position_orientation` -> `keep_still()` -> settle, and
`keep_still()` zeroes both velocities. `keep_still()` was added by the *previous*
fix, to stop the object being flung out of the house; it cured the fling and
brought the sleep forward. Both were real.

Two readers were being lied to, which is why the fix is central
(`coop_env._keep_task_objects_awake`, applied at build and after every reset) and
not at either call site:

- `_place_with_predicate` raised EXECUTION_ERROR "it did not come to rest there"
  for a third to a half of all placements -- which reads as a physics failure and
  is not one;
- `check_goal` cannot see a delivered apple that has gone to sleep, so **no run
  could have reported success no matter how well the agents played**. That is why
  "one of two apples delivered" was the ceiling for days.

Measured, not argued -- `feasibility_verify/measure_placement_rest.py`:

| | before | after |
|---|---|---|
| placements that come to rest, empty table | 10/20 | 20/20 |
| placements that come to rest, one apple already there | 11/20 | 20/20 |
| single-step `OnTop` reads of a settled apple | 0/20 | 20/20 |

The discriminating evidence: success and failure are geometrically identical --
same sampled and final z (0.415 -> 0.390), same `VerticalAdjacency` below/above
lists, and the radius-from-centre medians order *oppositely* in the two
conditions, i.e. noise. The neighbouring apple makes no difference (63 % vs
67 %), which killed the first hypothesis. The failing pose reports no contact
with **anything**, at velocity exactly `[0,0,0]`; `is_asleep` is the only thing
that differs, and `wake()` plus one step flips `OnTop` to True with the object
not having moved. Three geometric hypotheses were proposed and all three were
measured wrong before this one was measured right.

## Two things the seed-3 runs left behind (2026-09-09)

### `--steps` is not the budget: `--time-limit-seconds` defaults to 120

`run_*.py` takes a wall-clock deadline (`run_individual.py:326`) that defaults to
**120 s** and stops the episode independently of `--steps`. At ~26 env_step/s
that caps a run near 3100 steps, so `--steps 4000` has never once been reachable
and every run "to 4000" was really a run to 120 s. Individual seed 3 stopped at
3174 for exactly this reason.

Re-run with `--time-limit-seconds 0` to check: it still did not solve, so this
particular result stands. But the deadline is invisible in the run folder --
nothing records which limit fired -- and it silently makes `--steps` a
lower-bound-only knob. Pass `--time-limit-seconds 0` when the step count is
meant to be the experiment variable.

### The coffee table can fly away (seen once, not yet diagnosed)

In the 4000-step individual re-run, `coffee_table_gpkbiw_0` left the living room
under constant velocity: the `NO_SPACE_AROUND_TARGET` attribution logs it at
(-10.5, -10.1), then (-14.5, -17.9), (-17.6, -24.1) ... (-46.4, -80.8), roughly
equal steps in a fixed direction -- coasting, not accelerating, i.e. it carries a
velocity nothing damps. 97 rejections, all `{'room': 200, 'trav': 0, 'robots': 0}`:
every candidate pose was rejected for being outside the room, because the target
had left it. 18 of that run's 21 plans died this way (Y_plan 0.05).

Same shape as blocker #6 but on the *receptacle*, which is supposed to be
furniture. Not present in any of the three 120 s seed-3 runs, so it is an event
during the episode rather than a step-count threshold; the obvious suspect is a
placement impulse, and `_keep_task_objects_awake` keeps the table from ever
sleeping it off. Evidence kept in
`coop2/runs/individual_agents2_repair_off_seed3_20260909_025858_415866/`.

## Two bugs the viewport exposed (2026-09-09)

A robot vanishing on its first teleport and another toppling mid-episode turned
out to be three separate defects, found by measuring the startup stages apart
(`COOP2_PLACEMENT_VERBOSE=1`, and
`feasibility_verify/measure_base_joint_limits.py`).

**1. The template's robots were the ones being simulated.** `include_robots:
False` gates the robots in the scene *USD*; it cannot stop entries restored from
the task's own cached json, and the template *is* the scene file. Proof was in the
root anchors: (300, 300, 300) and (-52, -50, 0) are the sampler's, not coop2's
`[1.5 * i, 0, 0.05]`. The state being baked in was junk -- `agent_1` stored with
its base z joint at **-0.681**, i.e. 0.68 m below the floor, and `agent_0` with a
root anchor 300 m out and every non-base joint at zero rather than the tucked
reset pose its own init_info declares. This is also what
`_enforce_controller_config` was patching after the fact. Fixed by stripping robot
entries from the template (`strip_robots_from_template` in the sampler, applied to
the existing template in place so the object layout is untouched). Now the roots
read (0, 0, 0.05) and (1.5, 0, 0.05), and both robots come up level at 0.51 deg
instead of 0.51 and **18.52**.

**2. Teleporting a robot did not zero its velocity.** Same defect as the object
placement that used to fling apples out of the house. Because `agent_1` began
underfloor, physics was already ejecting it and it left `place_robots` at
**2.72 m/s**; the orientation it is given is level, but momentum that survives the
teleport tumbles it during the next settle. `place_robots` now calls
`keep_still()`. Worth knowing this fixed the velocity but *not* the tilt -- the
tilt was defect 1, and measuring after each fix is what separated them.

**3. `if robot.is_grasping(...)` accepted a definite NO.** `IsGraspingState` is an
IntEnum: `TRUE = 1`, `UNKNOWN = 0`, **`FALSE = -1`**. So a truthiness test is true
for FALSE and false for UNKNOWN. And FALSE is precisely what a robot holding apple
A returns when asked about apple B: the gripper controller reports TRUE for
"closed on something", then downgrades to FALSE when the fingers turn out not to
touch the object asked about (`robots/robot.py`). So `holder_of` named a holder for
every object in the scene, and **the second apple went permanently
OBJECT_CLAIMED the instant the first was picked up**, with `place_on_top` failing
PRE_CONDITION on an empty hand -- the ALREADY_HELD / OBJECT_CLAIMED /
PRE_CONDITION triad in the plan log. Latent for weeks: with the template's
mis-configured robots `is_grasping` raised and the `except: break` hid it, so
fixing 1 and 2 is what surfaced it. Both call sites (`holder_of` and
`world_state`'s held-object confirmation) now go through
`is_definitely_grasping`. The stubs used to return a plain bool, which is why no
CPU test could catch it; they now return the tri-state and one does.

## Placement no longer drops the object first (2026-09-09)

Upstream's `_place_with_predicate` calls `_release()` -- which is
`release_grasp_immediately` **followed by its own settle** -- and only teleports
afterwards. Those settle ticks are the object in free fall from gripper height, so
on screen the apple dropped to the floor and *then* jumped onto the table. The
release is now split: detach, teleport, `keep_still()`, settle once. Measured with
a per-tick height trace: the apple goes 1.223 m (gripper) -> 0.409 (sampled pose,
table top is 0.358) -> 0.390, with **0 ticks below 0.15 m**. A placement also went
from ~150 ticks to 101, because a whole settle phase is gone.

## The place/grasp distance gate, measured (2026-09-09)

There is a gate -- `_require_near(obj, verb, GATE_PLACE)` -- but it is looser than
it looks. `interaction_radius_for` is the navigation annulus's upper bound plus
0.35, and that is deliberate: the gate must accept anything `navigate_to` can
produce, or an agent loops navigate -> TOO_FAR forever. The consequence is that
the slack is `reach (1.5) + radius_margin (0.35) + clearance_margin (0.05)` =
**1.90 m of clear floor between the robot's edge and the object's edge**,
identical for every object, because object size only enters through the
half-diagonal that sets the non-overlap clearance:

| object | footprint | clearance | nav annulus | gate radius | edge gap |
|---|---|---|---|---|---|
| apple | 0.08 x 0.08 | 0.72 | [0.72, 2.22] | 2.57 | 1.90 m |
| coffee_table | 0.74 x 1.45 | 1.48 | [1.48, 2.98] | 3.33 | 1.90 m |
| armchair | 0.78 x 0.68 | 1.18 | [1.18, 2.68] | 3.03 | 1.90 m |

R1's arm reaches well under a metre, so placement from 1.9 m of clear floor is not
physically plausible. Tightening is one parameter -- `reach` -- and both the
annulus and the gate move together, so they cannot disagree. It is also close to
free: the outer ring of the wide annulus mostly falls outside the room or on
non-traversable floor, so shrinking it *raises* pose acceptance (coffee table
48.8 % at reach 1.5, 60.0 % at 0.6). Measured at reach 1.5/1.0/0.8/0.6/0.4; only
0.4 starts costing apple poses.

**`reach` is now 0.6 and `DEFAULT_RADIUS_MARGIN` is 0.05** (user, 2026-09-09).
The margin used to be 0.35, which quietly added a third of a metre to every
manipulation gate; it is now only what it claims to be -- a guard for the float
comparison and the millimetres a base drifts during a settle -- so **`reach` alone
is the distance an agent may reach across**, and every non-navigation verb
(grasp, place, open/close, toggle) gates on it through `interaction_radius_for`.

Resulting gap is a uniform **0.70 m** of clear floor: gate radius 1.37 m for an
apple, 1.83 m for an armchair, 2.13 m for the coffee table. The *gap* is what is
uniform, not the centre distance, and it cannot be the latter -- the coffee
table's own clearance is 1.48 m, so a centre distance tight enough to mean
anything for an apple would make the table unreachable from any pose.

Measured on the same seed, tightening 0.8 -> 0.6 **helped**: goal at env_step 1230
against 1485, and `TOO_FAR` 15 -> 11, with `NO_SPACE_AROUND_TARGET` still 0. A
narrower annulus puts the sampled standing pose closer to the target, so the
travel charge shrinks. **The charge is 30 ticks/m since 2026-09-11** (it was 60
when this was measured); travel is most of the tick budget, so no step count
recorded at 60 is comparable with one recorded now. Any metric recorded before
this has different
travel costs and is not comparable.

## Agent start poses are sampled, but seed-determined (2026-09-09)

`place_robots` calls `th.manual_seed(seed)` and then samples, so a run's robot
poses are a pure function of (scene, room, seed, robot geometry) -- not random
run to run, and not a fixed configured pose either. Verified by re-running
`place_robots` on one loaded scene: seed 0 -> (-7.55, 0.05)/(-10.15, 0.85),
seed 1 -> (-11.35, 1.85)/(-11.75, -1.95), seed 2 -> different again, and
returning to seeds 0 and 1 reproduced both exactly. `seed=None` differs every
call. Today's logs are the same result from the other direction: every seed-0 run
printed one of exactly two placements, and the switch between them is the commit
that stopped simulating the template's robots -- because `robot_radius` changed,
which changes both the trav-map erosion and the separation requirement.

Consequences for the metrics sweep: **the seed varies the agents' start poses and
nothing else about the scene.** The object layout is frozen in the cached
template, so across seeds only the start poses (hence travel distances and who is
nearer which apple) and the LLM's own sampling differ. Placement constraints are
also worth stating: one room, mutual separation >= 2 x robot_radius (1.236 m),
within `cluster_radius` 6 m of the first robot, and on traversable eroded floor.

## Why an agent flashed in the kitchen at startup (2026-09-09)

Robots are *created* at the placeholder poses in `build_multi_robot_config`, and
`og.Environment`'s construction and reset render several frames before
`place_robots` moves them into the task room. Those placeholders were
`[1.5 * i, 0, 0.05]`, and (0, 0) is inside the house -- `kitchen_0` in
Pomaria_1_int, whose floor spans x [-13.7, 1.1] -- so every episode opened with
both agents visible in the kitchen for a moment. It only became visible when the
template's robots stopped being the ones simulated (they came up already in the
living room). Placeholders are now parked at (-50 - 2i, -50), outside the floor
plan: nothing renders inside the house before placement, and having no floor for
those few frames is harmless because `place_robots` zeroes velocity on arrival.



## Primitive latency, measured (2026-09-09, reach 0.8)

`feasibility_verify/measure_primitive_latency.py`. One tick is one `env.step`,
i.e. 1/30 s at the default action frequency.

**Measured at 60 ticks/m, which is no longer the charge.** Halve every travel
figure below for the current 30, and note that the budgeting paragraph at the
end is stated in the old units.

| primitive | ticks | seconds | what sets it |
|---|---|---|---|
| `grasp` | **101** (n=4, no spread) | 3.4 | one `_settle_robot` |
| `place_on_top` | **101** (n=4, no spread) | 3.4 | one `_settle_robot` |
| `navigate_to` | **~100 + 60 per metre** | 5.9-11.0 measured | travel charge + settle |
| `wait(n)` | **n + 1** | n/30 | exactly what it is asked for |
| any rejected precondition | **51** | 1.7 | `apply_ref` settles after catching |

So ~100 ticks is the floor of every physical primitive, and `navigate_to` is the
only one whose cost varies -- `DEFAULT_TRAVEL_TICKS_PER_METER` (60 then, 30 now) is charged on
the distance to the **sampled standing pose**, not to the object's centre. Verified
against the `[nav]` lines: 3.8 m -> 227 ticks, 2.1 m -> 129, 0.9 m -> 53. Do not fit
ticks against centre distance; the pose is anywhere in the annulus and the fit
invents a slope (it suggested 31 ticks/m).

**Budgeting an episode**: one apple is navigate + grasp + navigate + place, so
roughly `4 x 100 + 60 x (d1 + d2)` -- about 500-700 ticks for in-room distances,
and both apples in parallel put the goal around env_step 1300-1500, which is what
runs actually report. A failed action adds 51 ticks and, because it terminates the
plan, one more LLM round trip.

Note these are much cheaper than the pre-2026-09-08 figures (grasp 500,
place_on_top 750, navigate ~490-550) quoted elsewhere in this file: those were
inflated by upstream retrying a placement whose predicate check was lying (the
sleeping-apple bug) and by the release settle that placement no longer performs.

## Second activity: coop_nine_apples_hall (2026-09-09)

`bddl3/bddl/activity_definitions/coop_nine_apples_hall/problem0.bddl` -- nine
`straight_chair`s and one `coffee_table-cjjayg` in `hall_glass_ceiling`'s
`empty_room_0`, an apple on each chair, goal = all nine on the table. Sampled the
documented way (`feasibility_verify/sample_nine_apples_hall.py`:
`online_object_sampling: True` -> `Environment(...)` -> `save_task()`), with the
model pinned via `sampling_whitelist` and the robot entries stripped afterwards so
`--agents` still controls the robot count.

Verified: BEHAVIOR's own `verify_definition` passes, all nine apples sit on their
own chairs after a 300-step settle, the goal is False at t=0, and the cached
instance loads with **nine** robots (9 apples + 9 chairs in scope, table is
`coffee_table-cjjayg`).

Notes worth keeping:

* `hall_glass_ceiling` has rooms `empty_room_0`, `corridor_0`, `bathroom_0`, and
  contains **no chairs and no tables** -- so every chair and the table are
  *imported* objects. That is why the init block uses
  `(ontop straight_chair.n.01_N floor.n.01_1)` and not `inroom`: `inroom` binds
  to furniture the scene already has. The `ontop ... floor.n.01_1` idiom is the
  standard one, used by 1008 shipped activities.
* `tests/bddl_tests.py batch_verify` **cannot run in this checkout** -- it
  hardcodes `parse_domain("omnigibson")` and only `domain_behavior-1k.bddl` and
  `domain_behavior-100.bddl` exist, so it dies before reaching any activity. Run
  `verify_definition` directly with `parse_domain("behavior-1k")` instead.
* **The distances are the problem, not the sampling.** `ontop floor` lets the
  sampler use the whole room and this hall is ~50 x 57 m. The cached draw puts
  chairs 4.7-43.0 m from the table (nine round trips = 550 m ~ 33 000 travel
  ticks at 30 ticks/m -- 33 000 at the 60 it was measured at), and `place_robots`
  starts the team ~47-53 m away because
  it clusters around wherever the first pose lands.

  That estimate was "of the order of 60 000 steps". **Measured instead: 8000
  steps gets 6 of 9 apples** with 9 or 12 robots (2026-09-11 sections below), so
  the parallelism is worth more than the per-robot round trips cost. The task is
  still unsolved at that budget -- it is the hard one of the three, and it is
  meant to be.

## Heterogeneous robots and one-LLM-per-team (2026-09-10)

Three changes, done in order, each verified before the next.

### 1. A team layout JSON says who is in the scene

`coop2/behavior_env/team_config.py`, `--team-config PATH` on the runners. One
file carries what each robot is, where it starts and who it works with, because
those three are not independent -- a team is a set of named robots, and a name
means nothing until that robot has a model and a place to stand.

    {"robots": [
      {"name": "agent_0", "model": "R1",    "position": [-8.5, -1.5], "team": "alpha"},
      {"name": "agent_1", "model": "Tiago", "room": "living_room_0",  "team": "alpha"},
      {"name": "agent_2", "model": "R1",    "room": "kitchen_0",      "team": "bravo"}
    ]}

Exactly one of `position` (exact world coordinates) or `room` (a room *instance*,
sampled inside it) per robot. `place_robots` applies pinned poses first so that
sampled robots keep clear of them rather than the reverse, and `cluster_radius`
became per room -- keeping everyone within 6 m of the first robot placed is
unsatisfiable once robots are in different rooms. A pinned pose too close to
another warns rather than refuses: two robots deliberately close together is a
legitimate thing to study.

Verified on GPU with `coop2/team_layouts/pomaria_mixed.json`: 3 robots built
(the layout beat `--agents 99`), models r1/**tiago**/r1, the pinned robot at
drift 0.000 m, and the other two in `living_room_0` and `kitchen_0` as asked.

**Robot models are a registry, not a fixed list**, because importing non-BEHAVIOR
robots is planned. Such a robot needs one thing from coop2 -- a primitives-style
YAML carrying the controller stack the symbolic primitives require -- so that is
the whole extension point: `"config": "/path/to/myrobot_primitives.yaml"` in the
layout, or `register_robot_model(name, path)` at import time. Nothing else here
asks what kind of robot it is driving; `q_to_action` and the base joints are the
only interface. Only r1, r1pro and tiago ship such a YAML.

### 2. One LLM per team -- the unit of all three topologies

`coop2/comm_topology/llm_team.py`. **Not a fourth mode.** individual /
broadcast_chain / centralized describe how *teams* address each other; inside
every team one LLM plans for all its robots. All three runners take
`--team-size K` (or `--team-config`) and default to 1, which is exactly their old
one-LLM-per-robot behaviour -- a generalisation, not a replacement. There is no
`run_team.py`.

**A team is the address, on both sides.** `wait_for` and `send_to` hold *team
names*; `TeamBrain._say` goes through `MessageBroker.send_team_message`; the log
records `sender: "team_0", recipients: ["team_1"], sender_type: "team"`.
Delivery still reaches every robot of an addressed team -- an interrupt has to
stop all of it, or the team splits across I and W and stalls its own interrupt
barrier -- but that expansion is the broker's (`delivered_to`), not the address.

The wiring used to be in agent ids: a team was addressed as its four robots and
waited on through whichever member "spoke for" it. Three things followed from
that, and all three are gone with it -- every send was credited to a robot that
had no part in composing it; one conversation between two teams appeared in the
log as four; and a team's own inbox, being the union of four robot inboxes,
quoted the same message four times into its next prompt. `_collect_heard`
de-dupes on (sender, timestamp, content) for the same reason.

Three brains carry the roles: `ChainTeamBrain` waits on the preceding team's
speaker then broadcasts its allocation onward; `LeaderTeamBrain` interrupts the
follower teams for status, waits, then plans; `FollowerTeamBrain` answers from
its own state rather than spending an LLM call to paraphrase what it already
knows. What a team heard goes into its next prompt.

The prompt says so too. `build_team_system_prompt` (not `build_system_prompt`,
which tells a model it *is* a robot and asks it for one plan) opens with "You
command TEAM 'team_0' -- 4 robots (...)" and carries `TEAM_ENV_DESCRIPTION`, the
same world in the third person: "a robot must be closer than", "give it wait",
"never give one robot an id that appeared only under another robot". The two
descriptions are separate texts and drift silently, so
`test_team_prompt_stubbed.py` fails if a shared rule leaves either one.

A team shares one brain: every member's observation goes into one prompt and the answer
is one plan per robot, so the allocation is made once and is visible. Four robots
on the two-apple task, one call:

  [team_0] allocation: The two apples are split between agent_1 and agent_3, the
  robots currently positioned within reach of distinct apples. Agents 0 and 2
  wait rather than duplicating claims.

**The barrier is the whole design problem, and it deadlocks if done naively.**
The requirement is that members return to reasoning together, once all have
finished. But the plan loop does not step the env while any agent is not ready
(`run_*.py`: `while not all(agent.ready)`), so a member that finished early and
simply waited would freeze the world -- and its teammates need the world to
advance to finish. That deadlocks on the first uneven round.

So an early finisher stays **ready** and holds position: a short `wait` plan,
repeated until the team is complete. The idle time is real and shows up in the
plan log as `wait_for_team(...)`, which is the honest price of a joint decision
point. `TeamBrain._awaiting` tracks who is done *across* those holds, so a member
idling three times is still "waiting to plan", not "planning again".

Messages interrupt the whole team; the brain answers resume-or-replan per robot
in one call. Per robot because a message that changes one robot's job usually
leaves the others' plans good. No deadlock there -- an interrupt reaches every
member at once.

One call per round, not N: the runners spawn a thread per agent, so the brain
locks and the first thread through makes the call while the rest read the result.

A team of one behaves exactly like the individual topology, which is why an
unteamed robot gets a team of its own: one-LLM-per-robot is this code at N=1,
not a second code path. `--team-size K` groups `--agents N` without a JSON file.

**Holding is decided against the barrier, not against the plan dict.** Two races
made a member idle through its own team's decision, and both were found in a
run's `agent_states.json` rather than by reading:

* A member that got "hold" and was still acting on it when the last teammate
  arrived is in R, which `_recall_holders` deliberately leaves alone -- R is not
  an idle state. It then went ready with a hold and sat in W for the whole call
  (measured: 15.8 s of `waiting` against its teammates' 15.8 s of `reasoning`).
* Asking the brain once more before committing to the hold is not enough: the
  recall runs *before* the call, so at that moment there is no plan to hand back.

`claim_pending_plan` settles it on whether the team is complete. If it is, the
member waits in R for the round to finish -- safe, because a frozen world is
exactly what the team is waiting on and the call needs no simulation -- and
counts `rounds` to tell "the call has not started" from "the call finished
without me". If it is not, teammates really are still working, and it holds.

**A robot has no role and no speaker order of its own any more** -- both belong
to its team. Three leftovers from the per-robot world were each caught by a run
rather than by reading: a helper that built agents before `team_layout` was in
scope, `agent.speaker_order` in the chain runner, and
`isinstance(agent, LLMLeaderAgent)` in the centralized one. They read
`agent.brain` now.

**Trap, found by the first GPU run.** The hold plan was built through
`parse_plan_response`, whose `_ensure_task_terminal_action` appends an action
matching the plan's TaskSpecification. Written as `holding(<self>)` that appended
`grasp(<self>)` -- a robot planning to pick *itself* up, which really appeared in
a run's plan log. Holds are constructed directly now, and the test asserts a hold
has exactly one action rather than only checking the first.

### 3. The model is a flag, and the runs must agree on it

No code change -- `--model gpt-5.6-terra`. Both terra and luna honour the team
schema (probed directly). A 2026-09-10 pair of runs had terra allocating both
apples on its first team call where luna needed two rounds; that comparison is
not repeated here because it predates the naming and prompt fixes, and one seed
was never a ranking anyway.

What survives is operational: `AZURE_OPENAI_MODEL=gpt-5.6-luna` is pinned in
`.env` so a forgotten `--model` cannot silently fall back to the code default
(`gpt-5.2-chat`), and **runs before 2026-09-11 are a mix of luna and terra**.
`llm_usage.json` records which. Align that column before any cross-topology
comparison means anything.

## Twelve robots, three teams, 8000 steps (2026-09-11)

The day's work, in the order the defects surfaced. Every number below is read
off a run folder, not estimated.

### The timeline was drawing idling as work

A team hold and a real primitive are both FSM state X, so only the plan's
specification separates them -- and most holds were never written down.
``_recall_holders`` ends one by assigning ``plan.status = INTERRUPTED``, which
it has to: ``needs_new_plan()`` asks the plan, and a member left in R with a
live one hangs the run. But nothing then *filed* it, so the plan sat in
``logger.current_plans`` until the replacement overwrote it and was gone. Only
the hold still running at episode end survived, archived by
``terminate_unfinished_plans`` -- which is why every filed hold in
``centralized_agents12_..._045959`` ends at exactly 4000.

Measured on that run: **87 % of the idling (11 612 of 13 303 robot-steps) had no
record at all.** agent_5 held 588->2296, 43 % of the episode, and plan_logs said
nothing.

Two fixes, and they are separate on purpose:

* ``log_plan_ended_elsewhere`` files a plan something else already marked
  terminal, and ``_sync_committed_plans`` calls it on the branch that used to
  fall through. That branch also reaches ``_reset_symbolic_action_state``, which
  it never did before -- see the next item.
* The timeline no longer *depends* on a hold having been filed. A step-gap can
  only open while the world is moving, and the world does not move while any
  agent is in R or W, so an env_step no plan covers is a robot standing in X
  with nothing to do. Reading the gaps makes runs recorded before the fix
  legible too.

Drawing them needed one more change. ``_inside_hold`` asked whether a whole span
sat inside one hold range, and ``_coalesce`` folds the sub-pixel R/W seam between
a primitive and the hold that follows it -- so the merged bar belongs to neither
and the bars worth shading were exactly the ones it refused. Spans are cut at
hold boundaries now, interpolating the split from the env_step range, which is
exact while the env is stepping.

### A recalled hold left its 600-tick WAIT running

The same missing branch. ``_recall_holders``' docstring claimed the wrapper
aborted the stale primitive "when the replacement plan is committed"; it did
not, because the abort sat behind the pending/executing test the recalled hold
fails. So the next plan's first action could not be issued and **reported the
WAIT's outcome as its own**:

```
navigate_to {'target': 'apple.n.01_8'}  3288-3441  success  | prim: WAIT
   outcome: {"primitive": "WAIT", "ticks": 600, "started_env_step": 2840}
```

``started_env_step 2840`` is the tell: that WAIT belongs to the previous plan.
The robot stood still for 153 steps, then tried to grasp from 5.63 m and failed
TOO_FAR. **13 actions in that one run.**

### A plan that only waits is a decision, not a missing action

``_ensure_task_terminal_action`` guards against a model that states a goal and
lists only navigation. It was also firing on plans that listed only ``wait``.
Asked to plan for a team whose apples were all claimed, the model answered
``[wait(600)]`` for all four members -- "wait rather than duplicate their
targets" -- and the guard appended ``place_on_top`` to each, so every one ran
600 ticks and then failed PRE_CONDITION for placing with an empty gripper.
Thirteen plans. Replaying the real seq-12 response through the real parse path
is how it was pinned, not by reading.

### The fallback plan was crafter's

``_generate_fallback_plan`` -- what a robot gets when an LLM call raises -- was
``move(left, 2)`` then ``collect(wood)``, carried over with the port. Neither
verb exists in L2, so the engine answered ``invalid`` and ``do``. One wait
instead: the plan has to *do* something, because the loop does not step the env
while any agent is not ready, and waiting is the only honest thing a robot can
do when the round trip that would have told it something is what just failed.

### A request can freeze the whole simulation, so it has a deadline now

R freezes the env, so a hanging call stalls every robot, not one. One Team Plan
Generation in ``individual_agents12_..._060731`` spent its **whole 128k
completion budget on reasoning tokens**, emitted nothing parseable, and took
**552 s -- 42 % of that run's wall clock**, during which env_step went
3119 -> 3120. Healthy calls in the same run averaged 7.3 s; the slowest was 10.6.

``LLMClient.REQUEST_TIMEOUT_SECONDS = 100``. The SDK's own ``max_retries`` is
off wherever it applies, or the bound is not a bound: the timeout is per
attempt, so two retries make 100 s mean 300 s. A timeout raises, which lands on
the hold-position fallback above.

Note ``llm_usage.json``'s ``total_api_latency_seconds`` does **not** include
failed calls -- the error path does not record latency -- so that 552 s is
invisible there. ``total_llm_errors`` is the only hint.

### A chain team decided out of turn

Per robot this was solved already: ``handle_interrupt`` calls
``_wait_for_previous_speaker()`` *before* deciding, and a RESUME still
broadcasts (``resume_ack``), because with only REPLAN speaking 81 % of messages
died where they landed. The team layer had neither. ``ChainTeamBrain`` overrode
``before_plan`` only, so one message interrupted every team behind the sender
and they all decided at once. On ``broadcast_chain_agents12_..._064820`` team_0
spoke at 183.28, team_1 and team_2 both went to I in that instant, team_2
finished deciding at 191.58, and team_1 did not relay until 235.14 -- **team_2's
decision was 44 s older than the information it was ordered behind.**

``before_interrupt_decision`` waits for the relay and folds it into the prompt;
``after_interrupt`` relays whatever was decided. Blocking is safe here and is
not on the planning path: every team behind the sender is already in I, so none
of them needs the world.

Two things a team has that a robot does not, and both bit:

* **The exit cannot be readiness.** Per robot, ``_execute_flow`` broadcasts in
  step 4 and ``create_agent_thread`` calls ``set_ready()`` only once the handler
  returns, so speaking strictly precedes going ready. A team inverts that:
  ``_decided.set()`` releases four members who go ready while ``after_interrupt``
  has not sent. The round therefore closes *after* the send, and
  ``has_open_interrupt_round()`` is what the team behind reads.
* **Round scoping cannot use the clock.** The broker writes timestamps relative
  to the run origin and ``_interrupt_opened`` is an absolute ``time.time()``;
  comparing them is a few seconds against 1.7e9, false forever. Fingerprint what
  was already in hand instead.

Verified on 8000 steps: **11/11 cascade rounds ordered** -- team_1 and team_2
enter I in the same instant, and team_2's span always ends after team_1's relay
(it holds 4-10 s in I for it). One relay per team per round, resume rounds
included. The 30 s backstop never fired.

The **planning path is still on the old readiness test** and is a known hole:
``_await_speakers`` releases as soon as the team ahead is `ready`, and a team
ahead that is *executing* is ready, so team_2 planned alone at t=126.66 and
154.32 with nobody having spoken to it. Tightening it is not a one-line change:
blocking there deadlocks, because the team ahead may have robots that need the
world to finish. The waiting would have to happen in a hold (X), not in R.

### Every robot but two was shown to the others under another robot's name

The worst of the day. ``entity_id_for`` sent robots through the category
fallback, which numbers by scene-enumeration order -- alphabetical: agent_0,
agent_1, agent_10, agent_11, agent_2, ... -- while robots[0] took its id from the
task scope and so consumed no number. Every robot after agent_1 came out
shifted:

| prim name (what "you are ..." says) | shown to everyone else as |
|---|---|
| agent_0 | agent.n.01_1 |
| agent_1 | agent_1 |
| agent_10 | **agent_2** |
| agent_11 | **agent_3** |
| agent_2 | **agent_4** |
| agent_3 | **agent_5** |
| agent_4 ... agent_9 | agent_6 ... agent_11 |

Verified 12/12 on ``broadcast_chain_agents12_..._133540``: each robot's own
block is missing exactly the id it is hidden under. The shifted names live in
the same string space as the real ones, and the real ones are what everything
else uses -- the header, the team plan's keys, ``held_objects``. So a robot was
told "you are agent_2", saw a teammate called agent_2 in the same room, and
**every cross-robot reference in every prompt named the wrong robot.** Only
agent_0 and agent_1 happened to be right, which is why it survived.

A robot is its own id now, and ``adopt_task_scope`` skips the agent binding.
Nothing is lost: BDDL binds ``agent.n.01_1`` to robots[0] and to no other robot
*by design* -- declaring a second agent crashes the sampler -- so adopting it
renamed one robot of twelve into a different scheme than the eleven beside it,
and no goal predicate mentions an agent anyway.

### "You can do:" folded into the room listing

They were two sections over the same objects, so each was printed twice and the
reader had to join them by id: 76 object lines then 65 more, per robot, twelve
robots to a team prompt. The verbs sit on the line that names the thing now:

```
empty_room_0:
  - agent_1  (teammate)
  - apple.n.01_1  -> unreachable, navigate_to  [35.0 m away]
  - coffee_table.n.01_1  -> place_on_top, navigate_to
```

One robot block went **6243 chars / 162 lines -> 5121 / 95**, about 800 lines a
prompt. Anything the rooms did not carry -- a held object is in no room -- is
listed under "Also available" rather than dropped.

Two things broke on this and had to be found rather than predicted:
``_first_useful_target`` was parsing the old section, so a follower's report to
its leader silently lost the one field the leader allocates on; and the prompt
rules still told the model to read ids "under You can do:".

### The prompts costed travel at twice the real rate

``DEFAULT_TRAVEL_TICKS_PER_METER`` halved to 30 and both prose copies of it
still said **60**, so every plan was costed against a world twice as expensive
as the one it ran in. Nothing failed, because prose cannot disagree with code
loudly. ``test_symbolic_contention`` now reads ``prompts.py`` and fails if the
two part -- by reading the source rather than importing it, since that file
stubs OmniGibson out and ``prompts.py`` keeps the literal on purpose.

### The sweep, and why its table is not here

Three topologies, 12 robots as 3 teams of 4, `coop_nine_apples_hall`, 8000
steps, seed 0. No stalls, no tracebacks, all three ran the full budget and none
reached the goal; `individual` got the most apples on that one seed, so
communication bought nothing measurable. The numbers are omitted because they
predate the naming fix and the prompt work in the sections that follow, both of
which change what the model is shown.

What survives the change and is worth carrying forward:

* **The team barrier costs about a fifth of every robot's steps**, and it is now
  drawn and logged as idling rather than as work. The cleanest pair in the file
  is `broadcast_chain` at 12 robots and at 9, both post-naming-fix, same task,
  budget and seed:

  | | 12 robots (3x4) | 9 robots (3x3) |
  |---|---|---|
  | apples | 6/9 | 6/9 |
  | idle | 23.7 % | 21.9 % |
  | work plans | 183 (66 ok / 30 failed / 87 cut) | 125 (65 / 38 / 22) |
  | LLM calls | 55 | 55 |
  | tokens | 706 467 | **483 052** |

  Three more robots bought no apples and cost 32 % more tokens. The idle
  difference is small enough to be noise at one seed -- an earlier reading of
  36.9 % came from a *pre-fix* run and does not belong in this comparison. What
  is not noise is the interrupted count: 87 plans cut short against 22, i.e. the
  bigger team spends much more of its planning being overtaken by events.
* **The chain's ordering holds under load.** 11/11 cascade rounds ordered at 12
  robots, 6/6 at 9, exactly one relay per team per round including rounds where
  every robot resumed, and the 30 s backstop never fired.
* Every hold is filed now: the timeline's gap inference finds **0** unrecorded
  idling in post-fix runs, which is the check that the logging fix holds.

## What the agent is shown (2026-09-11, evening)

Three changes to the prompt, and one of them had to be undone the same evening.

### The scene listing was the whole scene

`coop_nine_apples_hall` is about 9 apples, 9 chairs and a table; the hall also
holds 34 spotlights, pictures, bookcases and light switches, and every one was
listed with its verbs. That buried the task's own objects and invited plans
against the rest -- measured runs spent attempts on `electric_switch_wseglt_8`
and `picture_zsirgc_0`, both failing NO_SPACE_AROUND_TARGET.

The activity's `object_scope` now rides on the observation as
`task_entity_ids`, and the view drops what the activity never mentions -- from
the room listing, the verbs and the relations alike, so all three agree on what
exists. Teammates and whatever is held are never dropped (a held object is out
of every room, and hiding it would hide the only verb that puts it down), and an
empty scope means "no opinion", so a run without a BDDL task is unchanged.

### Every call started from a blank slate

The agent was told the world and its current plan, never what it had already
tried, so it re-proposed plans that had just failed with the reason nowhere in
front of it. The pieces existed and the chain was broken twice:
`parse_plan_response` discarded the model's `reasoning` at the door, and
`AgentMemory` is a ring buffer shared with message traffic, so a chatty round
evicts the plan events -- a history that silently forgets is worse in a prompt
than none. Outcomes go in a dedicated `agent.plan_history` now, recorded where a
plan actually terminates. An interrupt-and-*resume* is excluded structurally
rather than filtered: it is the same plan still running, and no transition
records anything for it. (A plan interrupted and *replaced* is recorded, as of
2026-09-12 -- see "Memory in the prompt" below.)

### Making the destination visible fixed one task and broke the benchmark

To solve `v4_s1_v4_ll` the agent needed to name a floor in another room, so one
commit made every object the activity declares visible from anywhere. It worked,
and it dissolved the premise the whole benchmark rests on -- an agent sees the
room it is standing in and no other. The cost was not confined to that task:
`coop_nine_apples_hall` declares 21 objects, so all nine apples, all nine chairs
and the table would have been visible to every agent from step 0, and the
exploration problem the activity exists to pose would have disappeared.

**The agent never needed to see the destination. It needed to be able to *name*
it.** Those are different, and only the second is required to write
`ontop(packing_box.n.02_1, floor.n.01_2)`. So the room listing is strictly local
again and the goal is stated in the ids it is written in:

```
childs_room_0:
  - agent_1  (teammate)
  - floor.n.01_1  -> navigate_to
  - packing_box.n.02_1  -> grasp, navigate_to

YOUR TASK, in the ids it is written in:
  ontop(packing_box.n.02_1, floor.n.01_2)   [floor.n.01_2 is in the bedroom]
```

`inroom` is part of the activity definition, so repeating it discloses nothing
the task had not already stated. The one structural exemption that stays is a
task floor in the agent's *own* room, which it is standing on. It also solves
faster than the version that broke the premise -- env_step 377 against 504, five
plans against six. **Telling an agent its task is better information than making
it infer the destination from a room listing.**

## Porting another COOHAVIOR task

`PORTING_COOHAVIOR.md` is the procedure, written from doing v4_s1_v4_ll: which
three files come out, the two BDDL parser traps that fail silently, where each
scene keeps its furniture edits (S1 in a separate layer, S2/S3 inline), and why
`scale` and `base_locked_while_holding` are not optional. The findings behind it
are below.

## Third activity: v4_s1_v4_ll, and robots from outside BEHAVIOR (2026-09-11)

`bddl3/bddl/activity_definitions/v4_s1_v4_ll/` -- three robots move a packing
box from a child's room to a bedroom floor. Solved at **env_step 504 of 2500**,
six plans, by the Ridgeback's suction arm; the four failures are OBJECT_CLAIMED,
which is contention working -- three robots went for one box and one got it.

The task was the easy half. What it cost was everything that has to be true
first.

**The robots are imported from their own upstream URDFs** through BEHAVIOR's
official importer, which is the first exercise of the "robot models are a
registry, not a fixed list" extension point above. Two of the three can grasp,
because the symbolic grasp teleports the object to the end effector and welds a
FixedJoint -- which is what a suction cup is. The finger-contact path it never
uses was the only thing in the way.

Facts worth keeping, each found by loading the robot rather than by reading:

* A ~30 line `ament_index_python` shim makes the import pipeline run without
  ROS. pip's `xacro` resolves `$(find pkg)` through that module, which ships
  only with ROS 2 and is not on PyPI; the shim reads the exact ament index
  layout `register_package` already builds.
* **Three of the four armless-robot failures are configuration**, and are in
  `coop2/robot_configs/v4_ridgeback_ur5_primitives.yaml`: no fingers declared,
  `grasping_mode: sticky`, `disable_grasp_handling: true` -- the documented
  switch for a caller that drives grasps itself, which the symbolic layer does.
* The fourth is a real OmniGibson bug, fixed in `robot.py`: joint index tensors
  are built from a list comprehension, and an empty list yields **float32**,
  which cannot index. An empty list is legitimate for a robot with no fingers.
  Four sibling properties have the same latent issue; only the one that fires is
  touched.
* **The Crazyflie needed five separate things.** `use_holonomic_joints` (the
  non-holonomic branch of `_get_robot_pose_from_2d_pose` forces z to 0.0, and a
  drone needs the altitude DOF), a `holonomic_base + locomotion` controller
  group, a zero-joint arm whose body is the suction face, `default_joint_pos`
  per virtual joint, and the float32 index fix above. It then loaded, was
  correctly placed and sized, and **rendered nothing**: OmniGibson hides the eef
  link of every manipulation robot on principle, because on a normal arm that
  link is a massless frame -- naming the body as the end effector hid the whole
  drone. It has its own `bottom_suction_mount` 1.5 cm below the belly now, which
  is what V4's own wrapper adds and for the same reason.
* **Re-import a wheeled base holonomic.** Navigation here is a teleport, and a
  real four-wheel base does not survive one: the wheels arrive intersecting the
  floor and PhysX ejects the robot. Every wheel joint fixed, base driven through
  six virtual joints, suspension rocker in the same list -- left drivable it is a
  joint with no controller and the load fails outright.
* **A UR5 at all-zeros stands straight out horizontally** and knocks furniture
  over on arrival. Measured with the base at the origin: all-zeros is
  1.356 x 1.0 x 1.000 m; folded back is 1.000 x 1.0 x 1.134, i.e. down to the
  chassis's own footprint, and leaves the end effector behind the base rather
  than in front of it.
* **Layouts carry `scale`**, because size is part of a layout and not a property
  of a model. V4 stages its three robots within half a metre of each other,
  which only fits because it shrinks them to 0.5/0.7/0.6. Loaded full size two
  overlapped, `place_robots` relocated them by 0.8 m and 1.9 m, and the
  relocation shoved the box 1.4 m across the room.

**A robot stands on a floor, not around it.** Navigating to a floor asked the
robot to stand 3.3-3.9 m from its centre -- the target's own half-diagonal,
right for a table and wrong for a surface spanning a room, since the ring lies
outside the room the floor is the floor of. 187 of 200 candidates rejected,
NO_SPACE_AROUND_TARGET every time. This is not a corner case: navigating to a
room's floor is the only way to say "take this to the bedroom", and with the
destination finally nameable the model wrote exactly the right plan and could
not execute it. A surface the robot stands on is sampled *within* rather than
around; the room filter and traversability check are unchanged.

**The scene instance reproduces the one V4 authored**: the box pinned to V4's
coordinate rather than sampled, 84 objects removed (19 pictures, 16 light
switches, 10 doors, the sofa, five armchairs), two moved, the box at 8 g. The
filter list is a **separate USD layer** the staging scenes reference -- easy to
miss by reading only their own edit block, which is how the count first came out
8 instead of 89. Beds, coffee table, shelf and fridge stay: they are BDDL
supports.

**`coop2/experiment/inspect_scene.py`** loads all of it and stops -- no agents,
no LLM, no plan loop -- and prints where each robot was asked to be against where
it ended up. It found three defects in its first three runs, none of which cost
an episode.

## Carrying, and the sequencing problem it exposed (2026-09-12)

`v4_s1_v4_ll` stages a base-locked arm and an armless carrier so that moving a
box needs both. Both constraints are enforced now, on both sides -- the engine
refuses, and the listing does not offer what the engine would refuse, which is
the half that makes the other worth having.

### The two constraints

**A locked base cannot drive while loaded.** `_require_base_free_to_move`
raises BASE_LOCKED (it already did), and `target_hints` now emits no
`navigate_to` at all while such a robot is holding something. The Holding line
says `[your base is locked while loaded: no navigate_to]`, because the
consequence is an *absence* and an absence explains nothing on its own. An
out-of-range target's "unreachable" note says the base is locked too, or it
reads as "navigate_to first" and there is no navigate_to to take.

**A carrier has no arm.** That is what makes it a carrier rather than a second
arm, so `navigate_to` and `wait` are the only verbs it gets. `load_onto` and
`unload_from` are hand verbs too -- they are performed *on* a carrier by
something that has one. `_require_arm` gates grasp, place, open, close, toggle
and both carry verbs with NO_ARM; without it the refusal still happened
somewhere useless (`load_onto` reported "you are not holding anything to load",
which is true and is not the reason).

### Cargo has to be visibly unavailable

A load is a FixedJoint from the carrier's base to the object -- the same thing
a grasp is, on a different link, which is why it survives the carrier driving
off. But no gripper reports it, so every "is anyone holding this" answered no,
and an agent shown a free box welds a *second* joint onto one that already has
one. `held_objects` covers cargo now, `holder_of` follows it through
`carrier_holding`, and OBJECT_CLAIMED names `unload_from` rather than telling
the agent to ask for a release that is not coming.

That change had to be unpicked once: reading cargo as held made the Jackal think
it was holding the box, and offered it `place_on_top` and `release` -- two verbs
it has no arm to perform, for an object it cannot let go of. `_holding` means
"in a hand"; a carrier sees its load on an `On your back:` line instead.

A carrier and its cargo are **one line**, because they are one thing to act on:

    - agent_1  (teammate)  [carrier, carrying packing_box.n.02_1]  -> navigate_to, unload_from

The cargo has no line and no verbs of its own. What the verbs *mean* is in
ENV_DESCRIPTION / TEAM_ENV_DESCRIPTION -- the listing carries state, the system
prompt carries meaning, once rather than on every line of every robot's section.

### What two runs showed

Same task, same seed, same everything but the team split:

| | teams | goal | what happened |
|---|---|---|---|
| `individual_agents3_..._032648` | alpha=(ridgeback, jackal), bravo=(drone) | **env_step 1662** | the full handoff ran |
| `individual_agents3_..._033353` | one team of three | **not solved in 2500** | the carrier drove off before the box was on it |

Both had one brain planning the Ridgeback and the Jackal together, so this is
not about information: the brain saw both robots in one prompt either way. It is
about **timing**. A team's plans all start at once, so the brain has to make the
carrier wait out the grasp, and the only tool it has is a `wait(N)` it must size
by guessing:

    succeeded:  agent_1  wait(400) -> navigate_to(floor_2)     grasp cost 297
    failed:     agent_1  wait(100) -> navigate_to(floor_2)     grasp cost 297
                agent_0  grasp[297] -> load_onto -> "4.34 m from agent_1"

and the failure is expensive, not merely a retry: the Ridgeback is left holding
a box it cannot drive anywhere with, so its only move is to put it back down.

Three ways out, cheapest first, none implemented:

1. Put the primitive costs in the system prompt (grasp/place ~100 ticks floor,
   navigate 100 + 30/m). Smallest change; still arithmetic.
2. Let a carrier **wait for an event** rather than a duration -- a
   `wait_for_load` that ends when something lands on its back. Removes the guess
   rather than informing it, and matches what the role actually is: drive to
   where an arm is waiting.
3. Make `load_onto` approach the carrier itself on failure. Does not help here,
   because the robot that needs it is exactly the one that cannot drive.

`coop2/team_layouts/v4_s1_v4_ll_one_team.json` is the second configuration,
kept because the comparison is the evidence for all of the above.

## Memory in the prompt: the last three plans, and both sides of every message (2026-09-12)

Asked for two things: that an agent planning sees its own last three plans
with their content and, where they failed, why; and that it sees the messages
it has received *and sent*. Checked upstream first (`coop2-llm-mas/ma_crafter`)
so as not to rebuild it. The honest finding is that most of it existed, in
pieces that did not reach the prompt that matters:

| piece | upstream | this port before | gap |
|---|---|---|---|
| `AgentMemory` ring buffer of `message_in/out` + `plan` events, `format_memory` | yes | yes, verbatim | rendered only by `build_observation_prompt`, the per-robot path -- and **every run now plans through `TeamBrain`**, which never read it |
| `agent.plan_history` -> "YOUR FINISHED PLANS" | no | yes, per member in the team prompt | showed the goal, the reasoning and the failure -- **not the actions**; limit 5; a plan interrupted and *replaced* left no entry |
| team `_heard` -> "WHAT THE OTHER TEAMS SAID" | no | yes | received only, and **cleared every round**: the model saw the latest turn and nothing before it |
| team `_say` | no | yes | sent to the broker and recorded **nowhere on the team's side** |

So the team quoted the other teams' words and none of its own, and forgot
both at the next plan. On `centralized_agents8_..._174328` the leader's
planning request appears in the follower's prompt exactly once, in the round
it arrived; the reply it sent is in no prompt at all.

### What changed

**Plans.** `record_plan_outcome` now stores the plan's actions
(`navigate_to(apple.n.01_6) -> grasp(apple.n.01_6) -> ...`) and a `status`:
`done`, `failed`, or `replaced`. `format_plan_history` renders the last
**three**, each with its `plan:` line between the goal and the reasoning:

```
YOUR FINISHED PLANS (most recent last):
  #2 [DONE] ontop(apple.n.01_1, coffee_table.n.01_1)
      plan: navigate_to(apple.n.01_1) -> grasp(apple.n.01_1) -> navigate_to(coffee_table.n.01_1) -> place_on_top(coffee_table.n.01_1)
      you chose it because: Assigned apple 1 exclusively to agent_0.
  #3 [FAILED] ontop(apple.n.01_6, coffee_table.n.01_1)
      plan: navigate_to(apple.n.01_6) -> grasp(apple.n.01_6) -> ...
      it failed because: grasp: PRE_CONDITION_ERROR: apple_137 is currently held by agent_9 ...
  #6 [REPLACED] ontop(apple.n.01_5, coffee_table.n.01_1)
      plan: wait(600)
      it was abandoned because: you replanned after an interrupt, before it finished
```

A plan that was interrupted and **replaced** is an outcome now -- the agent
decided on it and then decided against it -- recorded in
`_sync_committed_plans`, the one place the wrapper knows a plan has been
superseded. It is marked REPLACED, not FAILED: abandoning is not failing.
Resume is still excluded structurally. The recall of a team hold takes the
same path, so holds are recorded too and **hidden at render time**
(`_TEAM_HOLD_PREFIX`): a hold is the barrier's mechanics, not a plan the model
made, and listing it read as "you planned to wait" three times over.

**Messages.** `TeamBrain` keeps an `AgentMemory` -- the same class every robot
has, reused rather than re-invented, holding only messages so a chatty round
cannot evict anything else. `_say` records `message_out`; both routes in
(`_collect_heard`, `_file_heard`) go through one `_remember`, which de-dupes
across rounds as well as within one and records `message_in`. `_heard` keeps
its meaning -- what arrived since the last plan -- because `_was_asked`,
`_heard_from` and the chain's relay logic are queries against it; it now
supplies the `(new)` mark. One block, both directions, oldest first, replaces
"WHAT THE OTHER TEAMS SAID":

```
MESSAGES THIS TEAM SENT AND RECEIVED (oldest first):
  [step 0] You told follow: [team_0] Leader planning request: report each robot's ...
  [step 0] From follow: we will take the west apples, leave the east to you
  [step 1961] From follow (new): apple.n.01_5 is now on the table ...
```

News is quoted whole; older lines are cut at 240 chars -- the record is
context, the news is what the model must act on. The interrupt prompt gets
the record as "EARLIER MESSAGES" *minus* the messages it is deciding on, which
are quoted in full under "MESSAGES" right after it; before this an
interrupting message that had also been filed appeared twice. Bounded at
`TEAM_MESSAGE_HISTORY = 8` quoted of `TEAM_MESSAGE_MEMORY = 32` kept.

`TEAM_ROLE` says what the two blocks are for in one sentence each, so the
model is told to read them rather than left to notice them.

Tests: `test_plan_history_stubbed.py` (7-10: actions, REPLACED, hidden holds,
the team prompt), `test_recalled_hold_stubbed.py` (a replaced plan reaches the
agent's record; a recalled hold reaches it and not the prompt), and new
`test_team_message_memory_stubbed.py` (eight checks on the message record,
driven through a real broker and the centralized pair).

**Measured on GPU the same day** (`individual_agents9_..._202823`, section
below): across the three second-round team calls, every robot whose history
showed a `[FAILED]` entry -- five of five -- chose a different target, and the
allocation text cited the record ("Apples 1 and 9 are already held by other
teams"). The record also produced the run's most expensive decision: see
"the model read three identical failures" there. Both are the mechanism
working; what it is fed is the next problem.

## Nine robots, three teams, and the apple nobody could deliver (2026-09-12)

`individual`, 9 R1s as three teams of three, seed 0, 8000 steps, gpt-5.6-luna,
first run with the prompt memory above. Goal at **env_step 7244**. Run folder
`individual_agents9_repair_off_seed0_20260912_202823_973907`.

| | 9 robots (3x3) | 12 robots (3x4), earlier today |
|---|---|---|
| apples | **9/9 at 7244** | 9/9 at 4434 |
| Y_plan | 0.833 | 0.588 |
| plans | 111 (60 ok / 12 failed / 39 cut), 30 of them holds | 64 (20 / 14 / 30), 24 holds |
| LLM calls | 27 | 10 |
| tokens | 207 956 | 88 330 |
| idle | 22.0 % (14 331 of 65 196 robot-steps) | 32.6 % |

Failures on work plans: `OBJECT_CLAIMED` 7, `NO_SPACE_AROUND_TARGET` 3,
`TOO_FAR` 1, one empty-hand `PRE_CONDITION`. Apples on the table, read from
the `Relations:` section of each prompt: 2 at 1970, 3 at 2119, 4 at 2508, 6 at
3121, **8 at 3722** -- then the ninth took 3500 steps. Every step of that is
in the run folder, and it is three separate things.

### The table holds six intents, and the seventh fails (Open defect 7)

agent_5 picked up apple 5 at ~3000 and its `navigate_to(coffee_table)` failed
three times over the next 1300 steps (plans #3, #4, #6), every time with the
same attribution:

    rejected_by: {'room': 0, 'trav': 2, 'robots': 198}    target_xy: [-11.97, -18.8]

198 of 200 candidate poses rejected *by other robots*. `room: 0` and a fixed
`target_xy` rule out the drifting-table event of 2026-09-09. Five robots --
agent_1, 3, 6, 7, 8 -- listed the table as **in range**, all finished, all
holding `wait(600)` where they had placed; the first reading of this was
"five parked bodies fill the ring".

**That reading was too generous to bodies.** The chain run the same evening
(`broadcast_chain_agents9_..._210451`) hit the identical failure at steps
**1343, 1622 and 1796 -- before any robot had delivered** (first placement at
1871), `rejected_by robots: 199/200`. The six `[nav]` lines before it are
robots that had *sampled* a pose at the table and were still 37-42 m away.
What blocks a candidate is in `_clear_of_other_robots`: bodies **and**
`DestinationRegistry` reservations, at the same 1.24 m separation -- and the
reservation is made in `_sample_pose_near_object`, on the primitive's first
`next()`, *before* the contention layer's ~1100 travel ticks. So a robot's
spot at the table is taken the moment it decides to go, for the whole trip.
A coffee table's 1.24-1.84 m annulus holds about **six** such intents; the
seventh, eighth and ninth samplers fail, one per team in the chain run,
whoever they are and however well the teams coordinated. That is a capacity of
the receptacle, and no allocation of nine robots to one table gets round it.

### The model read three identical failures and did the reasonable wrong thing

With `[FAILED] ... no free floor space around it` three times in its history
block, agent_5's team wrote: "the coffee table has repeatedly been unreachable
because its surrounding floor is congested -- stage the held apple on the
nearby chair so agent_3 can retrieve it". It put apple 5 back on chair 5, 37 m
from the table. Given what it could see this was sound; what it could not see
is that the congestion was its own teammates and the other teams' finished
robots, and that they would still be there when agent_3 arrived. The prompt
carries object distances per robot and no robot-to-robot distances.

Then `_ensure_task_terminal_action` appended `place_on_top(coffee_table)` to
the staging plan, because the specification was still `ontop(apple_5, table)`
and the actions did not end in a placement. Empty hand, 37 m away, `TOO_FAR`,
51 ticks. The guard is doing what it was written to do; "put this down
somewhere else on purpose" is not a shape it knows. (Open defect 8.)

### One apple left a hand it was welded to (open, not diagnosed)

agent_3 fetched apple 5 from the chair (grasp OK at 5658), teleport-navigated
35.6 m to the table (navigate OK at 6830, 1067 travel + settle), and the next
precondition said it was holding nothing. At 6881 the world read
`ontop(apple.n.01_5, floor.n.01_1)`, and agent_3's re-fetch `navigate_to`
cost 127 ticks -- under a metre -- so the apple fell **at the table**, during
the navigate's own post-teleport settle. No PhysX warning in the log.

Once, in roughly twenty welded long teleports across today's two runs. The
12-robot run's one empty-hand placement (agent_10, #4 at 1961) is a different
thing: the robot had *already* delivered in plan #2 and the model wrote a lone
`place_on_top` anyway -- stale belief, not a broken weld. Upstream's symbolic
`_navigate_to_pose` is `robot.set_position_orientation(...)` then
`_settle_robot()`; the held object is not moved, it rides the AG FixedJoint
and PhysX closes a 35 m constraint violation in the following steps. That it
usually works is the measured fact; why it did not once is not. Neither run
logs robot positions per step, so whether agent_3 arrived into a parked robot
cannot be read off this run. The discriminating measurement is a
`feasibility_verify` script that teleports a robot with a welded apple N times
to an annulus with 0 and with 5 parked robots and counts detachments.

## The chain, seen through its prompts (2026-09-12, evening)

`broadcast_chain`, 9 R1s as three teams of three, **3000 steps** -- a budget
chosen to read prompts, not to solve. Run folder
`broadcast_chain_agents9_repair_off_seed0_20260912_210451_851009`.

| | value |
|---|---|
| apples at 3000 | 6/9 (2, 3, 4, 5, 8, 9), sixth at 2555 |
| plans | 27 (8 ok / 3 failed / 7 cut on work plans; 9 holds) |
| failures | `NO_SPACE_AROUND_TARGET` 3 -- the only code that fired |
| LLM calls / tokens | 9 / 71 537 |
| messages | 5, cascade ordered: team_1's relay 3.8 s after team_0's round-2 broadcast |
| idle | 18.7 % |
| Y_plan | 0.727 |

**What the memory blocks looked like, from the run's own `llm_calls.jsonl`:**

* Round one: team_0 has no message block (nothing is upstream of it); team_1
  quotes team_0's allocation once, `(new)`; team_2 quotes team_0 then team_1,
  oldest first, both `(new)`.
* Round two (team_0, env_step 2346): `[step 0] You told team_1, team_2: ...`
  -- the team's own words survived the round that clears `_heard`.
* Interrupts (2346, 2477): the record under `EARLIER MESSAGES`, the arriving
  broadcast under `MESSAGES:`. Compared by exact content, **0 duplicates** in
  all three interrupt prompts. (A first check by 60-char prefix reported one;
  team_0's two allocations share their first 75 characters and differ only at
  `apple.n.01_03` vs `apple.n.01_1`. Compare whole contents.) team_2's 2346
  prompt carried team_0's new allocation *and* team_1's relay together, which
  is the cascade working. Every decision was `resume`, so no `[REPLACED]`
  entry could appear -- a correct absence, not a gap.

**What the communication bought.** Round one partitioned the nine apples
perfectly -- 1-3 / 4-6 / 7-9 -- with no cross-team duplicate, where the
`individual` run at the same moment had two pairs of teams each sending a robot
to the same apple. Zero `OBJECT_CLAIMED` all run.

**What it could not buy.** All nine robots headed for one table at once, and
the table takes six (defect 7): the third robot of every team failed
`NO_SPACE` at 1343, 1622 and 1796, before anyone had delivered. Two of them
then read that failure in their history and **released** the apple where they
stood -- agent_0 with apple 1 at 2346, agent_5 with apple 6 at 2477, "so the
table-side agent can take over its long delivery" -- 37-42 m from the table.
The same reasonable wrong thing as the staging in the individual run, from the
same cause: the model is told the ring is full and not that the crowd is six
reservations in transit that will have turned over by the time it arrives.

## A goal is a tree, and reading it as a row killed both apple tasks (2026-09-12)

`_goal_terms` -- "Tell the agent its task", `c1a46d978` -- read every goal
clause as a predicate with atom arguments:

```python
predicate, args = clause[0], clause[1:]
where = "; ".join(f"{a} is in the {rooms[a]}" for a in args if a in rooms)
```

A goal is not that. It is a tree of quantifiers and connectives over
predicates, and both apple activities are quantified:

    coop_nine_apples_hall  [["forall", ["?apple.n.01", "-", "apple.n.01"],
                             ["ontop", "?apple.n.01", "?coffee_table.n.01_1"]]]
    coop_two_apples_pomaria  the same clause, verbatim
    v4_s1_v4_ll            [["ontop", "packing_box.n.02_1", "floor.n.01_2"]]

So `a` was a list, `a in rooms` raised `TypeError: unhashable type: 'list'`,
and **the first `env.reset()` died** -- on both apple tasks, for a day. Three
things kept it hidden:

* `v4_s1_v4_ll` has the only flat goal in the repo and was the task in hand
  when the code was written. Every run between the two dates was that task.
* **Isaac swallows the exception and the process still exits 0.** A crashed run
  is indistinguishable from a finished one by exit code; the traceback is in
  the log, prefixed `[py stderr]` by `omni.kit.app`, and nothing else says so.
* No CPU test could reach it: the renderer lived on a method of a class that
  imports OmniGibson.

Quantification exposed a second thing that the flat goal never could: **BDDL
prefixes every term inside a goal with `?`**, a bound variable and a concrete
instance alike, so the goal names the table `?coffee_table.n.01_1`. Nothing
here resolves that id, and the model already mangles instance suffixes.

Rendering is `symbolic_view.render_goal_terms` now -- L1b, where the prompt
text is made and where a CPU test can read it without Isaac. `coop_env`'s job
is only to find the two parsed condition lists on the `BehaviorTask`.

**A quantifier is stated, not expanded.** `forall ?apple.n.01` over nine
declared apples would name all nine in every prompt from step 0, which is the
same harm as showing the agent every room, and it would delete the exploration
`coop_nine_apples_hall` exists to pose. What the agent gets is what the goal
says:

    YOUR TASK, in the ids it is written in:
      ontop(every apple.n.01, coffee_table.n.01_1)

`feasibility_verify/test_goal_terms_stubbed.py` asserts the crash shape, the
`?` stripping, that no `apple.n.01_N` leaks, and that `forpairs`/`forn` -- which
no activity of ours uses -- render rather than crash.

## coop_nine_apples_hall, solved (2026-09-12)

`individual`, 12 R1s as three teams of four, seed 0, 8000-step budget,
`--time-limit-seconds 0`, gpt-5.6-luna. Goal at **env_step 4434**;
`check_goal` reports `{'satisfied': [0], 'unsatisfied': []}`. Run folder
`individual_agents12_repair_off_seed0_20260912_163009_025018`.

| | this run | `broadcast_chain` 12 (3x4), 8000 steps |
|---|---|---|
| apples | **9/9 at 4434** | 6/9, full budget |
| Y_plan | 0.588 | — |
| plans | 64 (20 ok / 14 failed / 30 cut), 24 of them team holds | 183 (66 / 30 / 87) |
| LLM calls | **10** | 55 |
| tokens | **88 330** | 706 467 |
| idle | 32.6 % (17 367 of 53 208 robot-steps) | 23.7 % |

Failures: `OBJECT_CLAIMED` 9, `TOO_FAR` 4, one other. Actions: grasp 9 ok /
13 failed, place_on_top 8 ok / 1 failed / 14 in flight at the end, navigate_to
28 ok. **Zero `NO_SPACE_AROUND_TARGET`** -- the flying-receptacle signature did
not appear. Zero LLM errors, 5.8 s average latency. 10 calls x 4 robots = 40
work plans, and 64 - 24 holds = 40.

**This is not a topology comparison and must not be read as one.** Four things
landed between the chain run and this one, all of which change what the model
is shown: the listing scoped to the activity's objects, plan history in the
prompt, the task stated in its own ids, and the goal-rendering fix above. The
8x token drop is most plausibly the scoping, not the topology. Comparing
topologies needs all three re-run against the current prompt.

### The prompt states intent in the same syntax as fact

Counting delivered apples by grepping a prompt for
`ontop(apple.n.01_N, coffee_table.n.01_1)` gives the wrong answer, and gave one
here mid-run. A team prompt lists each of its four robots' plan
*specifications*, in exactly that form, four times over -- plus the plan
history's `#4 [FAILED] ontop(...)` lines. Only the `Relations:` section states
facts. Read against it, the run went 2 apples at env_step 1961, 5 at 2536, 6 at
2699, 7 at 3137, 8 at 3467, and stayed at 8 until the last one landed at 4434 --
967 steps for the ninth.

## The route file: what BDDL cannot say, said beside it (2026-09-12)

Step 1 of ROUTE_SUPERVISION_PLAN.md. `coop2/behavior_env/route_spec.py` reads
`activity_definitions/<activity>/route.json`, found the way BDDL finds
`problem0.bddl`, and validates it against the parsed BDDL -- pure Python over
`bddl.parsing`, so `test_route_spec_stubbed.py` loads the real file. An
activity without one is unrouted, which is every activity but LL today.

The rules are the ones that fail late without them, each a CPU test: a node's
goal is a BDDL triple and nothing else; every id it names is in `:objects`;
the last node equals the BDDL `:goal` for that cargo (COOHAVIOR's own LL files
disagree on this -- their BDDL says a bedroom floor, the route they run says a
bed -- and the loader refuses rather than picks); order is list order, so a
`sequence` or `depends_on` field is refused; consecutive nodes with one goal
warn, because in Merom_1_int every room has one floor and the second would
score for free.

**Three of COOHAVIOR's support names were wrong for us**, found only by
checking each against the taxonomy *and* the scene: `bottom_cabinet.n.01` is
not a synset (the category is `cabinet.n.01`), `fridge.n.01` is
`electric_refrigerator.n.01`, and `shelf_owvfik_0` -- a node of all four S1
tasks -- is our `bookcase_owvfik_0`: the same model at the same pose under an
older asset name. The plan's own prose said every other synset was valid. It
was not, and a route file that named `shelf.n.01_1` would have passed the
loader and failed at the sampler.

**Growing a BDDL past its cached instance makes the task silently
unwinnable.** LL's BDDL went from 3 objects to 13 and its goal from the
bedroom floor to `bed.n.01_1`. `behavior_task.py:538` *skips* an instance the
template never bound, the scope entry stays None, and
`evaluate_bddl_predicate` returns False for a None entity -- so the old cached
instance loads without a word and `check_goal` can never fire. The sampler
whitelist now pins the nine supports by model (`inroom` separates same-model
instances in different rooms; the two `jrhgeu` cabinets share bedroom_0 and
either is acceptable), and the instance is re-sampled as part of the step.

Re-sampled 2026-09-12: all ten nodes bound, goal false at t=0, template
written. Two bindings are not COOHAVIOR's -- C4 took `armchair_qplklw_1` (V4
moves `_2` and deactivates the rest; `inroom dining_room` matches all three,
and the one the sampler picked was then kept off the deactivate list because
the BDDL bound it, so the scene has one armchair more than V4's) and C8 took
`bottom_cabinet_jrhgeu_1` (V4's files disagree on `_0`/`_1`). Neither changes
the route. The sampler prints every node's binding now, which is the only
place an unbindable support is visible before an episode is spent on it.

## Route supervision, wired (2026-09-12)

Steps 2 and 3 of ROUTE_SUPERVISION_PLAN.md. For an activity with a route file,
`terminated` is now **every route complete**, judged on the world's state at
each macro-step boundary; `check_goal` is kept as a cross-check and a
`[route] BDDL check_goal disagrees` line is printed if the BDDL final state is
not true at the moment the route completes -- one of the two files is wrong,
and that is said rather than settled silently. HL stops ending at env_step 0
for the same reason: the trivial BDDL goal no longer decides anything.

`RouteTracker` asks `relation_holds` for the expected node and, once each, for
every later node: the expected one holding is `completed` (and the loop
re-checks, so two nodes on one support do not stall); a later one holding is
`out_of_order` -- recorded, credited to nobody, nothing advances, and progress
resumes the instant the expected node holds (user, 2026-09-12: does not count,
is not punished). Scored nodes are never re-evaluated, because under state
detection a box still sitting where it scored is not a second visit. Ten
predicates a step for any V4 task; no fact-set build. Credit for a completion
goes to the one robot that held the cargo at the previous step, does not now,
and whose primitive terminated now; anything less certain is `None`, because
`unattributed` is a real value and a guess is not.

What the agent is shown is the whole route, marked (user's choice over
next-node-only), each support's room read from the BDDL `inroom` the file was
validated against, so nothing is disclosed the activity did not state:

```
YOUR TASK, in the ids it is written in:
  Route S1-M5, carry die.n.01_1 through these in order:
  NEXT  C1  ontop(die.n.01_1, cabinet.n.01_1)   [childs_room]
        C2  ontop(die.n.01_1, bookcase.n.01_1)   [kitchen]
        ...
        D   ontop(die.n.01_1, bed.n.01_1)   [childs_room]
A support reached out of order does not count, and nothing is lost by it: progress resumes when the NEXT one holds.
```

That is `inspect_scene --view ridgeback_1` on the re-sampled LL instance, verbatim.
Both system prompts carry one rule saying what a route is and that progress is
judged on where the cargo rests; `test_team_prompt`'s shared-rule check holds
both to it.

Post-episode: `route_progress.json` (every event, then the summary) beside
`plan_logs.json`; `compute_metrics` reads it and adds COOHAVIOR's numbers under
COOHAVIOR's names -- `Y_task` = completed nodes / required nodes (nodes, not
legs: user's choice), `out_of_order_visits`, `route_complete`, and `S_team`
credited by team through `team_timeline.json` with `unattributed` kept apart.
An unrouted run has no file and every existing metric is unchanged. All of it
runs on a fabricated run dir in `test_route_tracker_stubbed.py` (14 checks),
because post-episode code that only runs after a GPU episode is the defect
class that has cost a run per bug three times here.

One of those checks was earned the hard way in this very step:
`test_behavior_action_stubbed` builds the env with `object.__new__`, so an
attribute added to `__init__` does not exist there, and the first version of
`_goal_reached` raised on it. Every read of `route_tracker` is
`getattr(self, "route_tracker", None)` -- which is also what a subclass or a
stub would need.

**Not yet done at the time:** an LL episode against the route (step 4; budget
it at `--steps 8000`, ten nodes at roughly navigate + grasp + navigate + place
each), the lift gate (step 5, done later the same day -- "The lift gate"), HL's
five two-node routes (step 6, done), and the route files for the other nine
tasks (step 7, S1 done).

## Four route files, four re-samples (2026-09-12)

LH, HH and HL got what LL got: a `route.json` transcribed from
`tasks.modified.json`, a BDDL grown to declare every support the route names
with its `inroom`, goals corrected to the routes' destinations (the route is
the task -- user's decision -- so the BDDL follows it), the sampler pinning the
nine supports by model, and a re-sample. All four now load with every node
bound and the goal false at t=0; `test_route_spec_stubbed` checks all four
files' shapes on CPU.

| task | routes | cargo | may lift | what was wrong before |
|---|---|---|---|---|
| LL | 1 x 10 | die | arm, drone | goal was the stale bedroom floor |
| LH | 1 x 10 | notebook | **arm only** | goal was C1's cabinet; eight supports missing |
| HL | 5 x 2 | die | arm, drone | goals were the spawn floors, true at t=0 |
| HH | 5 x 2 | notebook | **arm only** | destinations right; three checkpoint cabinets missing |

HL's and HH's five routes chain -- M5's destination is M3's checkpoint, M3's
is M2's -- but each box is on its own route and the tracker judges them
independently. `unsatisfied: [0, 1, 2, 3, 4]` at load is the line that says the
env_step-0 problem is gone.

**Boxes start on floors, and COOHAVIOR's files had to be read three ways to
know it.** Its `spawn_nodes` annotate each HL/HH box's spawn support as a
fixture (fridge, table, bed) and the spawn *markers* sit on those fixtures; its
`initial_relation` says the floor; its staged box *prims* -- the coordinates
our sampler pins to -- are on the floor beside the fixtures. The prim is what
physically spawns, so the floor is what is true, and no node is a floor, so
nothing can be credited at load.

**A `*` anywhere in a definition is a wildcard, comments included.** The first
HL/HH re-sample died inside `Environment.__init__` with an `IndexError` from
`knowledge_base/models.py:_strip_wildcards`, which runs over the raw text and
treats every line containing `*` as a scope line to be split on ` - `. The
lines were my comments -- `the spawn *marker* sits -- on a fixture`. Two GPU
runs to learn that emphasis is syntax; the loader test now runs that pass over
every S1 definition in a millisecond. Recorded as trap 3 in
`PORTING_COOHAVIOR.md` step 1.

Bindings that are not COOHAVIOR's, recorded rather than hidden: LL and HL took
`bottom_cabinet_jrhgeu_1` for C8/M4-C1 where LH and HH took `_0` (both stand in
bedroom_0; COOHAVIOR's own files disagree), and LL took `armchair_qplklw_1`
where the other three took `_2`. `_apply_scene_edits` keeps whatever the BDDL
bound off V4's deactivate list, so LL's scene has one armchair more than V4's;
none of it changes a route.

## Robots are named by type and team ordinal (2026-09-13)

`agent_0 .. agent_14` said nothing about what a robot was, and the model had
to learn "agent_1 has no arm" from a bracket. Every S1 layout now uses one
scheme (user, 2026-09-13, unified after a first pass that kept V4's m-numbers
for the five-set files): teams are `team_1 .. team_k` in order of appearance,
and each robot is `<type>_<team ordinal>` -- `ridgeback_1, jackal_1, drone_1,
ridgeback_2, ...`. Which V4 set a team came from (M5, M3, ...) survives only in
each robot's `_v4_prim`. LL/LH are `team_1` (ridgeback_1, jackal_1) and
`team_2` (drone_2); the one-team variant has `drone_1`. Names are free strings
everywhere -- the layout is the only source, `env_setup` makes them the robot
names, the world model keys entities on them -- so nothing in code changed,
except the engine, which had been pairing ids with robots by position (see
"Step 4's first episode"). `make_s1_shared_spawn_layouts.py` emits the same
scheme. The `agent_N` fallback still applies to layouts without names and to
`--agents N`. Commands below use the new names (`--view drone_2`).

## The lift gate (2026-09-12)

COOHAVIOR's one constraint we had no word for: the Crazyflie may carry the
8 g box and "is not a load-bearing role for the 20 g box". The two weights
were already two objects (die, notebook); what was missing was the rule, and
the route file's `lift` table is where it lives (`"notebook.n.01": ["arm"]`,
`"die.n.01": ["arm", "drone"]`). Three decisions:

* **Role is a layout fact.** `RobotSpec.drone`, parallel to `carrier`, set
  as `"drone": true` on every `v4_crazyflie_cf2x` entry in the ten layouts.
  Not inferred from the model string -- that would make a name a capability.
  `carrier.lift_role(robot)` is the one place the role is computed: carrier
  first (no hand, so the question never arises), then drone, else arm.
* **Both sides, once each.** `coop_env._build` hands `route_spec.lift` to the
  world (`lift_rules`, copied onto each `SymbolicObservation`) and to every
  controller. The engine's `_require_may_lift` refuses `grasp` and
  `unload_from` -- unloading puts cargo in the hand, so it is a lift -- with
  reason code `CANNOT_LIFT`, which terminates the plan like `TOO_FAR`. The
  view's `_may_lift` mirrors it exactly and withholds the verb.
* **The rule is in the system prompt, not on the line.** The first version
  wrote `blocked, navigate_to [too heavy for you]` on the notebook's line;
  the user turned that down (2026-09-13): just say in the system prompt that
  the drone cannot lift the notebook. So both prompts now carry that one
  sentence -- die yes, notebook no, `CANNOT_LIFT`, only an arm can -- and the
  listing gives the robot the fact the sentence is keyed on: `(drone)` in
  its own header, `[drone]` on a teammate's line. Out of reach the notebook
  reads like any far object, `unreachable, navigate_to [1.2 m away]`.
* **No table, no opinion.** A synset the table does not name, or a run with
  no route file, is unchanged: anyone with a hand may lift anything. The
  apple tasks are untouched.

CPU: `test_carrier_view_stubbed` 8b (drone gets `grasp(die)` only, no mark,
header says `(drone)`; arm gets both; a carrier gets no grasp regardless) and
8c (unload withheld the same way); `test_team_config_stubbed` parses and
refuses a non-bool `drone`. GPU: `inspect_scene --view drone_2` on LH prints
`[route] lift rules: notebook.n.01 -> arm`, the header reads
`you are drone_2 (drone) in childs_room_0`, and the notebook's line carries
no `grasp` while ridgeback_1's reads `-> grasp, navigate_to`.

## Step 4's first episode: two nodes, five defects, room navigation (2026-09-13)

The first routed LL episode: nine robots as three teams (`s1/sets_3.json`,
`ridgeback_k / jackal_k / drone_k`; the team labels were still V4's m5/m3/m2 in
that run, since renamed `team_k`), individual topology, 8000 steps, video
on. **C1 credited at env_step 2310, C2 at 3607, then nothing** -- `Y_task`
0.2, `route_progress.json` written, `check_goal` never fired, 19 team calls
and ~140k tokens by step 4000. The run is
`coop2/runs/individual_agents3_repair_off_seed0_20260913_011125_636624`.
Two launches died before it, and the one that ran exposed three more things.
In the order they were hit:

1. **Tick 0 crash: `Action must be dimension 4, got dim 9`.** The engine
   zipped layout-ordered agent ids with `scene.robots`, which is *sorted by
   name*. `agent_0..8` happened to agree; `ridgeback_1, jackal_1, drone_1`
   did not, and ridgeback_1's 9-dim action went to the robot called
   drone_1. Mapped by name now (`MultiAgentPrimitiveEngine.__init__`), CPU
   test added. The same zip paired `agent_2` with robot `agent_10` in every
   twelve-robot run; their per-agent success counts look normal and I cannot
   explain why. Recorded, not resolved.
2. **A thousand steps circling a coffee table.** Everyone spawned in the
   living room; the die was in the child's room; the route block named ten
   supports and not where the cargo was. `Route.start_support/start_room`
   now come from the BDDL `:init`, the block opens with
   `it starts ontop(die.n.01_1, floor.n.01_1) [childs_room]` until C1 is
   credited, and both prompts say a task id is a `navigate_to` target from
   any room. The relaunch's first plans went straight to the die.
3. **Cargo on furniture is unreachable.** After C2 the die sat on the
   kitchen bookcase. `navigate_to(die)` wanted a spot 0.3-0.9 m from the die
   -- inside the bookcase or in the wall -- and failed NO_SPACE; `navigate_to
   (bookcase)` landed 1.1-2.1 m from the die and `grasp` was TOO_FAR at the
   die's own 0.9 m. Nine robots alternated the two for 2000 steps. Now
   `support_of(obj)` (geometric: the highest non-floor object whose footprint
   holds the object's xy at its bottom) makes `navigate_to(obj)` sample
   around the support and `interaction_radius_for(obj)` reach across it
   (support's gate + its half-diagonal, measured to the object). Same
   contract as before: anything the sampler produces is in range.
4. **Drones took floor space.** `_clear_of_other_robots` was 2-D; a drone
   parked at 1.2 m over the bookcase cost the arm every standing spot in a
   2.7 m2 kitchen. Bodies more than `LAYER_SEPARATION_Z` (0.6 m) apart in z
   are different layers now.
5. **Drone videos were one grey frame.** `chase_pose` put the eye 2.3 m
   above the robot; for a drone at 1.2 m that is inside the roof void. The
   eye is capped under `MAX_CAMERA_HEIGHT` and the aim point under the eye.

**Room navigation** (user, 2026-09-13): the listing is local, so an agent
could not name a room it had never been in and so could not go looking. The
header now carries `Rooms in this house (navigate_to any of them): ...` from
the seg map, `navigate_to(<room instance>)` is dispatched by the engine
(`is_room`) to `navigate_to_room`, which samples a free traversable spot
inside the room with the same robot-clearance and reservation filters, and
both prompts say so. CPU: engine, navigation, world-state and prompt tests.

What the run also showed and nothing was changed for: three teams on one
die hand it back and forth across team lines (drone_3 unloaded jackal_2's
cargo), each hop a plan; a carrier planned `place_on_top` and was refused;
"holding for the team" idles dominate the timeline. Those are the
cooperation problem the benchmark is about, not defects.

## The acceptance run, and what a die on a Jackal's back taught (2026-09-13)

**Second routed LL episode: five of ten nodes.** Same nine robots, same
budget, all of the previous section's fixes: C1 at 1037, C2 at 3379, C3 at
5539 (credited `by drone_2`, the first attribution), C4 at 6819, C5 at 7936;
`Y_task` 0.5, 32 team calls, 241k tokens, no crash, nine videos including
real drone views. Run
`coop2/runs/individual_agents3_repair_off_seed0_20260913_014222_865648`.
Its first launch died on every `navigate_to(<room>)` because upstream's
`get_random_point_by_room_instance` calls `th.randint(n)` without a size;
`navigate_to_room` samples cells itself now. Nineteen `navigate_to(<robot>)`
across the two runs were refused as "'jackal_2' is a robot" while the listing
offered exactly that on every carrier; the engine now dispatches a robot
target to `navigate_to_robot`, which bypasses `apply_ref`'s own refusal.

**The user watched the first run's video and saw a drone drop its die.** The
answer took ten GPU probes (`feasibility_verify/verify_carrier_cargo_gpu.py`,
no LLM: drone grasps the die, flies to the Jackal, loads it, the Jackal
drives away; every few ticks the die's pose against the Jackal's body). What
they found, in order:

* The die at floor height after a drone's grasp was the probe's own artifact:
  the LL layouts give the drone no z and it spawns on the floor. With
  `s1/sets_1` (1.2 m) the grasp holds the die 3 cm under the mount and it
  flies with the drone. **The grasp is fine.**
* `load_onto` puts the die at the Jackal's top centre (z 0.417 over a 0.394
  top, dxy 0.000) and it rides through a 6 m drive. **The handoff is fine.**
* The drone's reported angular velocity is 0.0 before the grasp and
  **4.445 rad/s for ever after**, on the root and on the body link, while its
  yaw, its position and all six virtual base joints do not change by a
  millimetre over 250 idle ticks, with or without gravity on the die, before
  and after a hop. The number is a ghost of the fixed-joint solver, not a
  motion. It is also why every drone grasp cost 501 ticks: the settle waited
  for a velocity that never falls below 0.01.
* `support_of(drone)` answered "dice_154": the die welded under it lay in
  its footprint at its bottom. A robot has no support, and a support is at
  least as large as what rests on it.

Three changes came out of it, all in `symbolic_contention.py` /
`symbolic_navigation.py`:

* **Settles are 10 ticks at most** (user). `NavigableSymbolicActionPrimitives.
  _settle_robot` replaces upstream's 50 unconditional + up to 500, and
  `tune_primitive_macros` sets `MAX_STEPS_FOR_SETTLING` to 10. Grasp went
  from 501 ticks to 11; `load_onto` from 51 to 1.
* **A teleporting robot takes its attachments.** A FixedJoint drags its child
  after a teleported parent only through the solver, over ticks; under the
  10-tick cap the Jackal's die was measured 0.76 m behind it when the
  primitive returned. `_teleport_with_attachments` moves what is in the hand
  and what rides on the back by the robot's own rigid transform, then stills
  them. After the fix: 1.6 cm.
* **A drone holds under its mount.** Upstream places a grasped object's centre
  at the eef link origin -- inside the suction mount's collision shape. The
  die now hangs half a mount plus half a die below it. Harmless where the
  spin turned out to be a ghost, correct regardless.

The die beside the Jackal in the video (7.0-7.5 s, ticks 840-900) sits inside
`load_onto`'s window; the scripted reproduction never puts it there, and no
log records object poses. Recorded as unexplained rather than fixed. What the
next run's videos show, with the ten-tick settles, is the test.

## The videos were lying: one camera, nine files (2026-09-13)

The user noticed that `episode_drone_1.mp4` and `episode_drone_2.mp4` of the
second routed run showed the same drone, and that it stopped moving after
about 20 s. Both true, for two different reasons.

**The same drone.** `MultiViewRecorder` had one viewer camera and moved it to
nine poses per capture, rendering and reading after each move. A probe that
set four far-apart poses and compared what came back against converged
references showed the camera's new pose reaching the renderer only
*sometimes* before the frame was read: the first slot of a pass came back as
the previous pass's last view, some later slots as their predecessor, and
neither two nor three renders per view nor rendering until two consecutive
reads agreed made it reliable -- a frame can be stable and still be the wrong
pose. So `episode_drone_2` held drone_1's view, and at 30 s, when drone_2 was
in the kitchen placing the die on the bookcase, its file showed the child's
room. Every file was one view behind its name.

The fix removes the moving part. When `video_path` is set, `coop_env`
declares one `VisionSensor` per robot in the OmniGibson env config
(`external_sensors`, `coop2_view_<name>`, 1280x720, the wide lens, parked
off-scene) and `MultiViewRecorder` takes them as `cameras`: each capture
poses all nine, renders once, and each writer reads its own sensor. A camera
that never changes owner has nothing to mix up; a pose that lands one render
late shows the same robot four ticks earlier. Nine captures of a nine-robot
scene came out as nine correct, centred views (drone_k and jackal_k
coincide at spawn, as they should -- the drone hovers over its Jackal), at
5.6 s per 60 ticks. The shared-camera path remains as the fallback when the
sensors are missing.

**It stopped moving.** That was drone_1, in both files. From env_step 2064
to 8000 every plan its team gave it was `wait`: "Drone_1 remains idle
because the die is already claimed", "drone_1 waits because it cannot lift
the die" (it can; the team's brain misread the lift rule for the die), and
so on for six rounds. The robot did what it was told. The misreading is a
prompt problem to watch, not an engine one.

## Open defects

Fixed ones are not listed here -- the fix and its reasoning live in the commit
and in the code comment at the site. What is still true:

1. **A rejected precondition costs 50 ticks, not 0.** ``apply_ref`` runs its
   "settle before returning" block after catching the error and before raising
   the group, so ``TOO_FAR`` and friends are not free. Visible in the plan log
   as ``ticks=50`` on a failed action.
2. **cuRobo, if it is ever reintroduced, rejects poses on the other robot but
   not poses on a small object.** Measured 2026-09-05: 8/8 candidate poses
   collision-free from 0.05 m to 0.45 m from a 5 cm apple, because the base
   genuinely clears it -- a target-clearance floor is needed on top of any
   collision check. One generator per R1 costs ~2.2 GB and 4-8 s, and
   ``batch_size=16`` OOMs a 16 GB card in ``mg.warmup()``; the default is 2.
   cuRobo is currently **not used at all** (user decision): pose validity comes
   from the trav map plus geometry.
3. **The chain's planning path still orders on readiness.** `_await_speakers`
   releases as soon as the team ahead is `ready`, and a team ahead that is
   *executing* is ready -- so a team can reach its own barrier and plan with
   nobody having spoken to it. The interrupt path was fixed (2026-09-11
   section); this one cannot be, the same way: blocking there freezes the world
   the team ahead needs to finish, so the waiting would have to happen in a
   hold (X) rather than in R.
4. **A failed LLM call is invisible in the metrics.** `_record_llm_error` does
   not record latency, so `llm_usage.json`'s `total_api_latency_seconds` omits
   it -- a 552 s call that returned nothing showed up only as
   `total_llm_errors: 1`. The 100 s timeout caps the damage; it does not make
   it visible.
5. **Two robots cannot be sequenced except by guessing tick counts.** The only
   way a team brain can make robot B act *after* robot A is to put a `wait(N)`
   in front of B's plan and hope N exceeds what A's action costs. It is not told
   what an action costs -- the system prompt states the travel charge and
   nothing else -- so it is arithmetic on a number it does not have. Measured on
   the two `v4_s1_v4_ll` runs of 2026-09-12: the handoff succeeded when the
   brain happened to pick `wait(400)` against a 297-tick grasp, and failed when
   it picked `wait(100)`. See "Carrying, and the sequencing problem it exposed".
6. **`coop2_trace.json` is always empty.** `plan_env_wrapper` logs
   `plan_committed` events into the repair shim's `Coop2TraceLogger`, whose
   `log()` discards them. Harmless while repair is unported -- nothing reads
   the file -- but the events are gone, and `plan_logs.json` does not carry the
   remaining-action list they had.
7. **A receptacle's annulus holds about six intents, bodies or reservations
   alike.** `DestinationRegistry` reserves a standing pose when the navigate
   is *sampled* -- ~1100 ticks before the body arrives -- and the sampler
   rejects candidates within 1.24 m of any reservation or body. A coffee
   table's 1.24-1.84 m ring therefore fills at six robots heading for it,
   and every later sampler gets `NO_SPACE_AROUND_TARGET` with
   `rejected_by robots: 198-199`. Measured twice on 2026-09-12: once per team
   at 1343/1622/1796 in `broadcast_chain_agents9_..._210451` before any
   delivery, and three times on the last carrier in
   `individual_agents9_..._202823`. It terminates the plan (by design, see
   TERMINATES_PLAN), so the model reads it as "the table is congested" and
   twice released or staged the apple tens of metres away. Options: a
   reservation could expire or be released on arrival so the ring turns over;
   NO_SPACE on a *moving* crowd could be a wait-and-retry rather than a plan
   failure; or the prompt could say the crowd is in transit. None exists.
8. **`_ensure_task_terminal_action` cannot tell staging from forgetting.** A
   plan whose actions end in `place_on_top(<somewhere else>)` gets a
   `place_on_top(<the goal's reference>)` appended, because the specification
   still names the goal. Deliberately putting an object down for a teammate
   is therefore always followed by an empty-hand `TOO_FAR` (51 ticks). Same
   run, plan #7 of agent_5.
9. **A team broadcasts its holds as if they were plans.** `_plan_summary` and
   `_current_allocation` print every member's specification, so team_2 was
   told `agent_3: wait_for_team(team_1); agent_5: wait_for_team(team_1)` in
   the chain run -- barrier mechanics that mean nothing to another team, the
   same leak `format_plan_history` now filters on the history side. The
   model's own id spelling travels raw too: team_0 wrote `apple.n.01_03`,
   `resolve_target` fixed it locally and agent_1 delivered apple 3, but the
   other two teams read `_03` and one wrote "likely 3".

See also "Open, not yet diagnosed" near the end of this file for behaviour that
is understood but not yet explained.

## Commands

Use the `behavior` conda env (see the repo root `AGENTS.md`), and
`OMNIGIBSON_HEADLESS=1` when there is no display.

```bash
# CPU-only regression checks -- no Isaac, no GPU, a few seconds each. RUN THESE
# FIRST after touching anything in coop2/. They are main() scripts, not pytest
# cases: `pytest` collects nothing from them.
for f in feasibility_verify/test_*.py; do python "$f"; done

# The real thing: an LLM-driven episode against the BDDL activity. Use `python
# -u` whenever stdout is redirected to a file: Isaac's shutdown ends the process
# without flushing, so the whole tail after the last simulator print is lost --
# including "[goal] BDDL goal satisfied at env_step N" and the plan statistics.
# A run that looks like it stopped early and said nothing is usually this.
python -u -m coop2.experiment.run_individual --agents 2 --steps 4000 --seed 0 \
  --scene Pomaria_1_int --room living_room_0 \
  --bddl-activity coop_two_apples_pomaria \
  --goal "Put both apples on coffee_table.n.01_1." \
  --model gpt-5.6-luna --llm-quiet
# run_centralized / run_broadcast_chain take the same flags.
# Output lands in coop2/runs/<topology>_agents<N>_..._<timestamp>/.

# Twelve robots as three teams of four on the nine-apple hall -- the 2026-09-11
# sweep, about 17 minutes a topology. --time-limit-seconds 0 is required or the
# 120 s default decides the episode length instead of --steps; --no-video is
# most of the difference between 17 minutes and an hour.
OMNIGIBSON_HEADLESS=1 python -u -m coop2.experiment.run_broadcast_chain \
  --agents 12 --team-size 4 --steps 8000 --seed 0 --time-limit-seconds 0 \
  --scene hall_glass_ceiling --room empty_room_0 \
  --bddl-activity coop_nine_apples_hall \
  --goal "Put all nine apples on coffee_table.n.01_1." \
  --model gpt-5.6-luna --llm-quiet --no-video
# Do NOT wrap this in `conda run`: it buffers stdout until the process exits, so
# an hour-long run is unmonitorable. Call the env's python directly.
# --agents 9 --team-size 3 is the cheaper shape: same 6/9 apples, 30 % fewer
# tokens, 22 % idle against 37 %.

# A layout of imported robots -- V4's Ridgeback+UR5, Jackal and Crazyflie. The
# layout beats --agents, and carries each robot's model, position and scale.
# v4_s1_v4_ll has a route.json, so the run prints a [route] line per node
# credited, ends when the route completes (not when check_goal fires), and
# writes route_progress.json. Ten nodes: budget --steps 8000, not 2500.
python -u -m coop2.experiment.run_individual \
  --team-config coop2/team_layouts/v4_s1_v4_ll.json \
  --scene Merom_1_int --room childs_room_0 --bddl-activity v4_s1_v4_ll \
  --steps 2500 --seed 0 --time-limit-seconds 0 \
  --model gpt-5.6-luna --llm-quiet --no-video

# Load a scene and stop: no agents, no LLM, no plan loop. Prints where each
# robot was asked to be against where it ended up, and --shot saves a picture.
# Reach for this before spending an episode on a new scene or a new robot.
python -u -m coop2.experiment.inspect_scene \
  --scene Merom_1_int --room childs_room_0 --bddl-activity v4_s1_v4_ll \
  --team-config coop2/team_layouts/v4_s1_v4_ll.json --shot /tmp/scene.png

# Delete run folders (dry run by default; --yes to actually delete).
python coop2/runs/clean_runs.py

# Watch it in a window. --gui is the ONLY way: --show is a no-op (the
# visualisation wrapper is a stub by design) and OMNIGIBSON_HEADLESS is not
# enough on its own, because CooperativeBehaviorEnv defaults headless=True and
# _build assigns gm.HEADLESS itself. Needs a DISPLAY. Verified: an
# "OmniGibson 3.9.2" X window at 1468x966, with carb.windowing.plugins and
# omni.kit.mainwindow started -- neither loads headless.
python -m coop2.experiment.run_individual --gui --agents 2 --steps 60 --seed 0 \
  --scene Pomaria_1_int --room living_room_0 \
  --bddl-activity coop_two_apples_pomaria --model gpt-5.6-luna --llm-quiet
# --gui turns recording OFF, and that is not a limitation to work around: a
# scene has ONE viewer camera, MultiViewRecorder captures its N views by moving
# that camera to each robot and back several times a second, and with a window
# open that camera *is* the window -- so recording and a live viewport cannot
# coexist without a second camera. In GUI mode the viewport is instead aimed at
# the task once, at build time, and then never touched: anything that re-aims
# during the episode is the flicker. Orbit it yourself.
# Headless runs still record: per-robot episode_agent_<i>.mp4 land in the run
# folder, because RENDER_VIEWER_CAMERA and HEADLESS are separate switches.

# Inspect the scene by hand once the episode is over. --keep-viewer implies
# --gui, and runs *after* every log, plot and metric file has been written, so
# Ctrl+C out of it costs nothing. It keeps stepping the sim and prints each
# BDDL task object's position and speed every 2 s -- which is how to watch for
# the drifting coffee table without trying to catch it by eye.
python -m coop2.experiment.run_individual --keep-viewer --agents 2 --steps 4000 \
  --seed 3 --time-limit-seconds 0 --scene Pomaria_1_int --room living_room_0 \
  --bddl-activity coop_two_apples_pomaria --model gpt-5.6-luna --llm-quiet

# Same stack with no credentials: substitutes StubLLMClient for the model, so a
# later failure with a real one is unambiguously the model and not the plumbing.
python feasibility_verify/verify_runner_offline.py

# BDDL: sample an activity instance (~45 s, exits non-zero without saving if the
# layout is unstable), then check it, then check that check_goal ends an episode.
python feasibility_verify/sample_coop_task_instance.py
python -u feasibility_verify/verify_coop_task_instance.py
python -u feasibility_verify/verify_bddl_terminates_episode.py

# Diagnostics. Reach for these before forming a hypothesis.
python -u feasibility_verify/preview_cameras.py --out /tmp/cams
python -u feasibility_verify/measure_target_capacity.py
COOP2_ENGINE_VERBOSE=1 python -m coop2.experiment.run_individual ...   # per-primitive ticks

# Older GPU demos, pre-BDDL: contention and concurrency in isolation.
python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan navigate_to
python feasibility_verify/multiagent_concurrent_primitives.py --mode exclusive
```

## The engine (L1c) in one paragraph

`MultiAgentPrimitiveEngine` is a **stepper, not a driver**: the caller owns the
main loop. `assign(agent_id, primitive, target)` starts a primitive without
advancing anything; `tick()` performs exactly one `env.step`, feeding each
active agent the next value from its `apply_ref` generator and every other
robot a hold-position action; `has_active(agent_id)` says whether an agent's
previous primitive is still in flight. `tick()` returns only the primitives
that terminated on that tick, so the normal return value is `{}`.

This shape exists so the barrier can sit at the **plan** boundary, matching
COOP²'s `PlanningEnvWrapper.step`:

```python
while not done:
    if not all_ready:                                  # plan-level barrier
        yield idle_step_return(...); continue          # tick() NOT called: physics frozen
    for agent_id in executing:
        if not engine.has_active(agent_id):
            engine.assign(agent_id, *next_primitive_of(agent_id))
    for agent_id, outcome in engine.tick().items():
        ...plan.advance_action() / complete_failed() / set_unready('plan_terminated')
```

`macro_step()` exists for scripted demos only. It aligns agents at every
primitive boundary, which COOP² does **not** do — never build the runner on it.

## Hard constraints (each of these fails silently or crashes)

- `scene.include_robots: false`, **and that is not sufficient with a BDDL
  activity.** `Environment._load_robots` is guarded by
  `if len(self.scene.robots) == 0`, so your `robots:` list is ignored whenever
  the scene has already imported robots -- which it has, because
  `BehaviorTask.verify_scene_and_task_config` points `scene_instance` at the
  cached template and the template contains the robots it was sampled with.
  The robots then come up with R1's defaults (IK arms, delta trunk) while
  `robot._controller_config` still reports yours, and nothing raises until
  `q_to_action` asserts on the first tick. `coop_env._enforce_controller_config`
  compares the live `ControllerView` registry against the config we built and
  calls `reload_controllers` when they disagree.
- `task.use_presampled_robot_pose: false`. The class default is **True** despite
  its docstring saying False, and a template sampled without presampled poses
  has no `robot_poses` metadata, so `BehaviorTask.reset` dereferences None.
  This facade places robots itself anyway.
- Every robot needs an explicit `name` — it is the action/obs dict key.
- Use `model: r1`, not the deprecated `type: R1`. Prefer **R1 over R1Pro**:
  cuRobo drops the DEFAULT embodiment at cuda capability (12,0) (RTX-50) while
  `update_obstacles` indexes it unconditionally → `KeyError`.
- `enable_head_tracking=False` always. `_overwrite_head_action` asserts
  `robot.model == "tiago"` and `_grasp` sets `_tracking_object`.
- `apply_ref(attempts=1)`. The default 5× retry is **not idempotent** and burns
  thousands of ticks per attempt.
- Call `tune_primitive_macros()` **before** constructing any controller: reading
  a macro locks it against writes. `coop_env._build` does this now -- it did not
  for most of the port, which left `MAX_STEPS_FOR_SETTLING` at upstream's 500
  and made a single PLACE_ON_TOP cost over 1000 ticks (`_release` and
  `_settle_robot` each burn the budget in full).
- Construct controllers only after the robots are at their reset pose —
  `_arm_targets` / `_reset_eef_pose` are frozen in `__init__`.
- Idle action is `robot.q_to_action(robot.get_joint_positions())`, **not**
  `controller._empty_action()` (which servos the arm to the frozen targets).
- `og.sim` is a process singleton. One env per process; parallel runs must
  fan out via subprocess.

## Primitive sets: what actually works

- **Physical** (`StarterSemanticActionPrimitives`): only GRASP, PLACE_ON_TOP,
  PLACE_INSIDE, NAVIGATE_TO, RELEASE. OPEN / CLOSE / TOGGLE_ON / TOGGLE_OFF
  `raise NotImplementedError`. One primitive costs 10³–10⁴ ticks.
- **Symbolic** (`SymbolicSemanticActionPrimitives`): OPEN/CLOSE/TOGGLE do work,
  but `NAVIGATE_TO` is **broken as shipped** — its inherited sampler
  dereferences the cuRobo motion generator the symbolic constructor never
  builds, and then passes a keyword the symbolic `_navigate_to_pose` rejects.
  Use `coop2.behavior_env.symbolic_navigation.NavigableSymbolicActionPrimitives`
  instead.
- Symbolic `_grasp` teleports the object to the end-effector **at any
  distance** and `_navigate_to_pose` is a pure teleport, so symbolic mode has
  essentially no resource contention as shipped. Use
  `symbolic_contention.ContentiousSymbolicActionPrimitives` (subclass of
  `NavigableSymbolicActionPrimitives`) to put it back — see below.

## Pose-filter comparison (measured 2026-09-05, N=9 planning)

`feasibility_verify/measure_pose_filters.py`, 120 candidate poses around apple_0,
one cuRobo generator as ground truth. R1 base radius measured **0.62 m**;
trav_map erosion radius **0.82 m**.

| filter | VRAM | scales to N=9 | holes vs cuRobo |
|---|---|---|---|
| per-robot cuRobo | 2.2 GB *each* -> 21.5 GB for 9 | **no** (card is 15.4 GB) | 0 by definition |
| shared cuRobo (`update_obstacles(ignore_objects=...)`) | ~2.2 GB total | yes | unverified: obstacles are expressed in the generator's own robot root frame |
| trav_map AND geometry | 0 | yes | **3 / 120** |

- `cuRobo accepts but geometry rejects = 76` is **not** a regression: it is almost
  entirely d <= 0.8 m, i.e. exactly the "standing on the apple" poses we want to
  reject and cuRobo does not.
- `cuRobo accepts but trav_map rejects = 84`: the baked `floor_trav_0.png` covers
  **all** of Rs_int's furniture, while the env loads only
  `["floors", "walls", "coffee_table"]`. The map is therefore more conservative
  than the actual scene. Loading full furniture would shrink this.
- The residual 3 holes survive raising robot separation from 0.8 to 1.3 m
  (`geometry accepts but cuRobo rejects` fell 15 -> 11, union stayed 3), so they
  are trav_map holes, not robot overlap. Cause not isolated.
- trav_map rejects 0/12 within 0.8 m of apple_0 -- that spot is genuinely cramped
  (it is also the `room=None` point). ~25% of candidates pass at d >= 1.0 m, so
  200 sampling attempts still succeed; the robot just stands further back.

## Scene choice: Rs_int is unusable (measured 2026-09-05)

`feasibility_verify/measure_teleport_risk.py` and `survey_scene_capacity.py`
read the baked `floor_trav_0.png` maps directly (0.01 m/px) -- CPU only, no
Isaac. R1's circumscribed radius is **0.62 m**
(`norm(reset_joint_pos_aabb_extent[:2]) / 2`, arms included).

| scene | bad-pose rate | dead targets | 9 robots fit? |
|---|---|---|---|
| **Rs_int** | **96.2%** | **46.3%** | no -- largest connected free region is **0.9 m2** |
| house_single_floor | 29.5% | 1.7% | yes (1709 m2) |
| Beechwood_0_int | 81.9% | 19.7% | marginal |
| Merom_1_int | 90.5% | 35.3% | no |
| office_large | 65.2% | 18.7% | marginal |

* **bad-pose rate** = sampled base poses landing where an R1 does not fit. A
  validity filter turns these into retries, so they are survivable.
* **dead targets** = targets with *no* valid pose anywhere in the 0-1.5 m
  annulus. A filter cannot help: NAVIGATE_TO just raises PLANNING_ERROR.

So **the scene must change before the filter matters**. Rs_int stays broken at
any radius (86.5% / 13.3% even at an unrealistically small 0.42 m). This is why
robots were visibly teleporting into walls and toppling in the demo videos.

Prefer a multi-room house over the big halls (`hall_arch_wood` has 4560 m2 but
is one undivided space): L1b's room-level world graph and COOP2's spatial
constraint both need real room separation. `house_single_floor` is the
candidate. 38 / 51 scenes fit 9 robots at 1.24 m separation.

## N=3 end-to-end, verified on GPU 2026-09-05

`house_single_floor`, `--n-robots 3 --plan navigate_to_then_grasp --contend`:
one winner, two losers, each with a legible reason.

```
agent_1  NAVIGATE_TO  153  ->  GRASP  success
agent_0  NAVIGATE_TO  272  ->  GRASP  OBJECT_CLAIMED (held by agent_1)
agent_2  NAVIGATE_TO  379  ->  GRASP  OBJECT_CLAIMED (held by agent_1)
overlap 0.75   env.step 431   wall clock 21 s
```

Two fixes got it there, and their effect was much larger than expected:

| | before | after |
|---|---|---|
| entities in the prompt | 218 | 47 |
| legal (primitive, target) pairs | 132 | 49 |
| env.step ticks | 2830 | 431 |
| wall clock | 214 s | 21 s |
| overlap ratio | 0.37 | 0.75 |

1. **Prompt filtering** (`symbolic_view.STRUCTURAL_SYNSETS` /
   `RECEPTACLE_ABILITIES`, since renamed from the original category lists). One corridor produced 78 walls, 24 shelves, 20
   switches, 16 downlights and 14 paintings, plus nonsense hints like
   `place_on_top(downlight#22)` and `place_inside(door#3)`. Filtering lives in
   L1b, **not** L1a: the world model stays complete for task evaluation and only
   the prompt is pruned.
2. **Objects clustered near the team** (`place_objects(near_robots=8.0)`).
   `place_robots` already clustered the robots, but objects were still sampled
   from the whole room -- a 20 m corridor put the contested apple 11 m away.

**Corrected 2026-09-08.** This section originally concluded: "the
`_settle_robot` blow-up was a symptom, not a separate defect ... no macro tuning
was needed", on the grounds that primitives went from 1100-1728 ticks to 118-379
once objects were clustered near the team. That conclusion is why
`tune_primitive_macros()` was written and then **never called**, and the defect
survived until 2026-09-08: with `MAX_STEPS_FOR_SETTLING` at upstream's 500, a
single PLACE_ON_TOP was measured at 1063 ticks and still rising
(`COOP2_ENGINE_VERBOSE=1` shows the count climbing linearly with env_step),
because `_release` and `_settle_robot` each burn the full budget and an R1's
holonomic base does not reach `velocity < 0.01`. Clustering objects helped, but
it did not remove the need for the macro -- both were required.

## M5 acceptance, GPU-verified 2026-09-06

One agent through the plan channel, no LLM
(`feasibility_verify/verify_plan_channel.py`):

```
navigate_to  success  551 ticks
grasp        success  100 ticks
navigate_to  success  523 ticks
place_on_top success  150 ticks
final held_objects: {}      decision_count=4   env_step=1328
apple relations: OnTop(apple#1, bookcase#2)
```

⚠️ **The first run of this printed PLAN COMPLETE while the apple was still in
the gripper.** `action_outcome` is true for exactly the tick its primitive
terminated on, but `step()` cached the whole `info` dict between refreshes and
served the stale outcome with it -- so every action reported its predecessor's
success the instant it was issued, and three of the four primitives never ran
(`decision_count` was 2, not 4). Fixed by always overwriting `action_outcome`
when reusing cached info. The status column could not catch this; only the
independent facts could -- `held_objects` and the scene graph's `OnTop`. Keep
verifying against physical state, not against the status field.

## Observation scope and refresh (decided 2026-09-06)

* `observation_for()` shows the agent's **current room only**;
  `include_seen_rooms=True` is opt-in. An observation that accumulates every
  room ever visited grows without bound over an episode and stops describing
  where the agent is, and the current room is the scope COOP2's spatial
  constraint is defined on anyway.
* The world model is rebuilt **only when a primitive terminates** — i.e. when an
  agent returns to the reasoning stage and actually has a reason to look. There
  is no timer refresh (`observation_every` defaults to 0): a primitive spans
  10^2–10^3 ticks, so a periodic rebuild would recompute the scene graph
  hundreds of times inside one primitive for nobody to read.

## Concurrent destination race (fixed 2026-09-06)

Separation was checked against other robots' **current** positions, which under
concurrency is a time-of-check/time-of-use bug. All N agents get NAVIGATE_TO on
the same tick; each samples its destination on its generator's first `next()`,
while every other robot still stands at its start pose metres away. Every check
passes, then all of them teleport beside the same object.

Measured on a 3-agent contend run: agent_0 and agent_1 ended **0.64 m** apart
against a 1.24 m requirement, and agent_1's assisted grasp latched onto
**agent_0** — `holding=agent_0`. The same `holding=<robot>` corruption the
separation filter was supposed to have removed.

Fix: `symbolic_navigation.DestinationRegistry`, **one per scene**, shared by
every controller. An agent reserves the pose it is about to occupy; every other
sampler avoids reservations as well as bodies. Reservations are overwritten,
never released — "this agent intends to be here" holds until it decides
otherwise, and once it arrives the reservation and its body coincide, so an
aborted primitive self-corrects instead of leaking a blocked spot.

After: closest pair 1.93 m, no `holding=<robot>`, and `OBJECT_CLAIMED` is back
as the contention signal instead of physics-induced `POST_CONDITION`.

The regression test is statistical on purpose: without the registry ~100/200
trials overlap, with it 0/200. A single-draw version of that assertion is flaky
(three random poses around one object are sometimes well separated) and would
eventually get deleted rather than fixed.

## L3 verified end-to-end, 2026-09-06

`feasibility_verify/verify_l3_plan_loop.py` drives the real
`PlanningEnvWrapper` (its ready barrier, plan lifecycle and logging) with a
scripted agent in place of the LLM:

```
Plan #1 navigate_to -> grasp -> release   all OK, SUCCEEDED at step 693
Plan #2 grasp(ghost#99)                   failed -> "terminating plan" -> reasoning
Plan #3 navigate_to                       OK -> complete -> reasoning
decision_count 4, env_step 888, barrier closed for exactly 3 ticks
```

Confirms the intended model: an arbitrary-length plan runs to completion
without the driver advancing it, only completion or failure returns the agent
to reasoning, and physics is frozen while it reasons.

⚠️ **Two executor sets is the trap here.** The facade builds
`CooperativeBehaviorEnv.executors` and calls `execute()` on them, while L3 reads
plan progress from `get_action_records()` on the *wrapper's*
`agent_actions`. When those were separate objects the wrapper's history stayed
empty, `action_status` came back None, and L3 re-issued action 1 forever. From
outside it is indistinguishable from a slow primitive -- it burned a 30-minute
timeout before being caught. `BehaviorSymbolicEnvWrapper._adopt_facade_executors`
now shares one executor per agent.

The verify script has a wall-clock cap and a stall detector (>12 primitives
issued without `current_action_index` moving) precisely because a timeout
cannot tell "slow" from "not progressing".

## Reuse audit (2026-09-06)

Systematic pass for hand-rolled code that OmniGibson or BDDL already provides.
Replaced:

| was | now |
|---|---|
| private `robot._ag_obj_in_hand` (3 sites) | `robot.is_grasping(arm, candidate_obj)` |
| `RECEPTACLE_CATEGORIES`, ~25 category names | `obj.abilities` (`fillable` / `openable`) from the taxonomy |
| `STRUCTURAL_CATEGORIES` name list | synset ancestry via `ObjectTaxonomy.is_descendant` |
| `imageio.get_writer` | `eval.utils.obs_utils.create_video_writer` / `write_video` |
| ids `apple#1` | BDDL instance naming `apple.n.01_1` |
| scene-graph edge labels | BDDL tokens via `bddl_utils.PREDICATE_TO_STATE` |
| hand-rolled erosion / connectivity / free-space sampling | `scene.get_random_point(floor, reference_point, robot)` |
| objects dropped on random floor cells | `obj.states[OnTop].set_value(surface, True)` |

A rendered fact is now `ontop(apple.n.01_1, breakfast_table.n.01_1)` -- the same
strings an activity definition and its goal predicates use, so M9's
`check_goal` needs no translation layer.

The state->token mapping matters more than it looks: `Hot` is
`object_states.Heated` and `Attached` is `AttachedTo`, so a name-equality check
would work for most predicates and fail silently on exactly those.

**BDDL cannot build the room world graph.** It is purely symbolic: its
predicate classes are empty (`class OnTop(BinaryPredicate): pass`), truth comes
from a callback into OmniGibson, `InRoom` is not even in `PREDICATE_TO_STATE`
so it cannot be evaluated at runtime, and `knowledge_base` is an offline
catalogue of what a scene's rooms contain *by design*, not what is in them now.
The live graph has to come from `SceneGraphBuilder` + `seg_map`, which is what
`world_state.py` does -- that part is not duplicated work.

Genuinely no upstream equivalent, and kept: mutual separation between N robots,
destination reservation under concurrency, room-scoped free space, the
concurrent primitive engine, and the symbolic contention layer.

Found while doing this: `is_fixed` was **always False**. `scene.fixed_objects`
is a `{name: obj}` dict, so `set(...)` of it is a set of names and `obj in` it
never matches. Nothing depended on it until `is_receptacle` did.

## Metrics: two different counters

`engine.env_step` counts ticks (what `Timeout(max_steps)` counts).
`engine.decision_count` counts primitives issued — **this is the denominator
for COOP²'s metrics**. One primitive is 10³–10⁴ ticks, so per-tick rates are
meaningless.

## Symbolic contention (L1)

`ContentiousSymbolicActionPrimitives` restores resource competition to the
distance-blind, holder-blind symbolic set with three coupled rules:

1. **Interaction radius** — GRASP / PLACE / OPEN / TOGGLE require the base
   within `interaction_radius` m of the target, else `TOO_FAR`.
2. **Travel cost** — `_navigate_to_pose` yields hold-position ticks
   proportional to distance **before** teleporting. Padding after the teleport
   would be wrong: the robot would arrive instantly and then idle, so a
   teammate reading the world during those ticks sees it already there.
3. **Claims** — acting on an object another robot holds raises
   `OBJECT_CLAIMED`. Upstream `_grasp` checks only `self.robot._ag_obj_in_hand`
   and `_establish_grasp` puts its joint under the *grasping* robot's eef, so
   without this the object is yanked out of the holder's hand, carries **two**
   FixedJoints, and both robots' post-conditions pass. Silent corruption.

⚠️ `interaction_radius` must be ≥ `distance_range[1]` (the nav sampler's upper
bound), or a successful navigate still sometimes lands out of range and the
agent loops navigate → TOO_FAR forever. The constructor rejects that outright;
the default derives the radius from `distance_range`.

Both new codes are raised as `PRE_CONDITION_ERROR` with
`metadata["reason_code"]` set; `ReasonCode.from_primitive_error` prefers that
over the five-member enum.

**Both are in `TERMINATES_PLAN`** (decided 2026-09-06). They are individually
recoverable — a teammate may release the object, walking closer fixes the
distance — but the plan that produced them was written against a world that has
since contradicted it, so its next action is a stale intention. Terminating
returns the agent to the **reasoning stage**, which is the only place it can
negotiate for the contested object or retarget. Grinding the plan on instead
would turn contention into silent wasted motion rather than a decision the
topology layer is measured on.

## M9: BDDL reconnected — first custom task (2026-09-07)

A real BDDL activity now exists and is cached as a task instance, and `coop_env`
already loads it (`bddl_activity=...` -> `BehaviorTask` with
`online_object_sampling=False`) with `compiled_task.check_goal` deciding `terminated`.
This section is the authoring record: how the activity was written and how to
regenerate the instance.

**The task**: `bddl3/bddl/activity_definitions/coop_two_apples_pomaria/problem0.bddl`
— `Pomaria_1_int` / `living_room_0`, one apple on each of the two armchairs, goal is
`(forall (?apple.n.01 - apple.n.01) (ontop ?apple.n.01 ?coffee_table.n.01_1))`.

```bash
# produce the instance (~45 s, writes the template json). NOT in git -- re-run after a machine change.
OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/sample_coop_task_instance.py
# acceptance test: loads the cached template, forces the goal, checks check_goal flips
OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/verify_coop_task_instance.py
```

Template lands at
`$OMNIGIBSON_DATASET/2026-challenge-task-instances/scenes/Pomaria_1_int/json/Pomaria_1_int_task_coop_two_apples_pomaria_0_0_template.json`.
`BehaviorTask` finds it on its own — with `online_object_sampling: false` and no
`scene_file`/`scene_instance`, `verify_scene_and_task_config` rebuilds that exact
filename from `{scene}_task_{activity}_{def_id}_{inst_id}_template`.

### Authoring facts (all verified, not read off docstrings)

- **A new activity directory is auto-discovered.** `get_all_activities()` is `os.listdir`
  on `activity_definitions/`; `activity_manifest.txt` and
  `activity_to_preselected_scenes.json` are read by no code at all.
  `kb.add_task(name, definition=<string>)` registers one at runtime with no files, but then
  `room_requirements` stays empty and `task.matching_scene(scene)` falsely returns "pass" —
  only the file route gets a real pre-flight check. Use `matching_scene` before burning a
  GPU run; it names the missing furniture per room instance.
- **`sampling_whitelist` is `{synset: {category: {model: None-or-bbox}}}`** — a dict, not
  the "list of valid models" the `BehaviorTask` docstring claims (`bddl_utils.py:894` calls
  `.keys()` on it). It is the only way to pin one of several same-synset scene objects:
  `living_room_0` has two coffee tables and BDDL only knows the synset.
- **Declare exactly one agent.** Omitting it entirely crashes `BDDLSampler.__init__` at
  `bddl_utils.py:486` with `KeyError: 'agent.n.01_1'`, because `update_activity` puts that
  key in `object_scope` unconditionally while `_object_instance_to_synset` comes from
  `parsed_objects`. Declaring a *second* agent also crashes (the sampler binds only
  `agent.n.01_1`, leaving `None` for the rest, and `_filter_object_scope` then dereferences
  `None.prim_type`); it works only with a patch to `bddl_utils.py:553`, which was written,
  verified, and then **reverted on purpose** — OmniGibson is unmodified. Robots past
  `robots[0]` are simply invisible to BDDL, which costs nothing while no goal mentions an
  agent. The *cached* path never needed the patch: `behavior_task.py:548` already maps
  `agent.n.01_N -> env.robots[N-1]`.
- **`inroom` cannot be evaluated** (`PREDICATE_TO_STATE` has no `InRoom`), so
  `compiled_task.check_initial_conditions()` raises `KeyError` on any task using it. Check
  the kinematic facts directly instead. Same trap for `broken` and `grasped`; `grasped` is
  one dict line away from working (`object_states.IsGrasping` already exists).
- **Sampling is unseeded and apples roll.** Different apple models get drawn per instance
  and one rolled off the armchair during the 300-step settle. Both apples are pinned to
  model `omzprq` via the whitelist, and the script exits non-zero rather than save a layout
  whose own initial conditions are already violated. Always cache a template; never sample
  per run. `--instance_id N` produces alternative layouts.

### The template is the scene, so what it omits is missing from every run

`save_task(task_relevant_only=False)` writes whatever was **loaded**, and the
template *is* the scene file each episode loads. So any load filter used while
sampling is baked into every later run.

The sampling script used to pass `load_room_types: ["living_room"]`, and that
produced a house with exactly one floor. Upstream exempts building structure from
the room filter, but the exemption is
`STRUCTURE_CATEGORIES - GROUND_CATEGORIES` and `floors` is a GROUND category
(`interactive_traversable_scene.py`) -- so walls and ceilings bypassed the filter
and floors did not. Every episode since has run in a house whose other six rooms
had no ground: 50 objects, 22 walls, 7 ceilings, **1 floor**. Nothing failed, which
is why it survived; it was visible the moment anyone opened `--gui`.

Filter removed and the template re-sampled: 124 objects and **7 floors**, one per
room (bathroom_0, corridor_0, kitchen_0, living_room_0, pantry_room_0,
storage_room_0, utility_room_0). Scene load goes 38 s -> 78 s. A run still solves
the activity (`individual`, seed 0, goal at env_step 2154).

**The layout changed with it.** Sampling is unseeded, so the re-draw moved the
apples (`apple_48`/`apple_49` are now `apple_122`/`apple_123`); the armchairs and
`coffee_table_gpkbiw_0` are the same instances. The 1303/1303/1278 three-topology
numbers above were measured on the one-floor template and are **not** comparable
to anything sampled after it -- treat them as a pre-change baseline and re-run the
sweep for the real table.

The script now also asserts that `armchair.n.01_{1,2}` and `coffee_table.n.01_1`
bind inside `ROOM_INSTANCE`. `inroom ... living_room` matches a room *type*, and
with the whole scene loaded the sampler could otherwise bind the furniture to a
different living_room instance than the one `place_robots` uses, leaving the team
in an empty room.

### What the template does and does not carry

- Robot **world poses are in there**, but as `joint_pos` of the holonomic base joints, not
  `root_link.pos` — the root stays at the spawn/park anchor (`agent_0` reads
  `[300, 300, 300]`). Real pose = anchor + first three joint values.
- Robot **controller configs are NOT in there.** `init_info.args` holds only
  `name / model / obs_modalities / default_reset_mode / scale`; the saved
  `controller_groups` is goal *state*, and its `arm_left` goal is `target_pos` +
  `target_ori_mat`, i.e. the R1 **default task-space controller**, not the
  `JointController` stack `r1_primitives.yaml` requires.
- ⇒ **Load it with `include_robots: False`** and supply coop2's own robot list. The template
  then contributes only the object layout (apples, armchairs, coffee table), which is all we
  want from it; `build_multi_robot_config` already sets that flag.
- `env.reset()` restores the initial file, so poses set after loading need either
  re-applying each reset or a `scene.update_initial_file()` (`prepare_robots` does this).

## The seven blockers, in the order they were found (2026-09-08)

All fixed, and all seven had to go before any topology could finish the activity.
The seventh -- the sleeping apple, section above -- is the one that made the other
six look insufficient, because it hid success even when the agents played well.

| # | symptom | cause | fixed |
|---|---|---|---|
| 1 | every plan died at grounding, `decisions` stayed 0 | the runners inherited crafter's `CooperativeEnv(...)` call and placed no objects | `77d175296` |
| 2 | ~3.4 s per env_step | two all-pairs scans per macro-step: `SceneGraphBuilder.step()` (14.9 s) and `CoopTaskTracker._fact_set()` (9.4 s) | `77d175296` |
| 3 | agent placed apples on the wrong coffee table, `check_goal` never fired | `entity_id_for` numbered instances in scene order, independently of `task.object_scope`; `living_room_0` has two coffee tables | `5fda3d283` |
| 4 | a single `PLACE_ON_TOP` cost >1000 ticks, so 2500 steps bought ~2 primitives per agent | `tune_primitive_macros()` existed, was measured (1100-1728 -> 118-379 ticks) and **was never called**; `MAX_STEPS_FOR_SETTLING` stayed at upstream's 500, and `_release` + `_settle_robot` each burn it in full | `efe4c0159` |
| 5 | `wait` was an instant no-op, so an agent yielding the floor **stopped the world** (the plan loop does not step the env while any agent reasons) | `wait` was in `COMMUNICATION_ACTIONS` | `5fda3d283` |
| 6 | 51 x `NO_SPACE_AROUND_TARGET`, all `{room: 200, trav: 0, robots: 0}`, target logged at 12 m -> 27 m -> 36 m -> 38 m from the room | upstream's `_place_with_predicate` does release -> `set_position_orientation` -> settle, and **`set_position_orientation` does not zero velocity**: the object arrives carrying the fall it accumulated while being released, and the settle integrates it out of the house | `efe4c0159` |
| 7 | `place_on_top` failed "it did not come to rest there" on a third to a half of placements, and `check_goal` never fired even after an apple was correctly delivered | `OnTop` is `Touching`, `Touching` is a contact-report query, and a slept PhysX actor reports no contacts; the default sleep threshold is 5e-05 and `keep_still()` zeroes the velocity on the way in | `e205c0e0b` |

Method note, because it cost most of the day: for #6 I proposed three geometric
explanations (the annulus round the table is full; the target is being carried by a
teammate; `DestinationRegistry` reservations accumulate) and **measured all three to
be wrong** -- `feasibility_verify/measure_target_capacity.py` shows 28-48 % of
candidate poses accepted in every reproducible state, so 200 consecutive rejections
were impossible. What settled it was making the failure report its own attribution
(`rejected_by` per filter, plus `target_xy`) rather than reproducing states by hand.
Same shape as #2, where four rounds of guessing lost to one cProfile run.

## Diagnostics that exist now, use them first

- `feasibility_verify/preview_cameras.py` -- one frame per camera view, then stops.
  Framing cannot be checked by reading pose numbers; four wrong poses got through
  that way. Also renders control shots straight at each robot, which separates "the
  framing is wrong" from "nothing renders".
- `feasibility_verify/measure_target_capacity.py` -- per-filter rejection histogram
  around any target, with and without a teammate parked next to it.
- `COOP2_ENGINE_VERBOSE=1` -- per-primitive progress lines (`agent_0:PLACE_ON_TOP@1063`).
  This is what exposed #4: tick counts rising linearly with env_step and never ending.
- `[nav]` lines -- sampled pose, distance and travel ticks charged, one per navigate.
- `NO_SPACE_AROUND_TARGET` metadata carries `rejected_by` and `target_xy`.
- `llm_calls.jsonl` + `llm_calls.log` -- every prompt and response, machine- and
  human-readable. `python -m coop2.cognitive.agent.llm_io_log <run_dir>` renders
  the log. This is what settled the wait-only-plan question: replaying the real
  response through the real parse path, rather than reading the parser.
- `agent_timeline.png` -- per-agent FSM over wall clock, with team lanes,
  messages as arrows, env_step ranges inside the bars, and holds shaded apart
  from real execution. Redraw an existing run with
  `python -m coop2.experiment.agent_timeline <run_dir>`.
- `COOP2_TEAM_VERBOSE=1` -- team barrier and chain-ordering decisions, one line
  each. `kill -USR1 <pid>` dumps every thread's stack (faulthandler is
  registered by the runners), which is the only thing that has ever explained a
  silent hang; `StallWatch` prints the not-ready set and why, every 25 s.

## Open, not yet diagnosed

- **Models mangle instance suffixes.** `apple.n.01_01` for `_1` appeared in three
  separate runs; `resolve_target` now normalises zero padding. But putting the id in
  the goal text produced `coffee_table.n.01_01_1` -- a *doubled* suffix, which the
  normaliser does not handle. Prompt wording was tried first and did not hold.
- **`TOO_FAR` after a successful `navigate_to` to the same object** (16 in one run).
  The prompt promises navigation puts you in range. The object was being carried by a
  teammate, so it moved; and the code returned `TOO_FAR` rather than `OBJECT_CLAIMED`,
  which means `holder_of` saw it as unheld -- consistent with the window inside
  `_place_with_predicate` where the object has been released but not yet placed.
- **`progress 0/4`** reads as "nothing started" when it means "action 1 is still
  running". Cosmetic, but it misled a diagnosis once.
- **A welded object left the hand during a teleport-navigate's settle**, once
  in ~20 such trips (agent_3, `individual_agents9_..._202823`, 6830). The
  apple fell at the destination, not at the origin, so it made the 35.6 m
  jump and detached afterwards. Not the same as the 12-robot run's empty-hand
  placement, which was a model writing `place_on_top` after it had already
  delivered. See "One apple left a hand it was welded to".

## Deliberately not implemented

**Target arbitration as a lock.** Contention is enforced as a *precondition
failure*, never as a refusal at `assign()`. The loser still burns the full
navigate and still issues its GRASP, so the wasted decision stays visible to
the cognitive layer — that waste is exactly what the centralized leader's
allocation and the broadcast chain's proposals are measured on. A pre-
assignment lock would hide the signal. `engine.held_objects()` exposes the
cross-agent "who holds what" view for L1b's text observation to surface.
