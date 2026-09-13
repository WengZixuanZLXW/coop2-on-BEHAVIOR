# Porting a COOHAVIOR V4 task into BEHAVIOR

How to turn one of COOHAVIOR's twelve V4 tasks into a BEHAVIOR activity this
project can run. Written from doing `v4_s1_v4_ll` end to end; the traps below
are the ones that actually cost time, not hypotheticals.

Source: `/home/zixuanwe/Desktop/COOHAVIOR/custom_task_suite/push_pull_arm_v4/`.

## What you are porting, and what you are not

COOHAVIOR runs on standalone Isaac Sim 5 + ROS 2 Nav2. **None of that comes
across, and none of it needs to.** Its own manifest says Nav2 is "navigation
infrastructure, not the benchmark metric", and its motion is a kinematic
unicycle integration of `cmd_vel` -- its bridge file says so outright:
"fake-motion", "No wheel or PhysX-contact claim is made."

What you port is the *task*: a BDDL goal, a scene layout, and a robot layout.
Three files come out the other end, and they divide cleanly:

| File | Holds | Does **not** hold |
| --- | --- | --- |
| `bddl3/bddl/activity_definitions/<activity>/problem0.bddl` | goal and initial conditions | any coordinate |
| `datasets/2026-challenge-task-instances/scenes/<scene>/json/<scene>_task_<activity>_0_0_template.json` | every object's pose | **any robot** |
| `coop2/team_layouts/<activity>.json` | robot models, teams, poses, scales, capabilities | anything about objects |

The twelve tasks and their scenes:

    S1-V4-{LL,LH,HL,HH}   Merom_1_int
    S2-V4-{LL,LH,HL,HH}   Beechwood_0_int
    S3-V4-{LL,LH,HL,HH}   Beechwood_1_int

All three scenes exist in `datasets/behavior-1k-assets/scenes/`, and every
synset the twelve tasks use is valid in our taxonomy. The goals are all
`ontop`. There is no asset work.

## Step 1 -- the BDDL

Copy `push_pull_arm_v4/bddl/<TASK>.bddl` to
`bddl3/bddl/activity_definitions/<activity>/problem0.bddl`, then make three
changes. The first two are not cosmetic: without them the file loads and is
**silently wrong**.

1. `(:domain omnigibson)` -> `(:domain behavior-1k)`, and the problem name to
   `<activity>-0`. Ours is the `behavior-1k` domain; there is no
   `domain_omnigibson.bddl` in `bddl3`.

2. **Put every instance of one synset on a single line.** BDDL's parser does
   `objects[type] = object_list` -- an assignment. Two lines of
   `- floor.n.01` means the second silently overwrites the first, and you lose
   an object with no error at all:

       floor.n.01_5 - floor.n.01              # WRONG: floor.n.01_5 vanishes
       floor.n.01_target_1 - floor.n.01

       floor.n.01_1 floor.n.01_2 - floor.n.01 # right

3. **Instance names must be `<synset>_<integer>`.** `synset_from_bddl_inst`
   takes everything before the last underscore, so `floor.n.01_target_1`
   yields the synset `floor.n.01_target`, which does not exist. Renumber.

Add `agent.n.01_1 - agent.n.01` and an `(ontop agent.n.01_1 <floor>)` initial
condition if the source lacks them; BEHAVIOR binds that instance to
`robots[0]`.

3. **No `*` anywhere in the file, comments included.**
   `knowledge_base/models.py:_strip_wildcards` runs over the *raw text* and
   treats every line containing `*` as a wildcard scope line; a comment such
   as `; the spawn *marker* sits -- on a fixture` then hits
   `line.split(" - ")[1]` and dies with `IndexError` inside
   `Environment.__init__`, after the scene has loaded. Found 2026-09-12 on
   the first HL/HH re-sample -- two GPU runs to learn that emphasis is a
   wildcard. `test_route_spec_stubbed.py` now runs that pass over every S1
   definition on CPU.

Verify before going further:

```bash
cd bddl3 && python -c "
import sys, json; sys.path.insert(0, 'tests')
import bddl.bddl_verification as ver, bddl.parsing as parse
from bddl_tests import verify_definition
syns = json.load(open(ver.SYNS_TO_PROPS_JSON))
*_, dp = parse.parse_domain('behavior-1k')
verify_definition('<activity>', syns, dp, csv=False); print('OK')"
```

Then check the parse kept everything:

```bash
python -c "
from bddl.activity import Conditions
c = Conditions('<activity>', 0, 'behavior-1k')
print(c.parsed_objects); print(c.parsed_goal_conditions)"
```

## Step 2 -- the scene edits

V4 edits the BEHAVIOR scene in three ways, and **where those edits live differs
per scene**. Get this wrong and you reproduce nothing:

| Scene | Deactivations in the staging file's own `BehaviorScene` block | Separate layer |
| --- | --- | --- |
| S1 | 8 | **89 more** in `isaac5/s1_task_only_furniture_filter.usda`, referenced at line 33 |
| S2 | 86 | none |
| S3 | 77 | none |

Reading only the staging file's own block gives you 8 removals for S1 instead
of 97. The reference is near the top of the file, above the prim tree.

The coordinate frames are the same one, so numbers transfer verbatim. Checked
on S1: of eight BEHAVIOR objects the staging scene edits, five sit at
*identical* coordinates in `Merom_1_int_best.json`, and the three that differ
are the moves V4 makes deliberately. Do the same check on a new scene before
trusting it:

```bash
python - <<'PY'
import json, math, re
usd = open("<staging>.usda").read()
b = usd[usd.index('def Xform "BehaviorScene"'):usd.index('def Xform "V4_Task_Staging"')]
reg = json.load(open("datasets/behavior-1k-assets/scenes/<scene>/json/<scene>_best.json"))
reg = reg["state"]["registry"]["object_registry"]
cur = None
for line in b.splitlines():
    m = re.search(r'over "([^"]+)"', line)
    if m: cur = m.group(1); continue
    if cur and "xformOp:translate" in line and cur in reg:
        v = [float(x) for x in re.search(r'\(([^)]*)\)', line).group(1).split(",")]
        print(f"{cur:28} moved {math.dist(v[:2], reg[cur]['root_link']['pos'][:2]):.2f} m")
        cur = None
PY
```

Write what you find to `coop2/team_layouts/<activity>_scene.json`:

```json
{
  "deactivate": ["door_lvgliq_1", "picture_...", ...],
  "move": {"armchair_qplklw_2": [-2.163, 6.817, 0.281]},
  "box_mass_kg": 0.008
}
```

Keep only names that exist in the target scene -- S1's list has 89 entries of
which 84 do. **Do not remove what the BDDL binds to.** V4's own filter keeps
the beds, the coffee table, the shelf and the fridge for exactly that reason;
its header comment lists them.

`box_mass_kg` comes from `COOHAVIOR/behavior_style_task/tasks.modified.json`,
`tasks[].box_mass_kg` -- the staging USD's `v4:physical_mass_kg` agrees for S1
and is absent for S2/S3. V4 sets 8 g so a
7 cm drone's suction cup can lift it. The symbolic grasp does not read mass --
it teleports and welds -- but the physics after a release does.

## Step 3 -- the instance

Copy `feasibility_verify/sample_v4_s1_v4_ll.py` and change the constants at the
top: `ACTIVITY`, `SCENE_MODEL`, the pinned object coordinates, `SAMPLING_WHITELIST`,
and `SCENE_EDITS`.

The script samples the activity natively, then **overrides the layout** with
V4's own coordinates rather than accepting what the sampler drew. That is the
point of porting: the sampler is used only to bind the BDDL scope and build a
loadable instance around the pinned objects.

Three things in it are load-bearing, all learned the hard way:

* **Apply the scene edits *after* `env.task.reset(env)`.** `reset` restores the
  scene from its initial file, so anything removed before it comes straight
  back. The first attempt printed "removed 84 objects" and saved a template
  with all eleven doors still in it.

* **Wake the task objects** (`sleep_threshold = 0`, `wake()`) before checking
  `OnTop`. A sleeping PhysX actor reports no contacts, so the predicate comes
  back False for an object sitting exactly where it should be.

* **Strip the robots from the written template.** `save_task` dumps the whole
  scene including whatever robot the sampler used, and
  `Environment._load_robots` is guarded by `if len(self.scene.robots) == 0` --
  so a robot left in the template silently overrides the entire team layout.
  Symptom: `KeyError: 'JointController'`, because the template's R1 is handed a
  config written for a different robot.

Run it and read the output: it reports the drift from each pinned coordinate
(want 0.000), whether the initial predicates hold, and that the goal is **not**
already satisfied.

## Step 4 -- the team layout

Positions come from `benchmark_audit/audit_index.json`, which already has every
robot's world pose for all twelve tasks at each robot-set count. No USD parsing
needed.

```json
{
  "robots": [
    {"name": "agent_0", "model": "v4_ridgeback_ur5",  "position": [-1.1703, 0.4624],
     "scale": 0.5, "team": "alpha", "base_locked_while_holding": true},
    {"name": "agent_1", "model": "v4_jackal",         "position": [-0.8904, -0.0350],
     "scale": 0.7, "team": "alpha"},
    {"name": "agent_2", "model": "v4_crazyflie_cf2x", "position": [-1.2314, 0.0176],
     "scale": 0.6, "team": "bravo"}
  ]
}
```

**The per-set-count layouts live in `coop2/team_layouts/s1/sets_<k>.json`**,
k = 1..5 -- the audit's `robot_set_count`, 3k robots -- and they do **not** use
V4's staged poses. Everyone spawns together in the most open room (decision:
user, 2026-09-12), packed around the centroid of that room's largest free
region, nearest-first in V4's route order M5, M3, M2, M1, M4, so `sets_<k>` is
exactly the first 3k robots of `sets_5`. "Most open" is measured, not eyeballed:
traversable floor from the no-object map, minus the oriented footprint of every
object V4 keeps, eroded by the largest robot footprint (a 0.7 m Jackal). For
Merom_1_int, in m2: living_room_0 12.0, childs_room_0 4.9, bedroom_0 3.6,
kitchen_0 2.7, corridor_0 0.6, dining_room_0 0.00. Clearances are square
(Chebyshev), matching the footprints -- a circular erosion under-protects a
square robot's corners by ~0.1 m and let one Jackal nick the coffee table.

Why not the staged poses: four of V4's fifteen are inside furniture the filter
keeps -- the kitchen Jackal and drone in `furniture_sink_czyfhq_0`, the dining
Jackal and Ridgeback in `breakfast_table_skczfi_0`'s footprint -- and the dining
room cannot hold a Jackal at all (0.00 m2 admissible). Found by footprint
geometry, not by eye: a robot inside a sink renders fine from above. The staged
pose is kept on each robot as `_v4_staged_position` for reference.

**`scale` is not optional.** V4 stages its three robots within half a metre of
each other, which only fits because it shrinks them -- Ridgeback 0.5, Jackal
0.7, Crazyflie 0.6, read off `xformOp:scale` in the staging file. Loaded at
full size two of them overlap, `place_robots` relocates them by 0.8 m and
1.9 m, and the relocation shoves the task's box 1.4 m across the room. Size is
part of a layout, not a property of a model.

`base_locked_while_holding` mirrors `v4:base_locked_while_holding` on the
staged prim. It is what makes a task need more than one robot: the arm that
picks the box up cannot drive, so it has to `load_onto` a carrier. Without it
one robot does the whole job and the others watch.

`carrier` is inferred for a robot with no arm; set it only to override.

The three robots are already imported and registered. A fourth would need
`prepare_official_robot_imports.py` plus a `<model>_primitives.yaml` in
`coop2/robot_configs/`, which is auto-registered at import.

## Step 5 -- look at it before spending an episode

```bash
OMNIGIBSON_HEADLESS=1 python -m coop2.experiment.inspect_scene \
  --scene <scene> --room <room> --bddl-activity <activity> \
  --team-config coop2/team_layouts/<activity>.json \
  --view agent_0 --shot /tmp/scene.png
```

Loads everything and stops -- no agents, no LLM, no plan loop. Check:

* every robot's **drift** from its asked-for pose is ~0. Anything larger means
  `place_robots` moved it, which means the poses collide at the scale you set.
* the **goal is not satisfied at load**.
* `--view AGENT` prints the text a robot is actually given. This is the cheapest
  way to catch a task that cannot be stated at all: if the destination is not
  nameable, the agent will plan to put the box on top of itself, and you will
  only find out after a full episode.
* `--shot` writes an overhead PNG. Overhead deliberately -- an oblique camera
  lands inside a wall in these rooms and returns flat grey.

Then run it:

```bash
OMNIGIBSON_HEADLESS=1 python -u -m coop2.experiment.run_individual \
  --team-config coop2/team_layouts/<activity>.json \
  --scene <scene> --room <room> --bddl-activity <activity> \
  --goal "<one line, plain English>" \
  --steps 3000 --time-limit-seconds 1200 --model gpt-5.6-luna --llm-quiet --no-video
```

`-u` matters whenever you redirect: Isaac's shutdown ends the process without
flushing, so everything after the last simulator print is lost -- including the
goal line and the plan statistics. A run that looks like it stopped early and
said nothing is usually this.

`--time-limit-seconds` defaults to 120, which any real task exceeds.

## What differs across the twelve

Only three things, so most of the work above is done once per *scene*, not per
task:

* **Boxes**: 1 in LL/LH, 5 in HL/HH.
* **Robots**: 1 set (3 robots) in S2/S3 LL/LH; 5 sets (up to 20) elsewhere.
* **Markers**: the route, which our BDDL expresses as `ontop` goals.

The furniture layer is *identical* across all four difficulties of a scene --
checked object by object, not by hashing the text. So three scene instances
cover twelve tasks, and S2-LL and S2-LH differ in nothing our symbolic layer
models: their scenes, robots and boxes are all the same, and the difference V4
intends lives in a package dependency graph we do not implement.

## What the second and third ports found (2026-09-12)

`v4_s1_v4_lh`, `v4_s1_v4_hh` and `v4_s1_v4_hl` are done, sampled and inspected. Doing them
turned up eight things the LL port had not, six of which fail *silently*.

### S1-V4-HL: the goal holds at t=0, and that is the task

All five of its goals are `ontop(box_i, floor.n.01_target_i)` where the target
floor is `inroom` the **same room** the box already stands in. V4's targets are
marker positions on a floor; ours is a room-level predicate over floor objects,
and Merom_1_int has exactly one floor object per room -- measured, all twelve.
So the BDDL collapses each start/target pair to one floor instance (BDDL's
parser could not carry two `- floor.n.01` lines anyway), and every goal clause
is literally an initial condition.

**Resolved 2026-09-12, the other way round.** COOHAVIOR's real task structure
is the ordered checkpoint sequence of each route in `tasks.modified.json`, and
the route is the task (user): when the BDDL disagrees with it the BDDL is what
gets corrected. HL's five goals are now the routes' destinations --
`ontop(die_1, bookcase.n.01_1)`, `(die_2, armchair.n.01_1)`, `(die_3,
coffee_table.n.01_1)`, `(die_4, bed.n.01_2)`, `(die_5, bed.n.01_1)` -- with the
checkpoint cabinets declared alongside, so nothing holds at t=0
(`unsatisfied: [0, 1, 2, 3, 4]` at load), the `allow_trivial_goal` switch is
gone, and `terminated` is decided by the route tracker
(`coop2/ROUTE_SUPERVISION_PLAN.md`, steps 1-3 done). The boxes still start on
their stations' floors: COOHAVIOR's `initial_relation` says so, its staged box
*prims* -- the coordinates the sampler pins to -- sit on the floor beside the
fixtures, and only its spawn *markers* sit on them; the prim is what spawns.

The same collapse will appear in S2-HL and S3-HL's BDDLs as read off
COOHAVIOR; write their route files first and the goals follow from them.

### The two box weights are two objects now

`COOHAVIOR/behavior_style_task/tasks.modified.json` is the authority on box mass:
`tasks[].box_mass_kg` is 0.008 for LL and HL, 0.02 for LH and HH, and V4 calls
every one of them `packing_box.n.02_N`. Nothing symbolic could tell them apart.
The light cargo is `die.n.01` (dice-iswudu) and the heavy one `notebook.n.01`
(notebook-aanuhi); the mass is still set explicitly. (The S1 staging USDs carry
the same numbers as `v4:physical_mass_kg`; S2/S3's carry none.)

**The weight is a capability constraint, enforced since 2026-09-12 by the
route file.** The same file's `condition_physical_execution_contract` says the
drone "may suction-lift/transport an 8 g box" and "is not a load-bearing role
for the 20 g box"; light packages need only a `push_car`, heavy ones need
`pull_car + push_car + robot_arm` together. That is the whole reason the two
weights had to become two objects -- so "the Crazyflie may grasp a die but not
a notebook" can be *said*. It is said in `route.json`'s `lift` table (cargo
synset -> roles), and a robot's role comes from the layout: `carrier` if it has
no arm, `drone` if the entry says `"drone": true`, else `arm`. Not a payload
number: COOHAVIOR's contract is a table of roles, and a mass would have made us
invent one. `symbolic_contention._require_may_lift` refuses `grasp` and
`unload_from` with `CANNOT_LIFT`, and `target_hints` withholds the verb --
both sides, or the agent burns a plan learning it. No mark on the line: the
system prompt says "the drone cannot lift the notebook" and the drone's header
says `(drone)` (ROUTE_SUPERVISION_PLAN.md section 5.4).

A side effect worth having: the notebook is 0.151 x 0.120 x 0.028, against the
packing box's 0.380 x 0.468. All five HH cargoes now sit at V4's exact staged
coordinates. The packing box could not -- `box_s1_m2` is staged at precisely
the coordinate the same file moves `armchair_qplklw_2` to, so a box with
collision lands on the armchair and its own `ontop(..., floor)` is false. V4's
box asset is documented "visual-only until collision is deliberately authored",
which is how its staging can contradict its own BDDL.

### `seg_map_resolution: 1.0` decides room membership, and gets it wrong

The shipped primitives configs set 1 m. At 1 m the whole of Merom_1_int is a
**20x20 grid** and an object's room is whichever of 400 cells its centre lands
in. Measured against the scene's own `in_rooms` annotations, 13 of 13 cabinets
agree at 0.1 and **4 of 13 at 1.0**: seven land on a boundary cell and read as
no room at all, and both bedroom cabinets near the party wall read as
`childs_room_0`.

That is not cosmetic. It turned `v4_s1_v4_lh` from "carry it to the bedroom"
into "put it on the cabinet behind you" -- the cabinet appeared in the agent's
own room listing 2.6 m away -- and left `v4_s1_v4_hh`'s dining-room notebook in
no room at all. `coop2/robot_configs/v4_*_primitives.yaml` are now 0.1. The
shipped R1 and Tiago configs still say 1.0 and are left alone: these tasks
only have to run under the three v4 configs (user, 2026-09-12).

### Sample against the map of the scene you actually run

`sample_kinematics(use_trav_map=True)` is a *reachability* test: it erodes the
floor map by the robot's radius and dilates by arm reach. The baked
`floor_trav_0.png` includes all the furniture COOHAVIOR deactivates, so it
describes a scene the task is not in. Free area per room after eroding by R1's
0.62 m, with objects -> without:

    dining_room_0   0.00 -> 1.23        living_room_0  3.65 -> 11.33
    childs_room_0   0.36 -> 5.35        bedroom_0      0.67 ->  7.06
    kitchen_0       0.41 -> 3.98

`dining_room_0` is **zero** with objects, so `ontop(cargo, that floor)` could
never sample at any attempt count -- it dropped the dining room out of the room
intersection and failed the whole activity. Thirty attempts bought four and a
half minutes of failure. The other four rooms scraped through on tenths of a
square metre, which is why only one room looked like the problem. Pass
`trav_map_with_objects: False`. Removing the objects first does **not** work:
the map is a baked PNG and deleting objects at runtime does not redraw it.

### Do not remove what the BDDL bound to

`inroom` binds by room *type* and a room holds several objects of a category --
bedroom_0 has four cabinets, dining_room_0 three `qplklw` armchairs -- so the
sampler may bind the very instance V4's filter deactivates. It did, on the first
LH draw. Nothing raises: the template saves, loads, and the goal names an object
that is not in the scene. `_apply_scene_edits` now skips anything in
`task.object_scope` and says which.

### Check room membership by the seg map, not by `in_rooms`

They disagree. `bottom_cabinet_jrhgeu_1` is annotated `bedroom_0` and, at the
runtime resolution, stands in `childs_room_0`. The world model, the room
listing and `inspect_scene` all read the seg map, so the seg map is what the
agent is shown -- and an assertion against the annotation passes while the port
is wrong.

### `scene.robots` is alphabetical, and two places paired it positionally

`scene_base.robots` returns `sorted(..., key=lambda x: x.name)`. At ten agents
that is agent_0, agent_1, agent_10, agent_11, ..., agent_2, while `agent_names`
is the layout's order. `coop_env` zipped the two to build the controller map
and to apply `base_locked_while_holding` / `carrier`, and `inspect_scene` zipped
them to print its table: at 15 robots agent_2's controller drove the robot named
agent_10, and the base lock landed on whichever robot sorted into that slot.
Invisible below ten agents, where the orders coincide -- the same trap as the
2026-09-11 `entity_id_for` bug. Fixed by keying on name in all three places.
**Any run with ten or more agents predates this fix.**

### Dice and Crazyflie are scale 2.0, for every port from here on

Decided 2026-09-12: the die (`iswudu`, 1.8 cm native) and the Crazyflie
(12 cm native) are both scaled x2, uniformly. The die's scale is the one-element
list in the sampler whitelist (`{"iswudu": [2.0]}` -- one dimension means scale,
three mean a bounding box) and is baked into the cached instance; the drone's is
`"scale": 2.0` in the layout. S2 and S3 use the same values.

### The scales in the LL layout are not V4's

`S1-V4-*.usda` says jackal 0.5, ridgeback_franka **1.0**, crazyflie 0.6. The LL
layout and Step 4 above both say "Ridgeback 0.5, Jackal 0.7, Crazyflie 0.6,
read off `xformOp:scale`", and two of the three do not match the file. LL solves
at env_step 504 with its numbers, so LH and HH keep them rather than change
quietly to V4's. Unresolved.


## Known gaps

* **The Jackal cannot grasp** -- no arm, by construction. It navigates, waits,
  and carries. That is the whole of its vocabulary.
* **HL/HH have five boxes and checkpoint FANUC arms.** The FANUC is not
  imported; its URDF ships with Isaac but its meshes do not, so it needs the
  ~614 MB `fanuc_description` clone.
* **`base_footprint_link_name` must be the link below all six virtual joints.**
  The drone's said `base_footprint_z` -- the link *above* the z joint, which
  does not move when z does -- so every pose readout (`get_position_orientation`,
  `inspect_scene`'s z, drift, room) came from a link that stays at 0.05 while
  the body was at 1.2 m. Measured link by link before believing a single number.
  It also decides gravity: only that link's fixed subtree keeps it, so the body
  had been floating weightless. Now `world` (the importer's name for the root
  body), like every other robot's real base link.
* **A drone's z joint is free unless you drive it.** The holonomic import puts
  a `PhysicsDriveAPI` on x, y and rz only -- what the 3-DOF base controller
  owns -- so once the body has weight a free z joint falls. `coop2/omnigibson_definitions/fix_drone_altitude_drive.py`
  applies a driven linear z joint (kp 100, kd 10) after import; upstream refuses
  a driven joint no controller owns, so the model YAML gives z to the drone's
  otherwise joint-less `arm_0` JointController (idle command = current position),
  and `place_robots` aims it at the spawn altitude, so a layout `[x, y, z]` holds. Spawn the Crazyflie 1.2 m above its own Jackal and the trio is born in
  one spot.
* **Every navigate lowered the drone by 5 cm.** Upstream's
  `_get_robot_pose_from_2d_pose` returns a holonomic base's z *joint* value as a
  *world* z; the joint is measured from the root anchor at the 0.05 m spawn
  height, so each teleport re-lands the body 5 cm under where it was. Measured
  1.200 -> 1.150 -> 1.100 -> 1.050 over three navigates. Wheeled bases never
  show it (the floor pushes back). `NavigableSymbolicActionPrimitives` now
  overrides it to keep the body's current world z.
* **Mesh colours are dropped by the URDF importer, and put back by a script.**
  Isaac 5.1's importer writes a URDF `<material>` only onto primitive visuals
  (a `<cylinder>`, a `<box>`); every `<mesh>` visual gets a white
  `DefaultMaterial*` per mesh *file* instead, and the coloured `material_<name>`
  Looks it does create end up bound to nothing the mesh inherits. Measured:
  Jackal 7 of 7 mesh prims white, Ridgeback+UR5 23 of 23 (the UR5's DAE parts
  carry their own colours), Crazyflie 0 (a DAE). Run
  `coop2/omnigibson_definitions/fix_robot_visual_materials.py` after every
  import: it walks the resolved URDF for link -> mesh -> colour and rebinds each
  white Mesh prim, per mesh rather than by recolouring the shared default --
  Ridgeback's `lights.stl` is white on the front link and black on the rear and
  both share one DefaultMaterial. Leaves `<name>.usda.orig` beside the file.
