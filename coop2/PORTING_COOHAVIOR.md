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

`box_mass_kg` is `v4:physical_mass_kg` off the staged box. V4 sets 8 g so a
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

## Known gaps

* **The Jackal cannot grasp** -- no arm, by construction. It navigates, waits,
  and carries. That is the whole of its vocabulary.
* **HL/HH have five boxes and checkpoint FANUC arms.** The FANUC is not
  imported; its URDF ships with Isaac but its meshes do not, so it needs the
  ~614 MB `fanuc_description` clone.
* **Materials are lost** in the URDF -> USD import. Every imported robot is
  white. Harmless for symbolic tasks, awkward in a replay video.
