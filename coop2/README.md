# coop2

COOP²'s multi-agent LLM cooperation stack, ported from `coop2-llm-mas/ma_crafter`
onto BEHAVIOR-1K. A run puts several robots in a house, gives each **team** one
LLM, and measures what the teams' way of talking to each other buys them.

`CLAUDE.md` is the working record — what is built, what is measured, what is
still broken, and why each decision went the way it did. This file is only the
map.

## The one idea the layout follows

A **team** is the unit of every cooperation mode. One LLM plans for all the
robots on a team; the mode decides only how teams address *each other*. Teams of
one reproduce the old one-LLM-per-robot behaviour, so there is no separate code
path for it.

That splits the tree in two. `behavior_env/` and `cognitive/` are what a robot
is and does, and know nothing about cooperation. `comm_topology/` is the
cooperation, and knows nothing about physics.

## Layers, bottom up

| directory | what lives there |
|---|---|
| `behavior_env/` | **L0–L1.** The simulator side. `coop_env.py` is the facade every layer above plugs into; `env_setup.py` builds an N-robot Isaac config; `world_state.py` keeps the live scene graph and `symbolic_view.py` renders it as the text an LLM reads; `primitive_engine.py` steps N primitives concurrently; `symbolic_navigation.py` and `symbolic_contention.py` put distance, travel cost and object claims back into a symbolic primitive set that has none; `route_spec.py` / `route_tracker.py` carry the ordered sub-goals BDDL cannot state; `carrier.py` is cargo on a robot's back; `recording.py` writes the per-robot videos. |
| `cognitive/` | **L2–L4.** `action/` grounds a symbolic action into a primitive, `plan/` owns the plan lifecycle and the `PlanningEnvWrapper` barrier, `agent/` holds the FSM, the memory, the message broker, the prompt builders and the LLM client. `viz/` is a deliberate stub. |
| `comm_topology/` | **L5.** One file per cooperation mode, over `llm_team.py`, which is the team brain and its barrier. `notifying_team.py` is the round `tag` and `board` share, so their difference is the shared space and nothing else. |
| `experiment/` | **L6.** Runners, sweeps, tables, figures, and the tools for reading a finished run. See `experiment/README.md`. |

## Everything else

| directory | what it is |
|---|---|
| `dig_tag/`, `dig_tag_tests/` | DIG-TAG's task-graph package, vendored unmodified at their commit `e39c56e`, with their own golden tests. Exactly one runtime file imports it: `comm_topology/llm_tag.py`. |
| `team_layouts/` | Who is in the scene: one JSON per configuration, giving each robot its model, its start pose, its scale and its team. The layout beats `--agents`. `s1/`, `s2/`, `s3/` are the COOHAVIOR families at 1–5 teams. |
| `robot_configs/` | Primitives-style YAML for the three robots imported from outside BEHAVIOR — Ridgeback+UR5, Jackal, Crazyflie. This is the whole extension point for a new robot model. |
| `omnigibson_definitions/` | The import pipeline that turned those robots' upstream URDFs into OmniGibson models, and the fixes each needed. |
| `plan_scripts/` | Plans written by hand for the `scripted` mode: three engine refusals provoked on demand, and the S1 LL route done by one drone. |
| `runs/` | Where a run lands by default (git-ignored), plus `clean_runs.py`. Sweeps write to `experiment_log/` instead. |
| `figures/` | Figures for the paper, committed so a change to one shows up as a diff. |
| `_repair_shim/` | A stub standing where COOP²'s repair layer would be. Repair is deliberately not ported; this exists so the imports resolve. |

## Documents

- **`CLAUDE.md`** — the record. Read it before changing anything: most of what
  looks like an obvious improvement here has been tried, measured, and written
  down.
- **`PROMPTS.md`** — every section of the prompt an LLM team receives, and the
  file and symbol that produce it.
- **`action.md`** — the closed set of actions a plan may contain, and the three
  files that must agree about each one.
- **`PORTING_COOHAVIOR.md`** — the procedure for porting another COOHAVIOR task,
  including the BDDL parser traps that fail silently.

The two plan documents that used to sit here were deleted once what they planned
was built; `CLAUDE.md`'s opening says so.

## Running something

```bash
# one episode, one mode
OMNIGIBSON_HEADLESS=1 python -u -m coop2.experiment.run_tag \
    --team-config coop2/team_layouts/s1/sets_3.json \
    --scene Merom_1_int --room living_room_0 --bddl-activity v4_s1_v4_ll \
    --steps 6000 --seed 0 --time-limit-seconds 0 --model gpt-5.6-terra --llm-quiet

# the CPU regression checks -- seconds, no GPU. Run these first, always.
for f in feasibility_verify/test_*.py; do python "$f"; done
```

`--time-limit-seconds 0` is not optional when the step count is meant to be the
experiment variable: the default is a 120 s wall-clock deadline that will end
the episode instead.
