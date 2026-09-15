# The action set

Every action an LLM may put in a plan. Three files decide what happens to one:

- **Schema** -- [`cognitive/agent/llm_client.py`](cognitive/agent/llm_client.py) lines 50-130.
  One Pydantic class per verb; the union is what structured output can return,
  so a verb not here cannot be spoken.
- **Guards and primitives** -- [`behavior_env/symbolic_contention.py`](behavior_env/symbolic_contention.py).
  `ContentiousSymbolicActionPrimitives` runs the action and raises the failures
  below. This is the authority: a plan is only what was asked for.
- **What the prompt offers** -- [`behavior_env/symbolic_view.py`](behavior_env/symbolic_view.py)
  lines 380-522. The per-target verb list a robot sees. It re-applies the same
  gates, so "not listed" and "would be refused" are two faces of one rule.

Roles, as S1 uses them: **arm** (Ridgeback + UR5, declares
`base_locked_while_holding`), **carrier** (Jackal, no arm at all), **drone**
(Crazyflie, a belly suction cup, `lift_role` = `drone`). The role is read by
[`behavior_env/carrier.py`](behavior_env/carrier.py) `is_carrier` / `lift_role`.

| action_name | available_robot | parameter | effect | failed return |
|---|---|---|---|---|
| `navigate_to` | arm / carrier / drone | `target` -- an object or a robot id | Charges 20 ticks/m of travel, then teleports to a standable pose near the target; wide furniture is approached at its edge band; whatever is welded to the robot (held object, cargo on its back) moves by the same rigid transform | `BASE_LOCKED` (holding something and this base locks), `INVALID_TARGET`, `PLANNING_ERROR` / `SAMPLING_ERROR` (no standable pose), `TIMEOUT` |
| `grasp` | arm / drone | `target` | Welds the target to the end effector | `NO_ARM`, `CANNOT_LIFT` (the route's lift table bars this role), `ALREADY_HELD`, `OBJECT_CLAIMED` (someone else holds it, or it rides a carrier), `TOO_FAR`, `POST_CONDITION_ERROR` |
| `place_on_top` | arm / drone | `target` -- a receptacle | Puts the held object on the target's top surface | `NO_ARM`, `OBJECT_CLAIMED`, `TOO_FAR`, `PRE_CONDITION_ERROR` (hand empty), `POST_CONDITION_ERROR` (the predicate still reads false after `PLACE_GRACE_TICKS` = 8) |
| `place_inside` | arm / drone | `target` -- openable or fillable | Puts the held object inside the target | as `place_on_top` |
| `release` | arm / drone | none | Drops the held object where the robot stands | `NO_ARM`, `PRE_CONDITION_ERROR` (hand empty) |
| `open` | arm / drone | `target` | Target `Open` = True | `NO_ARM`, `OBJECT_CLAIMED`, `TOO_FAR` |
| `close` | arm / drone | `target` | Target `Open` = False | as `open` |
| `toggle_on` | arm / drone | `target` | Target `ToggledOn` = True | `NO_ARM`, `OBJECT_CLAIMED`, `TOO_FAR` |
| `toggle_off` | arm / drone | `target` | Target `ToggledOn` = False | as `toggle_on` |
| `load_onto` | arm / drone | `target` -- a carrier robot id | Releases the held object from the hand and welds it to the carrier's back | `NO_ARM`, `INVALID_TARGET` (not a robot, has an arm of its own, or is yourself), `TOO_FAR`, `PRE_CONDITION_ERROR` (hand empty, or the carrier is already loaded) |
| `unload_from` | arm / drone | `target` -- a carrier robot id | Takes what rides on the carrier into this robot's hand | `NO_ARM`, `INVALID_TARGET`, `TOO_FAR`, `PRE_CONDITION_ERROR` (hand full, or the carrier is empty), `CANNOT_LIFT` |
| `wait` | arm / carrier / drone | `ticks`, 1-600, default 200 | Holds the current joint configuration so time passes | none -- an out-of-range value is clamped to [1, 600] |

## What the table does not say on its own

**A carrier has two verbs.** The Jackal has no arm, so everything else is
`NO_ARM`. Cargo goes onto and off its back through `load_onto(jackal_1)` and
`unload_from(jackal_1)` -- hand verbs, performed *on* the carrier by a robot
that has a hand, never by the carrier itself.

**A verb withheld from the prompt is also refused by the engine**, and the
reverse: the view drops `navigate_to` while a locking base is loaded, drops
`grasp` silently when the lift table bars the role (no "too heavy" mark on the
line, by decision), and collapses an out-of-reach object to a single
`unreachable` line. A verb the prompt offers and the engine refuses costs a
whole plan and an LLM round trip to discover.

**`CANNOT_LIFT` comes from the route file's `lift` table**, keyed by cargo
synset, not from mass. In S1 `notebook.n.01` allows `arm` only, so a drone's
grasp is always refused; `die.n.01` allows all three roles. With no route file
the table is empty and anyone with a hand may lift anything.

**`TOO_FAR` is measured against the support, not the object**, and standing at
the edge of wide furniture counts as being in reach -- except for walkable
surfaces such as floors, or a whole room would be graspable.

**`wait` is a real primitive.** The plan loop does not step the environment
while any agent is not ready, so an instant wait would stop the world instead
of yielding it: the teammate being waited for would advance by exactly nothing
while the wait burned an LLM call.

`open`, `close`, `toggle_on` and `toggle_off` are in the schema and the engine,
but they are offered only for an entity that actually carries the `Open` or
`ToggledOn` state (`_STATE_GATES`). None of the four S1 task objects does, so
they do not appear in an S1 run.
