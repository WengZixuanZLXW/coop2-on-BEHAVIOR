# coop2/experiment — L6

Everything that runs an episode, runs many of them, or reads what they left
behind. Nothing here knows how cooperation works; it only starts it and counts.

## Runners — one episode, one mode

`run_individual.py` is the real one. It builds the environment from a team
layout, a scene and a BDDL activity, creates one brain per team, and drives the
plan loop: agents plan while the world is frozen, then the world steps while
they execute. It also holds the argument parser every other runner inherits.

| file | |
|---|---|
| **`run_individual.py`** | The runner. Also serves `tag`, `board` and `scripted`, which are the modes with no speaking order — they differ only in what a team is shown and what it may say, which is `comm_topology`'s business, not the loop's. |
| **`run_tag.py`** | `main(topology="tag")`, 22 lines. Adds `--notify-budget`; the run gains `tag.json`, `tag_rounds.json` and the graph drawn to `tag.pdf`/`tag.png`. |
| **`run_board.py`** | `main(topology="board")`. DIG-TAG's ablation of `tag`: the same round with an unstructured message board where the graph was. Writes `board.json`, `board_rounds.json`. |
| **`run_scripted.py`** | `main(topology="scripted")`. Plans come from a JSON file instead of a model, so an episode needs no credentials and does exactly what you wrote. `--print-template` prints a skeleton for a layout. See `../plan_scripts/`. |
| **`run_broadcast_chain.py`** | Its own loop, because a chain has a speaking order: each team waits for the one ahead, then broadcasts onward. |
| **`run_centralized.py`** | Its own loop: the first team assigns, the rest answer, then it plans its own robots with the answers in hand. |

Flags worth knowing on all of them: `--team-config` (the layout is the authority
on who is in the scene, and beats `--agents`), `--time-limit-seconds 0` (the
default is a 120 s wall-clock deadline that will end the episode before
`--steps` does), `--no-video`, `--llm-quiet`, `--gui`.

## Sweeps — many episodes

Each cell is its own process: `og.sim` is a process singleton, so one
environment per process is not a style choice.

| file | |
|---|---|
| **`sweep_grid.py`** | Layouts × tasks × modes, globs allowed for layouts, `--parallel K` for K cells at once (it caps each child's torch threads, or K cells oversubscribe every core K times over and run slower in total than one). Records a cell that crashes or hangs and moves on. Writes each cell to its own folder with its full `stdout.log`, and `grid_summary.json` beside them. |

It is the only one. `run_s1_grid.py` (one layout, every mode × task) and
`run_grid.py` (the crafter-era topology × agent count × repair × seed, whose
repair axis did nothing, repair never having been ported) were deleted: both
were narrower spellings of what `sweep_grid` already does, and a second way to
launch a sweep is a second place for the flags to drift.

## Reading a finished run

| file | |
|---|---|
| **`cooperation_table.py`** | The results table for a sweep directory, one row per cell. CR (route nodes completed / required — not `check_goal`, which cannot say "in this order"), ES, WT wait time, IT interrupt time, ML message load counted per team, IC, RT reasoning time, TT, TO timed out, FZ the share of wall clock the environment was frozen. Also the source of truth other scripts import rather than retyping numbers. |
| **`llm_usage.py`** | Calls, tokens, latency and errors, per agent and per run. Imported by every runner to write `llm_usage.json`; note a *failed* call records no latency, so a call that hung and returned nothing shows only in `total_llm_errors`. |
| **`agent_timeline.py`** | Each agent's FSM state over wall-clock time, one lane per agent and one per team, with messages drawn as arrows between team lanes and team holds shaded apart from real work. `python -m coop2.experiment.agent_timeline <run_dir>` redraws an existing run. |
| **`annotate_video.py`** | Draws the plan record onto one robot's episode video: the whole plan with each action marked, the action running now, and the failure text from the tick it fails on. Alignment is arithmetic, not eyeballed — the recorder captures on `env_step % every == 0`, and every action carries its own start and end. Sized so two clips fit side by side on a slide. |
| **`animate_tag.py`** | Films the task graph growing, a frame per recorded call, each labelled with its env_step, the team that issued it and that team's own reasoning. Frames are real replays into a fresh graph, so a rejected call is shown rejected and not applied. `--calls A-B`, `--seconds N`, `--size WxH`. |
| **`build_results_table.py`** | The crafter-era table, built on constraint deficits (spatial / temporal / dependency) that this project does not measure. It also skips any folder without `team_score.json`, which the runners do not write, so **it cannot read current runs**. |

## Figures and setup

| file | |
|---|---|
| **`plot_scaling.py`** | Completion rate and the cost panels against team size, one line per method, pooling both scenarios and all four tasks — so a point is 24 episodes and its band is as much the spread *between tasks* as between seeds. Takes its numbers from `cooperation_table` rather than a transcribed table; `--out coop2/figures/cr_vs_n` is where the committed ones came from. |
| **`inspect_scene.py`** | Loads a task's scene and stops. No agents, no LLM, no plan loop; it prints where each robot was asked to be against where it ended up, and `--view <robot>` prints exactly what that robot would be shown. Reach for this before spending an episode on a new scene, task or layout. |
| **`stall_watch.py`** | Names the agent a stalled run is waiting for. Several paths can leave an agent that will never be ready and all of them are silent; this prints the not-ready set and why, every 25 s. Used by the runners, not run on its own. |

## Two things that bite

- **`python -u`, always, when stdout is redirected.** Isaac's shutdown ends the
  process without flushing, so everything after the last simulator print is
  lost — including the goal line and the plan statistics. A run that looks like
  it stopped early and said nothing is usually this.
- **Do not wrap a long run in `conda run`.** It buffers stdout until the process
  exits, which makes an hour-long run unmonitorable. Call the env's python
  directly.
