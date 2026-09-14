# Where an experiment's time and CPU actually go

Measured 2026-09-14 against the `S1_full` sweep and the archived runs in
`experiment_log/`, then re-measured after the fix in section 3. Two of the
three costs below were investigated by profiling rather than by reading, and
the profile disagreed with every guess that preceded it, including a plausible
one about a cache.

## The headline numbers

| Item | Before | After |
|---|---|---|
| Simulation per env step, 9 robots, HL | 469 ms | 47 ms |
| One macro step, 9 robots in one room | 606 ms | 41 ms |
| One macro step, 9 robots spread over 8 rooms | 214 ms | 11 ms |
| Simulation per env step, 3 robots | 23 to 28 ms | unchanged |
| Raw `env.step`, idle, 9 robots | 16.6 ms | unchanged |
| GPU memory / utilisation, one run | 1.85 GB of 16.3 GB, 0% | unchanged |
| CPU, one run | 5 to 13 cores | unchanged |
| Aggregate steps/s, 3 cells at once | 16.5 | 133.8 with one BLAS thread each |
| Isaac startup and scene load | 30 to 75 s | unchanged |

The end-to-end check: `individual`, `s1/sets_3`, `v4_s1_v4_hl`, seed 0. The
archived run reached 8 of 10 route nodes in 3972 steps and 2151 s. The same
configuration after the fix completed 10 of 10 at env step 893, with 0
out-of-order visits, in 266 s of wall clock.

## 1. Three quarters of the CPU is a PyTorch thread pool spinning

Sampled on a live run with `s1/sets_1` (3 robots), 209 s in: 1374 CPU-seconds,
averaging 6.6 cores, peaking near 13.

| Thread group | Threads | CPU seconds | Share |
|---|---|---|---|
| python workers | 15 | 1030 | 75% |
| python main thread | 1 | 90 | 6.5% |
| `tbb.worker` | 32 | 78 | 5.7% |
| `carb.tasking*` | 35 | 60 | 4.4% |
| rtx, vulkan, jemalloc, misc | ~20 | 110 | 8% |

The 15 workers are PyTorch's intra-op pool: the process maps torch's bundled
`libgomp-c9fef706.so.1` next to `libtorch_cpu.so`, `torch.get_num_threads()` is
16 here, and nothing on the runtime path calls `set_num_threads`. They are
spinning at a barrier, not computing: their CPU times agree to within 0.15% over
68 seconds, `stime` is 0.02 s against 62 s of `utime`, and `wchan` is 0.

**Capping them does not help a single run, and is worth an order of magnitude
across parallel ones.** Alternating thread counts three times over 60 ticks
each, one process alone: 16.6 ms per `env.step` on 16 threads against 17.8 ms
on one, so a lone cell is 7% slower capped.

Run several cells at once and that reverses completely, because each one opens
its own pool and they oversubscribe every core K times over. Aggregate steps
per second across all cells, nine robots each, stepping idle actions:

| Cells | Threads each | ms per step, per cell | Aggregate steps/s |
|---|---|---|---|
| 1 | 16 | 16.4 | 60.6 |
| 3 | 16 | 163 to 204 | 16.5 |
| 3 | 1 | 22.5 | 133.8 |
| 6 | 1 | 23 to 26 | 242.0 |

Three uncapped cells are slower **in total** than one cell alone. Capped, three
cells are 2.2 times one cell and six are 4 times. `sweep_grid` now sets
`OMP_NUM_THREADS` for its children whenever `--parallel` is above 1, through
`--threads-per-cell`, default 1.

How the thread breakdown was taken, which is cheap and safe to repeat during a
run:

```bash
python3 -c "
import os,collections
pid='<PID>'; base=f'/proc/{pid}/task'; HZ=os.sysconf('SC_CLK_TCK')
agg=collections.defaultdict(lambda:[0,0.0])
for t in os.listdir(base):
    comm=open(f'{base}/{t}/comm').read().strip()
    st=open(f'{base}/{t}/stat').read(); f=st[st.rindex(')')+2:].split()
    a=agg[comm]; a[0]+=1; a[1]+=(int(f[11])+int(f[12]))/HZ
for k,(n,c) in sorted(agg.items(), key=lambda kv:-kv[1][1])[:8]: print(k,n,round(c,1))
"
```

## 2. Most of an episode is the world frozen waiting for the model

The plan loop does not step the environment while any agent is not ready, so the
whole simulation stops for every model call. Union of the planning spans in each
run's `team_timeline.json`, over the episode's wall clock:

| Layout | Teams | Robots | Frozen share |
|---|---|---|---|
| `s1/sets_1` | 1 | 3 | 64% to 81% |
| `s1/sets_3` | 3 | 9 | 11% to 64% |

Union, not sum: with three teams some planning overlaps other teams' execution,
though only by about 18% on the chain LH runs, because teams reach their
barriers at similar times.

After section 3's fix this share goes up, not down. In the end-to-end check
above, 224 s of the 266 s episode was the world frozen and 42 s was ticking. For
nine robots the model is now the bottleneck, and it is latency rather than
compute: the CPU is idle for most of it.

## 3. The observation path recomputed the same geometry hundreds of times

**The symptom.** Per-step simulation cost, wall clock minus frozen time over env
steps reached:

| Run | Robots | Active robots | ms per env step |
|---|---|---|---|
| `sets_1`, all four tasks | 3 | 2 to 3 | 23 to 28 |
| `sets_3`, LL | 9 | 2 to 3 | 63 |
| `sets_3`, HL, individual | 9 | up to 9 | 469 |
| `sets_3`, HL, broadcast_chain | 9 | up to 9 | 667 |

Same scene, same robots, same build. LL carries one cargo so most robots idle;
HL has five parallel routes and keeps all nine working, which makes a macro step
happen far more often.

**Two guesses, both wrong.** The single-entry cache in `_facts_scoped` really
does thrash when robots query rooms A, B, A, and `entities()` really was rescanning
the whole scene once per agent. Fixing both moved a 606 ms macro step to 553 ms.
Together they were under 10% of the cost.

**What it actually was.** `cProfile` over three macro steps put 85% of the time
in one chain: `target_hints` calls `interaction_radius_for` per entity, which
calls `support_of`, which answers "what is this resting on?" by scanning **every**
object in the scene and taking each one's world AABB. 2277 `_aabb_of` calls per
three macro steps, and `entity_prim.aabb` rebuilds collision points in world
frame on every one of them, 0.32 s of it inside `torch.tensor` alone.

**The fix** is `coop2/behavior_env/geometry_cache.py`: world AABBs memoised for
the length of one tick, since nothing can move while nothing steps. The engine
clears it after every `env.step`, `reset` clears it after placement, and the four
places that teleport a body between ticks clear it too. `_aabb_of` calls per three
macro steps went from 2277 to 78. It lives in its own module, free of any
omnigibson import, so the engine can clear the cache without pulling in the
controller stack, which the stubbed CPU tests replace wholesale.

The other two changes were kept, being correct and cheap: `_facts_scoped` now
keys its cache by entity set instead of holding only the last one, and
`observation_for` takes the entity mapping as an argument so one macro step
scans the scene once for all nine agents.

**What was reverted.** Sharing the per-entity radius cache across agents. The
radius is `clearance + reach`, and both the reach and the robot radius differ per
robot, so one cache across agents would have handed a Crazyflie the Ridgeback's
reach gate. The AABB cache underneath removes the cost anyway.

## 4. What is left, in order of payoff

**Run cells in parallel, with the thread cap.** `sweep_grid --parallel K`. The
sweep that prompted all this passed no `--parallel`, so 48 cells ran one at a
time while the card sat at 11% of its memory and 0% utilisation and 32 cores
were mostly idle. Done together with section 1's cap, this is the dominant
remaining lever, since the simulator is no longer what the wall clock is made
of.

What limits K is not the GPU: six cells took 6.6 GB of the 16 GB card at 0%
utilisation. It is RAM, at about 5.3 GB per Isaac instance, so six leave little
of 59 GB free; and the shared LLM deployment, whose rate limits arrive K times
faster. Watch `total_api_rate_limit_retries` in `llm_usage.json`. Three cells
measured 0 retries and 0 errors here. Four is a comfortable default on this
box, six works with little headroom.

**Reduce LLM latency and calls.** Each call carries 4700 prompt tokens on average
and sections 1 to 5 of the system prompt barely change between calls, so prompt
caching should pay well. Separately, every avoidable failure costs a replan, which
is both a model call and the ticks that went into the abandoned plan.

**Reusing the Isaac process across cells.** 30 to 75 s per cell. Switching scene
and task inside one process means a reload anyway, so the gain is small and the
risk is not. Lowest priority.

**A note on `--time-limit-seconds`.** At 469 ms per step a cell reached about
1300 of the 6000 steps it was asked for before a 600 s wall-clock limit stopped
it, so mode comparisons on the larger layouts were partly measuring who got
furthest in ten minutes. At 47 ms per step the same limit is no longer binding
for nine robots. For `sets_5`, fifteen robots, it has not been measured.

## Reproducing any of this

The two benchmark scripts used here live in the session scratchpad rather than
the repo, since they hard-code one layout and one task: one loads the scene,
spreads the robots across rooms and times `world.step()` plus `_build_info()`
under each configuration; the other alternates torch thread counts over blocks
of 60 `env.step` calls. A third steps one loaded scene and prints its rate, so
running K copies of it prices contention between cells without the model's
variance in the way. `feasibility_verify/profile_tick.py` prices the per-tick
floor and is the right starting point for the next question of this kind. Run
any of them with no sweep running; they take a GPU and several cores.
