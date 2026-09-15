"""Sweep layouts x tasks x cooperation modes, one episode per cell, in sequence.

    OMNIGIBSON_HEADLESS=1 python -m coop2.experiment.sweep_grid \\
        --layouts coop2/team_layouts/s1/sets_1.json coop2/team_layouts/s1/sets_3.json \\
        --tasks v4_s1_v4_ll v4_s1_v4_hh \\
        --modes individual centralized \\
        --steps 4000 --output-root experiment_log/sweep_0913

Every cell runs as its own process (Isaac owns the GPU, one env per process),
in its own folder ``<output-root>/<mode>_<layout>_<task>_<stamp>/`` with the
runner's usual files plus ``stdout.log``. ``sweep_summary.json`` at the root is
rewritten after every cell, so a partial sweep is always readable, and
``--resume`` skips cells that already have an ``ok`` row there.

Scene and room per task are looked up, not typed: the scene is whichever
``datasets/2026-challenge-task-instances/scenes/<scene>/json/`` holds the task's
cached instance, and the room is the shared-spawn room the scene's layouts use
(S1 living_room_0, S2 living_room_1, S3 bedroom_0). Override either with
``--scene`` / ``--room`` when running one scene. A layout whose folder or name
says ``s2`` is not run against an ``s1`` task unless ``--allow-mismatch``.

Two more axes: ``--models`` (one run per model) and repeats, either
``--seeds 0 1 2`` or ``--repeats 3`` (seeds ``--seed`` .. ``--seed``+2). The run
name carries both: ``<mode>_<layout>_<task>_<model>_seed<N>_<stamp>``.

``--parallel K`` runs K cells at once. Each cell is its own process regardless
(``og.sim`` is a process singleton, one environment per process); K just means
K Isaac instances on the one GPU. The GPU is not what limits K: measured
2026-09-14, nine robots per cell, six cells took 6.6 GB of the 16 GB card at
0% utilisation. **RAM and CPU are.** Each Isaac is about 5.3 GB resident, so
six leave little of 59 GB free, and each opens a PyTorch intra-op pool of one
thread per core which spins at its barrier -- so ``--threads-per-cell``
(default 1) caps it. Aggregate steps/s across all cells, stepping idle:

    1 cell,  16 threads   60.6
    3 cells, 16 threads   16.5   <- three cells slower in total than one
    3 cells,  1 thread   133.8
    6 cells,  1 thread   242.0

Every cell also talks to the same LLM deployment, so rate limits arrive K
times faster; watch ``total_api_rate_limit_retries`` in ``llm_usage.json``.
4 is a good default here, 6 works with little RAM headroom.

Globs work for layouts: ``--layouts 'coop2/team_layouts/s1/sets_*.json'``.
Nothing here touches the runners; it only builds their command lines.

Modes: the three messaging modes and ``tag`` (DIG-TAG's shared task graph)
run by default; ``board`` -- DIG-TAG's ablation -- on request. ``--notify-budget``
is passed only to the runners that take it (``tag``, and the board), and a
``tag`` cell's summary row carries what its graph did (``tag``: rounds,
actions applied / rejected, notifications sent / dropped) beside the route
tracker's verdict.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = ROOT / "datasets" / "2026-challenge-task-instances" / "scenes"

RUNNERS = {
    "individual": "coop2.experiment.run_individual",
    "broadcast_chain": "coop2.experiment.run_broadcast_chain",
    "centralized": "coop2.experiment.run_centralized",
    "board": "coop2.experiment.run_board",
    "tag": "coop2.experiment.run_tag",
}

#: Where each scene's shared-spawn layouts put the team (their own _comment).
DEFAULT_ROOM = {
    "Merom_1_int": "living_room_0",
    "Beechwood_0_int": "living_room_1",
    "Beechwood_1_int": "bedroom_0",
    "Pomaria_1_int": "living_room_0",
    "hall_glass_ceiling": "empty_room_0",
}

#: Which COOHAVIOR scenario a scene is, to check layout/task compatibility.
SCENARIO_OF_SCENE = {"Merom_1_int": "s1", "Beechwood_0_int": "s2", "Beechwood_1_int": "s3"}


def scene_for_task(task: str) -> str | None:
    """The scene whose instance cache holds @task, or None if never sampled."""
    hits = sorted(INSTANCES.glob(f"*/json/*_task_{task}_0_0_template.json"))
    return hits[0].parent.parent.name if hits else None


def scenario_of_layout(path: str) -> str | None:
    """``s1`` for ``team_layouts/s1/sets_3.json`` or ``v4_s1_v4_ll.json``; None if unmarked."""
    p = Path(path)
    for token in (p.parent.name, p.stem):
        m = re.search(r"(?<![a-z0-9])(s[123])(?![a-z0-9])", token)
        if m:
            return m.group(1)
    return None


def layout_label(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}_{p.stem}" if p.parent.name in ("s1", "s2", "s3") else p.stem


def expand_layouts(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for m in matches:
            if m not in out:
                out.append(m)
    missing = [m for m in out if not os.path.exists(m)]
    if missing:
        sys.exit(f"layout file(s) not found: {missing}")
    return out


def build_command(args, mode: str, layout: str, task: str, scene: str, room: str,
                  model: str, seed: int, run_name: str) -> list[str]:
    command = [
        sys.executable, "-u", "-m", RUNNERS[mode],
        "--team-config", layout,
        "--scene", scene, "--room", room,
        "--bddl-activity", task,
        "--steps", str(args.steps),
        "--seed", str(seed),
        "--time-limit-seconds", str(args.time_limit_seconds),
        "--model", model,
        "--output-root", str(args.output_root),
        "--run-name", run_name,
        "--llm-quiet",
    ]
    if not args.video:
        command.append("--no-video")
    # Only the runners with a notify tool know the flag; the others would
    # refuse their command line over it.
    if mode in NOTIFYING_MODES:
        command += ["--notify-budget", str(args.notify_budget)]
    command += args.runner_arg or []
    return command


#: The modes whose runner takes --notify-budget.
NOTIFYING_MODES = ("tag", "board")


def tag_summary(run_dir: Path):
    """What a tag cell's task graph did, from tag_rounds.json, if written."""
    f = run_dir / "tag_rounds.json"
    if not f.exists():
        return None
    try:
        rounds = json.loads(f.read_text())
        actions = [a for r in rounds for a in r.get("tag_actions", [])]
        return {
            "rounds": len(rounds),
            "actions_applied": sum(1 for a in actions if a.get("result") == "applied"),
            "actions_rejected": sum(1 for a in actions if str(a.get("result", "")).startswith("rejected")),
            "notify_sent": sum(1 for r in rounds if r.get("notify", {}).get("sent")),
            "notify_dropped": sum(1 for r in rounds if r.get("notify", {}).get("dropped")),
        }
    except Exception:  # noqa: BLE001
        return "unreadable"


def route_progress(run_dir: Path):
    """The route tracker's verdict, if the run got far enough to write one."""
    f = run_dir / "route_progress.json"
    if not f.exists():
        return None
    try:
        s = json.loads(f.read_text()).get("summary", {})
        return {k: s.get(k) for k in ("completed_nodes", "required_nodes", "Y_task", "task_complete")}
    except Exception:  # noqa: BLE001
        return "unreadable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="+", required=True, metavar="PATH_OR_GLOB",
                        help="team layout JSONs; globs allowed")
    parser.add_argument("--tasks", nargs="+", required=True, metavar="ACTIVITY",
                        help="BDDL activity names, e.g. v4_s1_v4_ll")
    parser.add_argument("--modes", nargs="+", default=["individual", "broadcast_chain", "centralized", "tag"],
                        choices=sorted(RUNNERS))
    parser.add_argument("--notify-budget", type=int, default=1, metavar="N",
                        help="for tag (and the board): notifications a team may send between two env steps")
    parser.add_argument("--steps", type=int, default=8000, help="max env steps per episode")
    parser.add_argument("--output-root", default=str(ROOT / "experiment_log"),
                        help="where each run's folder and sweep_summary.json go")
    parser.add_argument("--seed", type=int, default=0, help="first seed (see --repeats / --seeds)")
    parser.add_argument("--repeats", type=int, default=1, help="run each cell with seeds seed..seed+repeats-1")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="explicit seed list; overrides --repeats")
    parser.add_argument("--parallel", type=int, default=1, metavar="K",
                        help="cells to run at once, each its own Isaac process; see the docstring before going above 2")
    parser.add_argument("--time-limit-seconds", type=float, default=0,
                        help="wall-clock cap per episode; 0 = --steps alone decides")
    parser.add_argument("--models", nargs="+", default=["gpt-5.6-luna"], help="one run per model")
    parser.add_argument("--scene", default=None, help="override the per-task scene lookup")
    parser.add_argument("--room", default=None, help="override the per-scene default room")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="also run layouts whose scenario (s1/s2/s3) differs from the task's scene")
    parser.add_argument("--order", choices=["layout", "task", "mode", "model", "seed"], default="layout",
                        help="outermost loop; default runs all tasks x modes of one layout before the next")
    parser.add_argument("--video", action="store_true", help="record per-robot videos (off by default)")
    parser.add_argument("--run-timeout-minutes", type=float, default=120,
                        help="kill a cell that has not finished by then and move on")
    parser.add_argument("--threads-per-cell", type=int, default=1, metavar="N",
                        help="OMP_NUM_THREADS for each cell when --parallel > 1 (default 1). "
                             "0 leaves it unset, which lets every cell open a pool per core "
                             "and makes three cells slower in total than one alone.")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells that already have an 'ok' row in sweep_summary.json")
    parser.add_argument("--runner-arg", action="append", metavar="ARG",
                        help="extra flag passed through to every runner verbatim (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = parser.parse_args()

    layouts = expand_layouts(args.layouts)
    output_root = Path(args.output_root)
    summary_path = output_root / "sweep_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = ({(r["mode"], r["layout"], r["task"], r.get("model"), r.get("seed")) for r in summary if r.get("status") == "ok"}
            if args.resume else set())

    # Resolve scene/room per task once, and refuse to start with an unknown task.
    scene_of: dict[str, str] = {}
    for task in args.tasks:
        scene = args.scene or scene_for_task(task)
        if scene is None:
            sys.exit(f"{task}: no cached instance under {INSTANCES} -- sample it first, or pass --scene")
        scene_of[task] = scene
    room_of = {t: (args.room or DEFAULT_ROOM.get(s)) for t, s in scene_of.items()}
    for task, room in room_of.items():
        if room is None:
            sys.exit(f"{task}: no default room for scene {scene_of[task]}; pass --room")

    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + max(1, args.repeats)))
    if args.parallel < 1:
        sys.exit("--parallel must be >= 1")

    # The grid, with layout/task scenario mismatches dropped unless asked for.
    cells, skipped = [], []
    axes = {"layout": layouts, "task": args.tasks, "mode": args.modes, "model": args.models, "seed": seeds}
    order = [args.order] + [a for a in ("layout", "task", "mode", "model", "seed") if a != args.order]
    import itertools
    for combo in itertools.product(*(axes[a] for a in order)):
        cell = dict(zip(order, combo))
        layout, task = cell["layout"], cell["task"]
        want = SCENARIO_OF_SCENE.get(scene_of[task])
        have = scenario_of_layout(layout)
        if want and have and want != have and not args.allow_mismatch:
            if (layout_label(layout), task) not in skipped:
                skipped.append((layout_label(layout), task))
            continue
        cells.append((cell["mode"], layout, task, cell["model"], cell["seed"]))

    print(f"{len(cells)} cells: {len(layouts)} layout(s) x {len(args.tasks)} task(s) x {len(args.modes)} mode(s)"
          f" x {len(args.models)} model(s) x {len(seeds)} seed(s)"
          + (f", {len(skipped)} layout/task scenario mismatches skipped" if skipped else "")
          + f"; {args.steps} steps each; {args.parallel} at a time; output {output_root}")
    for label, task in skipped[:6]:
        print(f"  skipped {label} x {task} (different scenario; --allow-mismatch to run)")

    output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1")
    # One BLAS thread per cell when cells share the machine. PyTorch opens an
    # intra-op pool of one-per-core (16 here) and it spins at its barrier, so
    # K cells oversubscribe every core K times over. Measured 2026-09-14 on
    # nine robots stepping idle, aggregate steps/s across all cells:
    #
    #   1 cell,  16 threads   60.6      (16.4 ms/step)
    #   3 cells, 16 threads   16.5      (163-204 ms/step)  -- worse than one
    #   3 cells,  1 thread   133.8      (22.5 ms/step)
    #   6 cells,  1 thread   242.0      (23-26 ms/step)
    #
    # A single cell is ~7% slower on one thread, which is why this is not set
    # for --parallel 1; past that the pool costs an order of magnitude.
    if args.parallel > 1 and args.threads_per_cell:
        env["OMP_NUM_THREADS"] = str(args.threads_per_cell)
    lock = threading.Lock()
    total = len(cells)

    def run_cell(index, mode, layout, task, model, seed):
        label = layout_label(layout)
        if (mode, label, task, model, seed) in done:
            print(f"[{index}/{total}] {mode} {label} {task} {model} seed{seed}: already ok, skipped (--resume)", flush=True)
            return
        scene, room = scene_of[task], room_of[task]
        run_name = f"{mode}_{label}_{task}_{model}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        command = build_command(args, mode, layout, task, scene, room, model, seed, run_name)
        print(f"\n[{index}/{total}] {run_name}  ({scene}/{room})\n  {' '.join(command)}", flush=True)
        if args.dry_run:
            return
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        started, status, code = time.time(), "ok", None
        with open(run_dir / "stdout.log", "w") as log:
            try:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.run_timeout_minutes * 60).returncode
                if code != 0:
                    status = f"exit {code}"
            except subprocess.TimeoutExpired:
                status = f"timeout after {args.run_timeout_minutes:g} min"
        # Isaac can swallow a traceback and still exit 0: an ok row needs the
        # runner's own end-of-run files, not just a clean return code.
        if status == "ok" and not (run_dir / "llm_usage.json").exists():
            status = "exit 0 but no llm_usage.json (crashed inside Isaac?)"
        row = {
            "run": run_name, "mode": mode, "layout": label, "task": task, "model": model, "seed": seed,
            "scene": scene, "room": room, "steps": args.steps, "status": status, "returncode": code,
            "minutes": round((time.time() - started) / 60, 1), "route_progress": route_progress(run_dir),
        }
        if mode == "tag":
            row["tag"] = tag_summary(run_dir)
        with lock:
            summary.append(row)
            summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[{index}/{total}] -> {status} in {row['minutes']} min  ({run_name})", flush=True)

    if args.parallel == 1 or args.dry_run:
        for index, cell in enumerate(cells, 1):
            run_cell(index, *cell)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run_cell, index, *cell) for index, cell in enumerate(cells, 1)]
            for future in futures:
                future.result()

    if not args.dry_run:
        print(f"\nsummary: {summary_path}")
        for row in summary:
            print(f"  {row['status'][:14]:>14}  {row['minutes']:>6} min  {row['run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
