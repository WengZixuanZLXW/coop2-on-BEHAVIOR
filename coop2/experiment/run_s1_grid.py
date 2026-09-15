"""Run every (cooperation mode x task) of one layout, one episode each, in sequence.

The user's grid (2026-09-13): the three messaging modes -- individual,
broadcast_chain, centralized -- on the four S1 tasks with the three-team S1
layout, twelve episodes, each in its own folder under ``experiment_log/``
named ``<mode>_<layout>_<task>_<YYYYmmdd_HHMMSS>``. Runs are sequential
because Isaac owns the GPU; a run that crashes or hangs is recorded and the
grid moves on. Each folder gets the runner's usual files plus ``stdout.log``
(the process's whole output) and the grid keeps ``grid_summary.json`` at the
root, one row per run, updated after every run so a partial grid is readable.

Usage:
    OMNIGIBSON_HEADLESS=1 python -m coop2.experiment.run_s1_grid
    ... --modes individual --tasks v4_s1_v4_ll --steps 300     # a smoke pass
    ... --dry-run                                               # print the commands
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNNERS = {
    "individual": "coop2.experiment.run_individual",
    "broadcast_chain": "coop2.experiment.run_broadcast_chain",
    "centralized": "coop2.experiment.run_centralized",
    "board": "coop2.experiment.run_board",
    "tag": "coop2.experiment.run_tag",
}

#: The three messaging modes, in the order the results table uses.
DEFAULT_MODES = ["individual", "broadcast_chain", "centralized"]
DEFAULT_TASKS = ["v4_s1_v4_ll", "v4_s1_v4_lh", "v4_s1_v4_hl", "v4_s1_v4_hh"]


def layout_label(path: str) -> str:
    """``coop2/team_layouts/s1/sets_3.json`` -> ``s1_sets_3``."""
    p = Path(path)
    return f"{p.parent.name}_{p.stem}"


def build_command(args, mode: str, task: str, run_name: str) -> list[str]:
    command = [
        sys.executable, "-u", "-m", RUNNERS[mode],
        "--team-config", args.layout,
        "--scene", args.scene, "--room", args.room,
        "--bddl-activity", task,
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--time-limit-seconds", str(args.time_limit_seconds),
        "--model", args.model,
        "--output-root", str(args.output_root),
        "--run-name", run_name,
        "--llm-quiet",
    ]
    if not args.video:
        command.append("--no-video")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=sorted(RUNNERS))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--layout", default="coop2/team_layouts/s1/sets_3.json")
    parser.add_argument("--scene", default="Merom_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit-seconds", type=float, default=0)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--output-root", default=str(ROOT / "experiment_log"))
    parser.add_argument("--video", action="store_true", help="Record per-robot videos (off by default)")
    parser.add_argument("--run-timeout-minutes", type=float, default=120,
                        help="Kill a run that has not finished by then and move on")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "grid_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    label = layout_label(args.layout)
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1")

    plan = [(mode, task) for mode in args.modes for task in args.tasks]
    print(f"{len(plan)} runs: {args.modes} x {args.tasks}, layout {label}, {args.steps} steps each")
    for index, (mode, task) in enumerate(plan, 1):
        run_name = f"{mode}_{label}_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = output_root / run_name
        command = build_command(args, mode, task, run_name)
        print(f"\n[{index}/{len(plan)}] {run_name}\n  {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        status = "ok"
        with open(run_dir / "stdout.log", "w") as log:
            try:
                completed = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.run_timeout_minutes * 60,
                )
                code = completed.returncode
                if code != 0:
                    status = f"exit {code}"
            except subprocess.TimeoutExpired:
                code, status = None, f"timeout after {args.run_timeout_minutes:g} min"
        elapsed = time.time() - started
        # The route tracker's own verdict, if the run got far enough to write
        # it: nodes credited out of nodes required, and whether it finished.
        progress = None
        route_file = run_dir / "route_progress.json"
        if route_file.exists():
            try:
                total = json.loads(route_file.read_text()).get("summary", {})
                progress = {k: total.get(k) for k in ("completed_nodes", "required_nodes", "Y_task", "task_complete")}
            except Exception:  # noqa: BLE001
                progress = "unreadable"
        row = {
            "run": run_name, "mode": mode, "task": task, "layout": label,
            "status": status, "returncode": code, "minutes": round(elapsed / 60, 1),
            "route_progress": progress,
        }
        summary.append(row)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  -> {status} in {row['minutes']} min", flush=True)

    if not args.dry_run:
        print(f"\nsummary: {summary_path}")
        for row in summary:
            print(f"  {row['status']:>8}  {row['minutes']:>6} min  {row['run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
