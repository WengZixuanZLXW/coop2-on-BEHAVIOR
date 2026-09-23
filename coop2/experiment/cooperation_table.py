"""The cooperation results table for a sweep directory.

Reads a folder of run directories -- what ``sweep_grid --output-root`` writes,
e.g. ``experiment_log/S1_all`` -- and prints one row per cell:

    CR  completion rate: the route's own progress, completed_nodes /
        required_nodes. This is what the activity asks for, in order; it is
        not check_goal, which cannot express "in this order" and is only a
        cross-check.
    ES  the environment steps the episode used. The episode ends when the
        route completes, so for a finished run this is the completion step and
        for a failure it is how far it got -- both are reported (user,
        2026-09-15). ES alone does not say which happened; read it with CR.
    WT  wait time: seconds each robot spent in `waiting`, summed per team and
        then over teams.
    IT  interrupt time: seconds each team spent in its `interrupted` stage,
        summed per team and then over teams.
    ML  message load: how many messages **teams** received -- every message
        counted once per recipient team, so one notification naming two teams
        is two, whatever each team's size.

        Counted per team rather than per robot (user, 2026-09-16). A message
        is addressed to a team and answered by a team: the brain reads it
        once and replies once, so expanding it to each robot it was delivered
        to multiplies by a number that has nothing to do with how much
        communication happened. It also made the mean look artificially
        exact -- with three robots a team and three seeds, the factor three
        cancelled the division and every average landed on a whole number.
    IC  interrupt count: how many times each team was interrupted, summed.
    RT  reasoning time: seconds each team spent in its `planning` stage,
        summed per team and then over teams.
    TT  thinking time, RT + IT.
    TO  whether the episode ended on the wall clock rather than by finishing
        the route or exhausting the step budget -- 0/1 per episode, so the
        mean over seeds is the timeout rate. It is the consequence the other
        cost columns predict, and the one that ends runs.
    FZ  the share of the episode's wall clock during which the environment was
        not stepping. Not TT/(wall clock): the plan loop stops the world while
        *any* team is reasoning or interrupted, so what is frozen is the
        **union** of the spans, and teams overlap. TT double-counts that
        overlap on purpose -- it is per-team cost -- while FZ does not.

WT is per robot because `waiting` is a robot's state (it has finished its plan
and the team has not been asked again); RT, IT and IC are per team because one
LLM drives a whole team and those are stages of that one call.

Usage:
    python3 coop2/experiment/cooperation_table.py experiment_log/S1_all
    python3 coop2/experiment/cooperation_table.py experiment_log/S1_all --csv out.csv
    python3 coop2/experiment/cooperation_table.py experiment_log/S1_all --markdown

Standard library only, so it runs without the `behavior` env and touches no
GPU: every number comes from the JSON a finished run already wrote.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: A run directory is ``<mode>_<layout>_<task>_<model>_seed<n>_<stamp>``. Used
#: only when the sweep summary does not name the cell (a directory copied in by
#: hand, or a single ``run_*`` invocation).
MODES = ("individual", "centralized", "broadcast_chain", "decentralized_messageboard", "tag")
_DIRNAME = re.compile(
    r"^(?P<mode>" + "|".join(MODES) + r")_(?P<rest>.+)_seed(?P<seed>\d+)_\d{8}_\d{6}_\d+$"
)

#: Columns, in the order they are printed. The first four identify the cell.
COLUMNS = ["mode", "layout", "task", "seed", "teams",
           "CR", "ES", "WT", "IT", "ML", "IC", "RT", "TT", "FZ", "TO"]


def _load(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def union_seconds(spans: Sequence[Tuple[float, float]]) -> float:
    """Total length of @spans with overlap counted once."""
    ordered = sorted((float(a), float(b)) for a, b in spans if b is not None and a is not None)
    total = 0.0
    current_start: Optional[float] = None
    current_end = 0.0
    for start, end in ordered:
        if end <= start:
            continue
        if current_start is None or start > current_end:
            if current_start is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def state_seconds(states: Dict[str, List[Sequence[Any]]], wanted: str, end_of_run: float) -> float:
    """Seconds every agent spent in @wanted, summed over agents.

    ``agent_states.json`` is a list of ``[wall_clock, env_step, state]`` marks
    per agent: a mark says when a state *began*, so its length runs to the next
    mark, and the last one runs to the end of the episode.
    """
    total = 0.0
    for marks in states.values():
        for index, mark in enumerate(marks):
            if len(mark) < 3 or mark[2] != wanted:
                continue
            start = float(mark[0])
            end = float(marks[index + 1][0]) if index + 1 < len(marks) else end_of_run
            if end > start:
                total += end - start
    return total


def episode_end(states: Dict[str, List[Sequence[Any]]], spans: Dict[str, List[Dict]]) -> float:
    """The episode's wall clock: the last thing either record saw."""
    last = 0.0
    for marks in states.values():
        if marks:
            last = max(last, float(marks[-1][0]))
    for team_spans in spans.values():
        for span in team_spans:
            if span.get("end") is not None:
                last = max(last, float(span["end"]))
    return last


def read_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """One row, or None when the directory is not a finished run."""
    route = _load(run_dir / "route_progress.json")
    timeline = _load(run_dir / "team_timeline.json") or {}
    states = _load(run_dir / "agent_states.json") or {}
    messages = _load(run_dir / "message_log.json") or []
    if route is None and not states:
        return None

    spans: Dict[str, List[Dict]] = (timeline.get("spans") or {})
    end_of_run = episode_end(states, spans)

    reasoning = interrupt = 0.0
    interrupt_count = 0
    for team_spans in spans.values():
        for span in team_spans:
            start, end = span.get("start"), span.get("end")
            if start is None or end is None:
                continue
            length = float(end) - float(start)
            if span.get("kind") == "planning":
                reasoning += length
            elif span.get("kind") == "interrupted":
                interrupt += length
                interrupt_count += 1

    frozen = union_seconds([
        (span["start"], span["end"])
        for team_spans in spans.values() for span in team_spans
        if span.get("start") is not None and span.get("end") is not None
    ])

    teams = len(timeline.get("teams") or {}) or None

    summary = (route or {}).get("summary") or {}
    required = summary.get("required_nodes") or 0
    completed = summary.get("completed_nodes") or 0
    complete = bool(summary.get("task_complete"))
    events = (route or {}).get("events") or []
    metrics = _load(run_dir / "coop2_metrics.json") or {}
    # The environment steps the episode used, whether it finished or not (user,
    # 2026-09-15). The episode ends when the route completes, so for a finished
    # run this *is* the completion step; for a failure it is how far it got,
    # which is a real number and was previously blanked. Read it with CR: ES
    # alone does not say which of the two ended the episode.
    steps = ((metrics.get("decision_overhead") or {}).get("T")
             or (int(events[-1]["env_step"]) if events else None))
    finish_step = int(events[-1]["env_step"]) if (complete and events) else None

    # `recipients` is in team names, `delivered_to` in robot names; the first
    # is the unit a message is addressed and answered in.
    received = sum(len(m.get("recipients") or []) for m in messages) if isinstance(messages, list) else 0

    # Whether the episode ended on the wall clock rather than on the route or
    # the step budget. Carried as 0/1 so averaging seeds gives the rate.
    usage = _load(run_dir / "llm_usage.json") or {}
    timed_out = 1.0 if usage.get("timed_out") else 0.0

    return {
        "run": run_dir.name,
        "teams": teams,
        "CR": (completed / required) if required else None,
        "ES": steps,
        "finished_at": finish_step,
        "WT": state_seconds(states, "waiting", end_of_run),
        "IT": interrupt,
        "ML": received,
        "IC": interrupt_count,
        "RT": reasoning,
        "TT": reasoning + interrupt,
        "TO": timed_out,
        "FZ": (frozen / end_of_run) if end_of_run > 0 else None,
    }


def label(run_dir: Path, from_summary: Dict[str, Dict]) -> Dict[str, Any]:
    """mode / layout / task / seed for @run_dir, from the sweep summary if it
    named this cell, else from the directory name."""
    known = from_summary.get(run_dir.name)
    if known:
        return {k: known.get(k) for k in ("mode", "layout", "task", "seed")}
    match = _DIRNAME.match(run_dir.name)
    if not match:
        return {"mode": None, "layout": None, "task": None, "seed": None}
    rest = match.group("rest")
    # `<layout>_<task>_<model>`, and the task is the only part starting `v4_`.
    at = rest.find("_v4_")
    layout = rest[:at] if at > 0 else None
    tail = rest[at + 1:] if at > 0 else rest
    task, _, _model = tail.partition("_gpt") if "_gpt" in tail else (tail.rpartition("_")[0], "", "")
    return {"mode": match.group("mode"), "layout": layout,
            "task": task or None, "seed": int(match.group("seed"))}


def collect(root: Path, latest_only: bool) -> List[Dict[str, Any]]:
    summary = _load(root / "sweep_summary.json")
    by_name = {row["run"]: row for row in summary if isinstance(row, dict) and "run" in row} \
        if isinstance(summary, list) else {}

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        measured = read_run(run_dir)
        if measured is None:
            # A directory with no route record and no agent states is a cell
            # that died before it could write one -- in `S1_all`, three whose
            # Isaac shut down 26 s in, which the sweep then re-ran. Say so
            # rather than skipping quietly: a crash that leaves a directory
            # behind looks exactly like a cell that was never asked for, and
            # Isaac exits 0 either way.
            skipped.append(run_dir.name)
            continue
        rows.append({**label(run_dir, by_name), **measured})
    if skipped:
        print(f"skipped {len(skipped)} directory(ies) with no run record "
              f"(started and died before writing one):", file=sys.stderr)
        for name in skipped:
            print(f"  {name}", file=sys.stderr)

    if latest_only:
        # A re-run writes a second directory for the same cell; the sweep keeps
        # both. Directory names end in a timestamp, so the last one sorts last.
        newest: Dict[Tuple, Dict[str, Any]] = {}
        for row in rows:
            newest[(row["mode"], row["layout"], row["task"], row["seed"])] = row
        rows = list(newest.values())

    rows.sort(key=lambda r: (str(r["task"]), str(r["layout"]), str(r["mode"])))
    return rows


#: Averaged over seeds, not summed: every metric here is per episode.
AVERAGED = ("CR", "ES", "WT", "IT", "ML", "IC", "RT", "TT", "FZ", "TO")


def average_seeds(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (mode, layout, task), the mean over the seeds that ran.

    `seeds` says how many went into it, so a cell that lost an episode is
    visible rather than quietly averaging fewer. `teams` comes along unaveraged
    -- it is a property of the layout, the same in every seed.
    """
    grouped: Dict[Tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["layout"], row["task"]), []).append(row)

    averaged: List[Dict[str, Any]] = []
    for (mode, layout, task), group in grouped.items():
        out: Dict[str, Any] = {"mode": mode, "layout": layout, "task": task,
                               "seed": "mean", "seeds": len(group),
                               "teams": group[0].get("teams")}
        for column in AVERAGED:
            values = [r[column] for r in group if r.get(column) is not None]
            out[column] = (sum(values) / len(values)) if values else None
        averaged.append(out)
    averaged.sort(key=lambda r: (str(r["task"]), str(r["layout"]), str(r["mode"])))
    return averaged


#: The LaTeX block: which method each row is, and which layout each N is.
LATEX_METHODS = [("Individual", "individual"), ("Centralized", "centralized"),
                 ("Chain", "broadcast_chain"), ("Task Graph (Ours)", "tag")]
#: Team-only metrics. With one team there is nobody to wait for, nobody to
#: interrupt you and nobody to message, so a 0 here would read as "efficient"
#: when it means "undefined" -- and WT, which is measurable, is then measuring
#: the gap between being handed a plan and starting it, not waiting for anyone.
#: All three are struck out at one team (user, 2026-09-15).
TEAM_ONLY = ("WT", "IT", "ML")

#: The five-then-six columns of one task block, in printing order. ES is the
#: episode's environment steps whether or not it completed (user, 2026-09-15):
#: a failure's step count is a real number and blanking it hid how far the run
#: got. Read it with CR beside it -- ES alone does not say whether the episode
#: ended by finishing or by running out.
LATEX_COLUMNS = ("CR", "ES", "WT", "IT", "ML")


def latex_block(rows: List[Dict[str, Any]], scenario: str, tasks: Sequence[str],
                sizes: Sequence[Tuple[str, str]]) -> str:
    """One scenario's rows of `tab:results-by-setting`, ready to paste."""
    by = {(r["mode"], r["layout"], r["task"]): r for r in rows}

    def cells(row: Optional[Dict[str, Any]]) -> List[str]:
        if row is None:
            return ["--"] * len(LATEX_COLUMNS)
        alone = (row.get("teams") or 0) <= 1
        out: List[str] = []
        for column in LATEX_COLUMNS:
            value = row.get(column)
            if value is None or (alone and column in TEAM_ONLY):
                out.append("--")
            elif column == "FZ":
                out.append(f"{100 * value:.1f}")
            else:
                # One decimal on every metric (user, 2026-09-15). Uniform
                # rather than per-metric: a table whose columns carry
                # different precisions invites reading the differences as
                # meaningful, and each of these is a mean of six episodes.
                out.append(f"{value:.1f}")
        return out

    # Team size outside, method inside (user, 2026-09-15): the comparison the
    # table is for is between methods at one size, so they should be adjacent
    # rows. The header is `& \(N\) & Method` to match.
    width = len(LATEX_METHODS) * len(sizes)
    # Three label columns, then one block of LATEX_COLUMNS per task -- so the
    # rule under a size block, and the table's own column spec, follow from
    # how many metrics there are rather than being written out.
    last = 3 + len(tasks) * len(LATEX_COLUMNS)
    lines = [r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{%s}}" % (width, scenario)]
    for index, (count, layout) in enumerate(sizes):
        if index:
            lines.append(r"\cmidrule(l){2-%d}" % last)
        for position, (label, mode) in enumerate(LATEX_METHODS):
            head = f"& {count} & {label}" if position == 0 else f"& & {label}"
            body: List[str] = []
            for task in tasks:
                body += cells(by.get((mode, layout, task)))
            lines.append(f"{head:26} & " + " & ".join(f"{c:>5}" for c in body) + r" \\")
    return "\n".join(lines)


def _cell(row: Dict[str, Any], column: str) -> str:
    value = row.get(column)
    if value is None:
        return "-"
    if column == "CR":
        return f"{100 * value:.0f}%"
    if column in ("FZ", "TO"):
        return f"{100 * value:.0f}%"
    if column == "ML":
        # One decimal: a team-level load is single digits for the quieter
        # modes, where rounding to units hides most of the difference.
        return f"{value:.1f}"
    if column in ("WT", "IT", "RT", "TT", "ES", "IC"):
        return f"{value:.0f}"
    return str(value)


def render(rows: List[Dict[str, Any]], markdown: bool) -> str:
    widths = {c: max(len(c), *(len(_cell(r, c)) for r in rows)) if rows else len(c)
              for c in COLUMNS}
    sep = " | " if markdown else "  "
    edge = "| " if markdown else ""
    tail = " |" if markdown else ""
    lines = [edge + sep.join(c.ljust(widths[c]) for c in COLUMNS) + tail]
    if markdown:
        lines.append("|" + "|".join("-" * (widths[c] + 2) for c in COLUMNS) + "|")
    for row in rows:
        lines.append(edge + sep.join(_cell(row, c).ljust(widths[c]) for c in COLUMNS) + tail)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", type=Path, nargs="+",
                        help="sweep output directories; give one per seed and the cells are "
                             "averaged, e.g. experiment_log/S1_all_1 experiment_log/S1_all_2")
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    parser.add_argument("--markdown", action="store_true", help="pipe-delimited, for pasting")
    parser.add_argument("--mean", action="store_true",
                        help="average the seeds even when given one directory")
    parser.add_argument("--all-runs", action="store_true",
                        help="keep every directory; by default a re-run cell keeps its latest")
    parser.add_argument("--latex", metavar="SCENARIO", default=None,
                        help="emit this scenario's rows of tab:results-by-setting instead "
                             "of the plain table, e.g. --latex S1")
    parser.add_argument("--latex-tasks", nargs="+", default=None,
                        help="task order for --latex; default is the ll/lh/hl/hh of the "
                             "scenario's own prefix")
    parser.add_argument("--latex-sizes", nargs="+", default=None,
                        metavar="N=LAYOUT",
                        help="rows for --latex, e.g. 3=s1_sets_1 9=s1_sets_3 15=s1_sets_5")
    args = parser.parse_args(argv)

    for root in args.roots:
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2

    rows: List[Dict[str, Any]] = []
    for root in args.roots:
        rows.extend(collect(root, latest_only=not args.all_runs))
    if not rows:
        print(f"no runs found under {', '.join(str(r) for r in args.roots)}", file=sys.stderr)
        return 1

    per_seed = rows
    if len(args.roots) > 1 or args.mean:
        rows = average_seeds(rows)
        short = {row["seeds"] for row in rows}
        if short != {len(args.roots)}:
            counts = sorted(short)
            print(f"note: cells average {counts} episodes, not all {len(args.roots)} "
                  f"-- a cell short of episodes lost one", file=sys.stderr)

    if args.latex:
        prefix = args.latex.lower()
        tasks = args.latex_tasks or [f"v4_{prefix}_v4_{kind}" for kind in ("ll", "lh", "hl", "hh")]
        if args.latex_sizes:
            sizes = [tuple(pair.split("=", 1)) for pair in args.latex_sizes]
        else:
            sizes = [("3", f"{prefix}_sets_1"), ("9", f"{prefix}_sets_3"), ("15", f"{prefix}_sets_5")]
        print(latex_block(rows, args.latex, tasks, sizes))
        return 0

    print(render(rows, args.markdown))
    print(f"\n{len(rows)} cells from {', '.join(str(r) for r in args.roots)}"
          f" ({len(per_seed)} episodes)")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run", "seeds"] + COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
