"""Completion rate against team size, one line per method.

Pools both scenarios and all four task configurations, so each point is the
mean of 2 scenarios x 4 tasks x 3 seeds = 24 episodes. Error bars are the
standard error over those 24, which includes the spread *between* tasks as
well as between seeds -- LL and HL are near-ceiling for everyone while LH is
not, so a bar here is wide by construction and should be read as "how mixed
the 24 were", not as seed noise alone.

``--task lh`` keeps one difficulty instead (user, 2026-09-15). Worth having
beside the pooled figure rather than instead of it: pooling averages the one
task that separates the methods together with three that do not, so the
pooled curve understates the gap; the filtered one shows it at a quarter of
the episodes per point, and its bars are correspondingly wider.

Usage:
    python coop2/experiment/plot_scaling.py \\
        --scenario S1 experiment_log/S1_all_1 experiment_log/S1_all_2 experiment_log/S1_all_3 \\
        --scenario S2 experiment_log/S2_all_1 experiment_log/S2_all_2 experiment_log/S2_all_3 \\
        --out coop2/figures/cr_vs_n

Needs matplotlib, so run it with the `behavior` interpreter.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cooperation_table", HERE / "cooperation_table.py")
_ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ct)

#: Drawn in this order, so the legend reads baseline-to-ours.
METHODS: Sequence[Tuple[str, str, str, str]] = (
    ("individual", "Individual", "#6E7B8B", "o"),
    ("centralized", "Centralized", "#2E75B6", "s"),
    ("broadcast_chain", "Chain", "#C55A11", "^"),
    ("tag", "Task Graph (Ours)", "#1F7A3D", "D"),
)

#: layout suffix -> robots. `sets_<k>` is k teams of three.
SIZES = {"sets_1": 3, "sets_3": 9, "sets_5": 15}


def episodes(roots: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for root in roots:
        for row in _ct.collect(root, latest_only=True):
            layout = str(row.get("layout") or "")
            size = next((n for key, n in SIZES.items() if layout.endswith(key)), None)
            if size is None or row.get("CR") is None:
                continue
            row["N"] = size
            rows.append(row)
    return rows


#: Type sizes, named so the panels cannot drift apart and so the two figures
#: agree: every tick number is one size, whatever the panel's magnitude
#: (user, 2026-09-15), and the title carries the weight. Both `cr` and `cost`
#: read from these -- they go in the same document, and two figures whose
#: axis numbers differ by a point read as two different documents.
TITLE_SIZE = 13
TICK_SIZE = 11
AXIS_LABEL_SIZE = 12
LEGEND_SIZE = 10

#: The cost panels: column, title, y label, scale, and the ceiling its
#: quantity cannot pass (None for none).
#:
#: The title's parentheses carry **our own column abbreviation** (user,
#: 2026-09-15) -- the same ML / IT / WT / TO that `cooperation_table.py`
#: prints and the paper's tables use -- so a reader moving between a table
#: and this figure does not have to match them up by prose. The unit moved
#: to the y label, where it was always supposed to be. The ceiling clips the shaded band only -- a band that
#: runs past 100% of episodes claims something impossible, and a reader takes
#: the shape of a band more literally than a whisker.
#:
#: What each number is summed over used to be drawn inside the panel in grey
#: (ML "count", IT "summed over teams", WT "summed over robots", TO "600 s wall
#: clock"). Removed (user, 2026-09-15): four greyed asides sat on top of the
#: lines they annotate, and the caption is where that belongs.
COST_PANELS = (
    ("ML", "Message load (ML)", "Count", 1.0, None),
    ("IT", "Interrupt time (IT)", "Seconds", 1.0, None),
    ("WT", "Wait time (WT)", "Seconds", 1.0, None),
    ("TO", "Episodes timed out (TO)", "% of episodes", 100.0, 100.0),
)


def draw_cost(rows, sizes, plt):
    """Four panels, 2x2: what the protocol costs, against team size.

    Messages alone understate it. A message interrupts every robot of every
    team it names, and the team it stops then has to decide again -- so the
    same protocol shows up three times, and the third (robots standing still)
    is the largest. The fourth panel is what the first three buy: the share of
    episodes that never finished because the 600 s wall clock ran out. It is
    the consequence the costs predict, and the sharpest contrast in the data --
    at N=9, Chain times out in 62% of episodes against Task Graph's 8%.

    Every method is 0 at N=3 by construction: one team has nobody to talk to.

    The spread is a shaded band of **+-1 standard error**, not a whisker
    (user, 2026-09-15). Same statistic a whisker carried; the band reads as a
    region the trend lives in, which is what it is, and four methods' whiskers
    at three x positions collided.

    It was a 95% t interval for one revision and was reverted (user): at n=24
    the band is 2.069 SE wide, and in the timeout panel that makes all four
    methods overlap at N=15 -- true, and it buries the N=9 contrast, which is
    the one this figure is for. So the band is +-1 SE and claims only what
    that is. **What it is over matters more than its width**: the n at each
    point is every episode there -- 2 scenarios x 4 tasks x 3 seeds -- so the
    variance is the spread *between tasks* as much as between seeds, LL and HL
    being near-ceiling for everyone while LH is not. Read it as "how mixed
    those 24 were", not as the seed noise of one condition, and do not read
    non-overlap as a test.
    """
    figure, grid = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True)
    axes = grid.flatten()
    for index, (panel, (column, label, unit, scale, ceiling)) in enumerate(zip(axes, COST_PANELS)):
        for mode, name, colour, marker in METHODS:
            means, lows, highs = [], [], []
            for size in sizes:
                values = [scale * r[column] for r in rows
                          if r["mode"] == mode and r["N"] == size and r.get(column) is not None]
                mean = statistics.fmean(values) if values else float("nan")
                error = (statistics.stdev(values) / len(values) ** 0.5) if len(values) > 1 else 0.0
                means.append(mean)
                # Clipped at the quantity's own bounds: none of these can be
                # negative, and a share of episodes cannot pass 100%. A band
                # drawn past a hard bound claims something impossible, and a
                # reader takes a band's shape more literally than a whisker's.
                lows.append(max(0.0, mean - error))
                highs.append(min(ceiling, mean + error) if ceiling is not None else mean + error)
            panel.fill_between(sizes, lows, highs, color=colour, alpha=0.16,
                               linewidth=0, zorder=2)
            panel.plot(sizes, means, label=name, color=colour, marker=marker,
                       markersize=5.0, linewidth=1.7, zorder=3)
            print(f"  {column:3} {name:20} " + "  ".join(
                f"N={s}: {m:7.1f}" for s, m in zip(sizes, means)))
        panel.set_title(label, fontsize=TITLE_SIZE, fontweight="bold")
        panel.set_ylabel(unit, fontsize=AXIS_LABEL_SIZE)
        panel.set_xticks(sizes)
        panel.set_ylim(bottom=0)
        # Both axes, every panel: sharex hides the top row's x labels but not
        # its size, so setting it per panel keeps one size across the figure.
        panel.tick_params(axis="both", labelsize=TICK_SIZE)
        if index >= 2:
            panel.set_xlabel("Robots ($N$)", fontsize=AXIS_LABEL_SIZE)
        panel.grid(axis="y", linewidth=0.5, alpha=0.35, zorder=0)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper left")
    figure.tight_layout()
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", action="append", nargs="+", required=True,
                        metavar=("NAME DIR", "DIR"),
                        help="a scenario label followed by its seed directories; repeatable")
    parser.add_argument("--out", type=Path, default=Path("cr_vs_n"),
                        help="output path without extension; writes .png and .pdf")
    parser.add_argument("--task", default=None, metavar="SUFFIX",
                        help="keep only tasks whose name ends in this, e.g. lh. Both "
                             "scenarios' LH pool together (v4_s1_v4_lh + v4_s2_v4_lh), "
                             "so a point is 2 scenarios x 3 seeds = 6 episodes rather "
                             "than 24. Affects the cr figure; the cost panels pool "
                             "every task by design")
    parser.add_argument("--figure", choices=("cr", "cost"), default="cr",
                        help="cr: completion rate against team size. cost: what the "
                             "coordination costs -- messages, interrupt time, idle time")
    parser.add_argument("--ylim", type=float, nargs=2, default=(0.6, 1.0),
                        metavar=("LO", "HI"),
                        help="y range (default 0.6 1.0). A truncated axis magnifies the "
                             "differences, so the break is drawn on the spine; pass 0 1.05 "
                             "for the untruncated version")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[dict] = []
    for block in args.scenario:
        name, roots = block[0], [Path(p) for p in block[1:]]
        if not roots:
            print(f"scenario {name} has no directories", file=sys.stderr)
            return 2
        for row in episodes(roots):
            row["scenario"] = name
            rows.append(row)
    if not rows:
        print("no episodes found", file=sys.stderr)
        return 1

    if args.task:
        # The four tasks are not one population -- LL and HL sit near the
        # ceiling for every method while LH does not -- so the pooled figure
        # averages across a difference rather than over noise. Filtering to one
        # difficulty is the honest way to show the spread that pooling hides,
        # at the price of a quarter of the episodes per point.
        suffix = args.task.lower().lstrip("_")
        rows = [r for r in rows if str(r.get("task") or "").lower().endswith("_" + suffix)]
        if not rows:
            print(f"no episodes for task suffix {suffix!r}", file=sys.stderr)
            return 1
        print(f"task filter {suffix!r}: {sorted({r['task'] for r in rows})}")

    scenarios = sorted({r["scenario"] for r in rows})
    sizes = sorted({r["N"] for r in rows})
    print(f"{len(rows)} episodes, scenarios {scenarios}, sizes {sizes}")

    if args.figure == "cost":
        figure = draw_cost(rows, sizes, plt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            figure.savefig(args.out.with_suffix(suffix), dpi=200)
            print(f"wrote {args.out.with_suffix(suffix)}")
        return 0

    # One cost panel's axes box is 2.805 x 1.955 in after its own tight_layout,
    # and this figsize puts the CR axes at 2.808 x 1.954 -- so at any shared
    # scaling the two figures' 13pt titles, 12pt labels and 11pt tick numbers
    # are the same size on the page (user, 2026-09-15). Matching the *canvas*
    # would not do it: a point is absolute, so the same 13pt on a wider figure
    # reads smaller once both are scaled to a column. Re-measure this pair
    # (scratchpad solve_cr.py) if either figure's labels or type sizes change.
    figure, axes = plt.subplots(figsize=(3.78, 2.98))
    for mode, label, colour, marker in METHODS:
        means, errors, lows, highs, counts = [], [], [], [], []
        for size in sizes:
            values = [r["CR"] for r in rows if r["mode"] == mode and r["N"] == size]
            mean = statistics.fmean(values) if values else float("nan")
            error = (statistics.stdev(values) / len(values) ** 0.5) if len(values) > 1 else 0.0
            means.append(mean)
            errors.append(error)
            # A completion rate is a fraction of route nodes: it cannot pass 1,
            # and every method sits near that ceiling, so an unclipped band
            # would claim impossible values exactly where the figure is read.
            lows.append(max(0.0, mean - error))
            highs.append(min(1.0, mean + error))
            counts.append(len(values))
        # A band of +-1 SE, as in the cost figure -- same statistic a whisker
        # carried, and four methods' whiskers at three x positions collided.
        axes.fill_between(sizes, lows, highs, color=colour, alpha=0.16,
                          linewidth=0, zorder=2)
        axes.plot(sizes, means, label=label, color=colour, marker=marker,
                  markersize=5.5, linewidth=1.8, zorder=3)
        print(f"  {label:20} " + "  ".join(
            f"N={s}: {m:.2f}+-{e:.2f} (n={c})" for s, m, e, c in zip(sizes, means, errors, counts)))

    # The filtered figure has to say so in the title: the two are the same
    # axes with a quarter of the episodes, and side by side they are otherwise
    # indistinguishable while telling different stories.
    axes.set_title("Completion rate (CR)" + (f" -- {args.task.upper()} only" if args.task else ""),
                   fontsize=TITLE_SIZE, fontweight="bold")
    axes.set_xlabel("Robots ($N$)", fontsize=AXIS_LABEL_SIZE)
    axes.set_ylabel("Fraction of route nodes", fontsize=AXIS_LABEL_SIZE)
    axes.set_xticks(sizes)
    axes.tick_params(axis="both", labelsize=TICK_SIZE)
    low, high = args.ylim
    axes.set_ylim(low, high)
    if low > 0:
        # Mark the break on the spine. A truncated axis magnifies every gap,
        # and a reader who does not notice the origin reads these as several
        # times larger than they are; the caption alone is too easy to skip.
        for offset in (-0.012, 0.012):
            axes.plot([-0.018, 0.018], [offset, offset + 0.022],
                      transform=axes.transAxes, color="black",
                      linewidth=1.0, clip_on=False, zorder=5)
    axes.grid(axis="y", linewidth=0.5, alpha=0.35, zorder=0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(frameon=False, fontsize=LEGEND_SIZE, loc="lower left")
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(args.out.with_suffix(suffix), dpi=200)
        print(f"wrote {args.out.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
