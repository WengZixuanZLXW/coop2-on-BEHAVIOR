"""Draw each agent's FSM state over wall-clock time, one lane per agent.

Replaces metrics_timeline.png, which plotted constraint counters this project
does not use.

Messages are overlaid as arrows from the sender's lane to each recipient's, at
the moment they were sent. Under a team topology the sender and the recipient
are *teams*, so each team gets a lane of its own above its robots and the arrow
runs between those lanes. Drawing it from a member's lane instead -- the team's
first robot standing in for all four -- claimed that robot sent something it had
no part in, and put the arrowhead on one recipient robot when the message
interrupts four. Content is deliberately not drawn --
the question this figure answers is *when* an agent talked and *to whom*, which
is what ties a frozen world (the I and R spans) to the thing that froze it. The
message clock and the state clock are both seconds since the run started, so
they share the x axis directly.

The x axis is wall clock, not env_step, and that is the point of the figure.
The plan loop does not step the environment while any agent is not ready, so an
agent in R or I freezes the whole world: those spans occupy real seconds while
env_step does not move at all. Plotted against env_step they would collapse to
zero width, hiding the one cost the figure exists to show.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["plot_agent_state_timeline", "save_team_timeline"]

#: FSM state -> colour. Deliberately loud for R and I: those are the spans that
#: stop every other agent.
STATE_COLOURS = {
    "reasoning": "#d1495b",    # R -- LLM call, world frozen
    "interrupted": "#edae49",  # I -- message arrived, world frozen
    "waiting": "#8d99ae",      # W -- ready, waiting for the others
    "executing": "#2a9d8f",    # X -- primitive advancing
}

#: Not a fifth state. A hold *is* state X -- the member is running a `wait`
#: primitive, which is how it keeps the world moving while its teammates
#: finish -- so agent_states.json says "executing" for it and cannot say more.
#: What separates the two is the plan's specification in plan_logs.json, which
#: is why this is a shading of executing and not an entry beside it.
#:
#: Worth separating because the two look identical and are not: agent_2 in
#: centralized_agents8_..._040821 spent 33 consecutive 60-tick holds from
#: env_step 474 to the end, 81 % of the episode, and every one of them drew as
#: a green bar saying it did something.
HOLDING_COLOUR = "#9dd6cf"

#: Messages are drawn in one colour on purpose: the arrow already carries the
#: direction, and colouring by metadata["type"] would compete with the state
#: colours for the reader's attention.
MESSAGE_COLOUR = "#22223b"

#: What a team lane shows. A team is either thinking -- which freezes the world
#: for everyone, so it takes the same loud colours as the member states that
#: freeze it -- or it is not, and its robots are acting on what it last decided.
TEAM_SPAN_COLOURS = {
    "planning": STATE_COLOURS["reasoning"],
    # The whole window from the message arriving to the decision being
    # published, the report composed in between included. "deciding" is the
    # older name for the decision call alone; kept so an existing run still
    # draws.
    "interrupted": STATE_COLOURS["interrupted"],
    "deciding": STATE_COLOURS["interrupted"],
    "waiting": STATE_COLOURS["waiting"],
}
TEAM_IDLE_COLOUR = "#e9ecef"


def _coalesce(spans, min_width: float):
    """Absorb sub-pixel spans into their neighbour, then merge like with like.

    A chain wave wakes agent_i once per upstream relay, and with the turn gate
    those wake-ups return immediately: measured at 34 ms of `interrupted`
    across 42 of them in an 877 s episode. Drawn as bars they are 1e-5 of the
    axis, but each still paints its white 0.5 pt edge, so agent_5's lane came
    out shredded by events that together cost a thirtieth of a second. Widening
    them to a visible pixel would overstate them by five orders of magnitude,
    so they are folded into the span around them instead -- the messages are
    still on the figure as arrows, which is where a wave is legible anyway.

    Returns ``(spans, absorbed_count)``.
    """
    kept, absorbed = [], 0
    for start, stop, state, first_step, last_step in spans:
        if stop - start < min_width and kept:
            # Hand the time back to the span it interrupted.
            prev = kept[-1]
            kept[-1] = (prev[0], stop, prev[2], prev[3], last_step)
            absorbed += 1
            continue
        kept.append((start, stop, state, first_step, last_step))

    merged = []
    for span in kept:
        if merged and merged[-1][2] == span[2] and abs(merged[-1][1] - span[0]) < 1e-9:
            prev = merged[-1]
            merged[-1] = (prev[0], span[1], prev[2], prev[3], span[4])
            continue
        merged.append(span)
    return merged, absorbed


def _spans(
    transitions: List[Any], end_time: float, end_step: Optional[int] = None,
) -> List[Tuple[float, float, str, int, int]]:
    """``[(start, end, state, env_step_in, env_step_out), ...]``.

    agent_states.json records the moment a state was *entered*, so a span runs
    to the next entry, and the last one to the end of the episode. Both ends of
    the env_step range are carried because only the range is informative: an
    entry step alone reads as "this span cost 0 steps" on the first span of the
    run, which starts at env_step 0 and can run for thousands of ticks.
    """
    # Sorted, because the history is appended from several threads and one
    # entry stamped from a foreign clock used to land before transitions that
    # were already recorded -- which drew as a single span covering the run.
    # The stamping is fixed at the source; this keeps a future one from lying.
    transitions = sorted(transitions, key=lambda entry: float(entry[0]))
    spans = []
    for index, entry in enumerate(transitions):
        timestamp, env_step, state = float(entry[0]), int(entry[1]), str(entry[2])
        following = transitions[index + 1] if index + 1 < len(transitions) else None
        stop = float(following[0]) if following is not None else end_time
        stop_step = int(following[1]) if following is not None else (
            end_step if end_step is not None else env_step
        )
        if stop > timestamp:
            spans.append((timestamp, stop, state, env_step, stop_step))
    return spans


def _step_label(first: int, last: int, mode: str = "range") -> str:
    """``"280"`` when the world did not move, ``"0-280"`` when it did.

    A degenerate range is the point, not a defect: R, W and I all freeze the
    env, so a single number *is* the reading, and seeing the same number on
    three consecutive bars is how the barrier shows up in the figure.

    ``mode="end"`` prints only the step the span ended on. Half the ink for a
    figure read at a glance, at the cost of the range -- and the range is the
    part that says how much simulation the span bought, so keep "range" when
    that is the question.
    """
    if mode == "end":
        return str(max(first, last))
    return str(first) if last <= first else f"{first}-{last}"



def _draw_messages(axes, messages, lane_of, end_time, linewidth: float = 1.0,
                   arrow_scale: float = 10.0, solid: bool = False) -> int:
    """Overlay sender -> recipient arrows. Returns how many were drawn.

    One arrow per (message, recipient). ``lane_of`` holds whatever can be
    addressed in this run: team lanes where the topology talks team to team,
    agent lanes for anything sent by a single robot (COOP2 repair). A message
    lands on the lane of whoever actually sent it, so a team's broadcast is one
    stroke from the team's own lane rather than four from a member that had no
    part in it. Slightly curved so that several messages exchanged at almost the
    same instant do not collapse into a single vertical stroke.
    """
    drawn = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        try:
            when = float(message.get("timestamp"))
        except (TypeError, ValueError):
            continue
        sender = message.get("sender")
        if sender not in lane_of or not 0.0 <= when <= end_time:
            continue
        recipients = [
            r for r in (message.get("recipients") or [])
            if r in lane_of and r != sender
        ]
        if not recipients:
            continue
        # A message that stops a teammate mid-primitive and one that waits in
        # their buffer cost completely different amounts -- the first is an LLM
        # round trip, the second is free -- so they must not be the same stroke.
        # `interrupts_execution` is the field the broker itself reads.
        interrupting = bool((message.get("metadata") or {}).get("interrupts_execution", True))
        # `solid` gives up that distinction on purpose, for a figure where
        # the strokes are meant to read as one thing.
        style = "-" if (interrupting or solid) else (0, (2, 2))
        axes.plot(
            [when], [lane_of[sender]], marker="o", markersize=3.5,
            color=MESSAGE_COLOUR, zorder=5,
        )
        for recipient in recipients:
            axes.annotate(
                "",
                xy=(when, lane_of[recipient]), xytext=(when, lane_of[sender]),
                arrowprops={
                    "arrowstyle": "-|>", "color": MESSAGE_COLOUR,
                    "linewidth": linewidth, "shrinkA": 1.5, "shrinkB": 1.5,
                    "mutation_scale": arrow_scale,
                    "connectionstyle": "arc3,rad=0.12",
                    "linestyle": style, "alpha": 1.0 if interrupting else 0.55,
                },
                zorder=5, annotation_clip=False,
            )
            drawn[interrupting] = drawn.get(interrupting, 0) + 1
    return drawn


def plot_agent_state_timeline(
    agent_states: Dict[str, List[Any]],
    output_path: str,
    title: Optional[str] = None,
    end_step: Optional[int] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    teams: Optional[Dict[str, Any]] = None,
    holds: Optional[Dict[str, List[Any]]] = None,
    legend: bool = True,
    title_fontsize: Optional[float] = None,
    step_label: str = "range",
    step_fontsize: float = 7.0,
    message_linewidth: float = 1.0,
    note_absorbed: bool = True,
    tick_fontsize: Optional[float] = None,
    lane_fontsize: Optional[float] = None,
    xlabel_pad: Optional[float] = None,
    title_fontweight: Optional[str] = None,
    show_lane_labels: bool = True,
    message_arrow_scale: float = 10.0,
    message_solid: bool = False,
) -> Optional[str]:
    """Write a Gantt-style figure of agent states. Returns the path, or None.

    Args:
        agent_states: ``{agent_id: [[wall_clock, env_step, state], ...]}``, the
            contents of agent_states.json.
        output_path: where to write the PNG.
        title: figure title; defaults to the run directory's name.
        end_step: env_step the episode ended on, for the last span of each
            agent. Without it that span's range is left degenerate rather than
            guessed at.
        messages: message_log.json's contents, or None. Only ``timestamp``,
            ``sender`` and ``recipients`` are read; the content is not drawn.
        teams: team_timeline.json's contents, or None. When present each team
            gets a lane above its robots, and messages are drawn between those
            lanes instead of between the members that carried them.
        legend: draw the key below the axes. Turn it off when several of these
            are composed into one figure with a shared key -- and note it also
            changes the *plot* width, not only what is under it: `tight_layout`
            runs with the legend in place and narrows the axes to fit it, so a
            run with more legend entries draws a narrower plot. Panels meant to
            be read side by side have to be drawn without it, or the same
            second is a different number of pixels in each.
    """
    if not agent_states:
        return None

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    agents = sorted(agent_states)
    team_members = (teams or {}).get("teams") or {}
    # Dict order is the order the topology wired the teams in, which is the
    # chain's speaking order and puts the leader first -- worth preserving, so
    # this is not sorted the way the agent lanes are.
    team_order = [
        name for name, members in team_members.items()
        if any(member in agent_states for member in members)
    ]
    rows: List[Tuple[str, str]] = []
    if team_order:
        placed = set()
        for name in team_order:
            rows.append(("team", name))
            for member in team_members[name]:
                if member in agent_states and member not in placed:
                    rows.append(("agent", member))
                    placed.add(member)
        rows.extend(("agent", a) for a in agents if a not in placed)
    else:
        rows = [("agent", a) for a in agents]

    end_time = max(
        (float(entry[0]) for entries in agent_states.values() for entry in entries),
        default=0.0,
    )
    if end_time <= 0:
        return None
    # The last state of every agent runs to the end of the run, and the run is
    # at least as long as the last transition anyone made.
    end_time *= 1.02

    figure, axes = plt.subplots(figsize=(14, 1.4 + 0.8 * len(rows)))
    seen_states = []
    # Half a pixel at the figure's own width: below this a bar is nothing but
    # its own edge stroke.
    min_width = end_time / (14 * 140 * 2)
    lane_of_agent = {name: lane for lane, (kind, name) in enumerate(rows) if kind == "agent"}
    lane_of_team = {name: lane for lane, (kind, name) in enumerate(rows) if kind == "team"}

    # The team lanes first, so the member bars and the arrows sit above them.
    team_spans = (teams or {}).get("spans") or {}
    for name, lane in lane_of_team.items():
        # A full-width ground bar: where it shows through, the team is not
        # thinking and its robots are acting on what it decided last.
        axes.barh(lane, end_time, left=0.0, height=0.34,
                  color=TEAM_IDLE_COLOUR, edgecolor="none", zorder=1)
        for span in team_spans.get(name, []):
            try:
                start, stop = float(span["start"]), float(span["end"])
            except (KeyError, TypeError, ValueError):
                continue
            kind = str(span.get("kind", ""))
            axes.barh(lane, max(stop - start, min_width), left=start, height=0.34,
                      color=TEAM_SPAN_COLOURS.get(kind, "#cccccc"),
                      edgecolor="none", zorder=2)
        # W, which the brain cannot record. It writes its own R and I -- it is
        # the thing doing them -- but a team is in W once the plan is handed out
        # and its robots have not started yet, and by then nothing calls the
        # brain again. So it is read off the members, where it already is: the
        # team is waiting while any of its robots is. Without this the lane went
        # straight from red or amber to the idle ground bar and the team alone
        # appeared to skip a state its every robot passes through.
        for start, stop in _team_waiting(
            agent_states, (teams or {}).get("teams", {}).get(name) or [],
            end_time, end_step, min_width,
        ):
            axes.barh(lane, max(stop - start, min_width), left=start, height=0.34,
                      color=TEAM_SPAN_COLOURS["waiting"], edgecolor="none", zorder=2)

    absorbed_total = 0
    for agent_id, lane in lane_of_agent.items():
        spans, absorbed = _coalesce(
            _spans(agent_states[agent_id], end_time, end_step), min_width
        )
        absorbed_total += absorbed
        agent_holds = (holds or {}).get(agent_id) or []
        # Holding is state X too, so the split happens here and not in the FSM:
        # only the plan's specification separates a team hold from real work.
        pieces = [
            piece for span in spans for piece in _split_on_holds(span, agent_holds)
        ]
        pieces, sliver = _coalesce(pieces, min_width)
        absorbed_total += sliver
        for start, stop, state, first_step, last_step in pieces:
            axes.barh(
                lane, stop - start, left=start, height=0.55,
                color=(HOLDING_COLOUR if state == "holding"
                       else STATE_COLOURS.get(state, "#cccccc")),
                edgecolor="white", linewidth=0.5,
            )
            if state not in seen_states:
                seen_states.append(state)
            # The env_step range inside the span, where it fits: it is how a
            # reader ties this figure back to plan_logs.json, and the width of
            # the range is how much simulation the span actually bought.
            if step_label != "none" and stop - start > end_time * 0.05:
                axes.text(
                    (start + stop) / 2, lane,
                    _step_label(first_step, last_step, step_label),
                    ha="center", va="center", fontsize=step_fontsize, color="white",
                )

    # Teams and robots share one address space on the figure, because they do
    # in the log: a message record names its sender, and that is a team name
    # when a team sent it and an agent id when a robot did.
    drawn_messages = _draw_messages(
        axes, messages or [], {**lane_of_agent, **lane_of_team}, end_time,
        linewidth=message_linewidth, arrow_scale=message_arrow_scale,
        solid=message_solid,
    )
    message_count = sum(drawn_messages.values())

    for lane, (kind, _name) in enumerate(rows):
        if kind == "team" and lane:
            axes.axhline(lane - 0.5, color="#adb5bd", linewidth=0.6, zorder=0)

    axes.set_yticks(range(len(rows)))
    # Off for every panel but the first when several are stitched together:
    # the lanes are the same robots in the same order, so repeating the
    # column costs width and says nothing.
    axes.set_yticklabels(
        [name if kind == "team" else f"   {name}" for kind, name in rows]
        if show_lane_labels else [""] * len(rows),
        fontsize=lane_fontsize if lane_fontsize is not None else tick_fontsize,
    )
    for label, (kind, _name) in zip(axes.get_yticklabels(), rows):
        if kind == "team":
            label.set_fontweight("bold")
    axes.set_ylim(-0.6, len(rows) - 0.4)
    axes.invert_yaxis()
    # A sliver of left margin: a leader's opening broadcast is sent at t~0, and
    # against xlim=(0, ...) its marker and arrowhead sit on the spine.
    axes.set_xlim(-end_time * 0.012, end_time)
    xlabel = "time (s)"
    if step_label != "none":
        xlabel += ("  --  labels inside the bars are the env_step "
                   + ("the span ended on" if step_label == "end" else "range"))
    if absorbed_total and note_absorbed:
        # Kept by default: it is a disclosure, not decoration -- spans too
        # narrow to draw were folded away, and a reader counting bars is
        # otherwise not told. Turn it off only for a figure that has said
        # it elsewhere.
        xlabel += f"  |  {absorbed_total} sub-pixel spans folded into their neighbour"
    axes.set_xlabel(xlabel)
    if tick_fontsize is not None:
        # One size for both axes and the axis name. The y labels take it
        # through set_yticklabels; the x numbers and the axis name each need
        # saying separately or they stay at the rcParam default.
        axes.tick_params(axis="x", labelsize=tick_fontsize)
        axes.xaxis.label.set_fontsize(tick_fontsize)
    if xlabel_pad is not None:
        axes.xaxis.labelpad = xlabel_pad
    axes.set_title(title or os.path.basename(os.path.dirname(os.path.abspath(output_path))),
                   fontsize=title_fontsize, fontweight=title_fontweight)
    axes.grid(axis="x", alpha=0.3, linestyle=":")

    order = [s for s in ("reasoning", "interrupted", "waiting", "executing", "holding")
             if s in seen_states]
    # "holding" is a shading of executing, not a state of its own, and the
    # label says so -- side by side and unqualified they read as five states.
    labels = {"holding": "executing: holding for the team (idle)"}
    handles = [
        mpatches.Patch(
            color=HOLDING_COLOUR if s == "holding" else STATE_COLOURS.get(s, "#cccccc"),
            label=labels.get(s, s),
        )
        for s in order
    ]
    if lane_of_team:
        # Only the ground bar gets an entry. A team lane is red exactly when its
        # members are, and amber exactly when they are interrupted, so it reads
        # off the state colours already listed -- a second swatch of the same
        # red under a different name would read as two different things.
        handles.append(mpatches.Patch(
            color=TEAM_IDLE_COLOUR, label="team not thinking (robots acting)",
        ))
    if message_count:
        # Proxy artists, because an annotate() arrow is not a legend handle.
        from matplotlib.lines import Line2D  # noqa: PLC0415

        for interrupting, label in ((True, "message, interrupts"), (False, "message, buffered")):
            count = drawn_messages.get(interrupting, 0)
            if not count:
                continue
            handles.append(Line2D(
                [0], [0], color=MESSAGE_COLOUR, marker="o", markersize=4, linewidth=1.0,
                linestyle="-" if interrupting else (0, (2, 2)),
                alpha=1.0 if interrupting else 0.55,
                label=f"{label} (n={count})",
            ))
    if legend:
        axes.legend(
            handles=handles,
            loc="upper center", bbox_to_anchor=(0.5, -0.28),
            ncol=len(handles) or 1, frameon=False,
        )

    figure.tight_layout()
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _merge_ranges(ranges: List[Any], low: int, high: int) -> List[Tuple[int, int]]:
    """The hold ranges clipped to ``[low, high]``, overlaps merged, sorted."""
    clipped = []
    for start, end in ranges:
        start, end = int(start), int(max(end, start))
        first, last = max(start, low), min(end, high)
        if last > first:
            clipped.append((first, last))
    clipped.sort()
    merged: List[Tuple[int, int]] = []
    for first, last in clipped:
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def _split_on_holds(span, ranges: List[Any]):
    """Cut an executing span where the agent was holding for its team.

    A span is not one thing. `_coalesce` folds the sub-pixel R/W seams between
    consecutive plans into their neighbour, so a single bar routinely covers a
    real primitive *and* the `wait_for_team` that followed it -- and a bar is
    only honest if the two are drawn in different colours. Asking whether the
    whole bar sits inside a hold answers "no" for exactly those bars, which is
    the ones worth splitting, so the question is where the boundaries fall
    rather than which side of one the span is on.

    env_step is the only clock both the span and the hold are written in, so
    the split points come from interpolating time linearly across the span's
    step range. That is exact while the env is stepping, which is what an
    executing span is.
    """
    start, stop, state, first_step, last_step = span
    if state != "executing" or not ranges:
        return [span]
    if last_step <= first_step:
        # No range to interpolate over: it is one or the other, so ask which.
        for low, high in ranges:
            if low <= first_step <= max(high, low):
                return [(start, stop, "holding", first_step, last_step)]
        return [span]

    merged = _merge_ranges(ranges, first_step, last_step)
    if not merged:
        return [span]
    scale = (stop - start) / (last_step - first_step)
    at = lambda step: start + (step - first_step) * scale

    pieces, cursor = [], first_step
    for low, high in merged:
        if low > cursor:
            pieces.append((at(cursor), at(low), "executing", cursor, low))
        pieces.append((at(low), at(high), "holding", low, high))
        cursor = high
    if cursor < last_step:
        pieces.append((at(cursor), at(last_step), "executing", cursor, last_step))
    return pieces


def _team_waiting(agent_states, members, end_time, end_step, min_width):
    """When a team is in W: the union of its robots' waiting spans, merged.

    The union rather than the intersection, for the same reason the lane shows
    one planning span for four robots -- W here means "this team has committed
    and is not moving yet", and that is true from the first robot entering it.
    Sub-pixel slivers are dropped: a member passes through W in microseconds on
    the paths where the plan was already in hand, and drawing those would stipple
    the lane with events that cost nothing.
    """
    intervals = []
    for member in members:
        for start, stop, state, _first, _last in _spans(
            agent_states.get(member) or [], end_time, end_step
        ):
            if state == "waiting" and stop - start >= min_width:
                intervals.append((start, stop))
    intervals.sort()
    merged: List[Tuple[float, float]] = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _holds(run_dir: str, end_step: Optional[int] = None) -> Dict[str, List[Any]]:
    """Per-agent env_step ranges spent idling rather than working.

    Read from plan_logs.json, because agent_states.json cannot say: a hold and a
    real primitive are both state X, and only the plan's specification
    distinguishes them.

    Two sources, because a hold is not always written down. The explicit ones
    are the ``wait_for_team`` plans. The rest are the *gaps*: env_steps in which
    an agent had no plan on record at all. A gap can only open while the world
    is moving, and the world does not move while any agent is in R or W, so a
    step-gap means this robot was standing in X with nothing to do -- which is
    the same idling by another name. Reading them saves the figure from
    depending on a hold having been filed, and lets it tell the truth about runs
    recorded before it was.
    """
    path = os.path.join(run_dir, "plan_logs.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    plans = payload if isinstance(payload, list) else (payload.get("plan_history") or [])
    ranges: Dict[str, List[Any]] = {}
    working: Dict[str, List[Tuple[int, int]]] = {}
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        start, end = plan.get("start_step"), plan.get("end_step")
        if start is None:
            continue
        span = (int(start), int(end if end is not None else start))
        agent_id = str(plan.get("agent_id"))
        if str(plan.get("specification", "")).startswith("wait_for_team"):
            ranges.setdefault(agent_id, []).append(span)
        else:
            working.setdefault(agent_id, []).append(span)

    for agent_id, spans in working.items():
        cursor = 0
        for low, high in sorted(spans):
            if low > cursor:
                ranges.setdefault(agent_id, []).append((cursor, low))
            cursor = max(cursor, high)
        if end_step is not None and end_step > cursor:
            ranges.setdefault(agent_id, []).append((cursor, int(end_step)))
    return ranges


def _episode_end_step(run_dir: str) -> Optional[int]:
    """Last env_step of the episode, from plan_logs.json.

    agent_states.json only holds transitions, so the final span of every agent
    has no recorded end. Plans are closed out with the episode (they are marked
    interrupted at the step it ended on), so the largest ``end_step`` there is
    that step.
    """
    path = os.path.join(run_dir, "plan_logs.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if isinstance(payload, list):
        plans = payload
    else:
        # The saver writes {"plan_history": [...], "current_plans": {...}}.
        plans = payload.get("plan_history") or payload.get("plans") or []
    steps = [p.get("end_step") for p in plans if isinstance(p, dict)]
    steps = [int(s) for s in steps if isinstance(s, (int, float))]
    return max(steps) if steps else None


def _messages(run_dir: str) -> List[Dict[str, Any]]:
    """message_log.json, or an empty list.

    Absent or empty for the `individual` topology, which is not a failure: that
    topology has no communication channel at all, so the figure simply carries
    no arrows.
    """
    path = os.path.join(run_dir, "message_log.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    return payload if isinstance(payload, list) else []


def _teams(run_dir: str) -> Optional[Dict[str, Any]]:
    """team_timeline.json, or None.

    Absent for runs made before teams existed, and for any run whose agents are
    not team agents; the figure then falls back to one lane per robot.
    """
    path = os.path.join(run_dir, "team_timeline.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def save_team_timeline(agents: Dict[str, Any], output_path: str) -> Optional[str]:
    """Write which robots each team is made of, and when it was thinking.

    Recorded by the brains rather than derived from the members, because a team
    is the thing that thinks here and only it knows when it started. Times are
    made relative to the same origin the agent states use, so both land on one
    x axis. Messages are not written here: they are addressed team to team, so
    message_log.json already records them at team level.
    """
    brains, origins = {}, []
    for agent in agents.values():
        brain = getattr(agent, "brain", None)
        if brain is not None and getattr(brain, "team_name", None):
            brains[brain.team_name] = brain
        start = getattr(agent, "_start_time", None)
        if start is not None:
            origins.append(float(start))
    if not brains or not origins:
        return None
    origin = min(origins)

    payload = {
        "teams": {name: list(brain.member_ids) for name, brain in brains.items()},
        "spans": {
            name: [
                {
                    "kind": span["kind"],
                    "start": float(span["start"]) - origin,
                    "end": float(span["end"]) - origin,
                }
                for span in getattr(brain, "timeline", [])
            ]
            for name, brain in brains.items()
        },
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {len(brains)} team timelines to {output_path}")
    return output_path


def plot_from_run_dir(run_dir: str, filename: str = "agent_timeline.png") -> Optional[str]:
    """Draw the timeline for an existing run directory."""
    states_path = os.path.join(run_dir, "agent_states.json")
    if not os.path.exists(states_path):
        return None
    with open(states_path) as handle:
        agent_states = json.load(handle)
    end_step = _episode_end_step(run_dir)
    return plot_agent_state_timeline(
        agent_states,
        os.path.join(run_dir, filename),
        title=os.path.basename(run_dir),
        end_step=end_step,
        messages=_messages(run_dir),
        teams=_teams(run_dir),
        holds=_holds(run_dir, end_step),
    )


if __name__ == "__main__":
    import sys

    for directory in sys.argv[1:] or ["."]:
        written = plot_from_run_dir(directory)
        print(f"{directory}: {written or 'no agent_states.json'}")
