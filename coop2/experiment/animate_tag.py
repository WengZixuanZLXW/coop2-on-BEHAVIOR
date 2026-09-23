"""Replay a run's task graph and film it growing, one call at a time.

``tag.png`` is the graph a run ended with: every version, every action, all at
once. Which is the wrong picture for the question the mode exists to answer --
what did the teams *do to each other's understanding*, and when. That is a
sequence, and this draws it as one: a frame per recorded call, each labelled
with the env_step it landed on, the team that issued it and the sentence that
team wrote in the same breath.

The frames are real replays, not cropped drawings. Each call is re-issued into
a fresh ``TAGParallelInterface`` with its original issuer and arguments, so
what a frame shows is the graph as the next team would have observed it,
produced by the same code that produced it live. A call the graph rejected at
the time is shown as rejected and not applied, because it did not happen.

    python -m coop2.experiment.animate_tag <run_dir> [--out PATH] [--hold 1.2]

The panel is measured once, against the FINAL graph, and every frame is drawn
at that size. Re-measuring per frame would fill each one edge to edge and make
every existing node jump as the next arrives, which is unreadable in motion:
the point of an animation is that what has not changed stays put.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import imageio.v2 as imageio
import matplotlib
import numpy as np

#: Seconds each call is held on screen before the next lands.
DEFAULT_HOLD = 1.2
DEFAULT_FPS = 30
#: Extra inches under the graph for the caption.
CAPTION_H = 1.4
#: Every font drawn here, multiplied. The record style is sized for a figure
#: printed in a paper column; on a screen at video size its version ids come
#: out too small to read. `font_scale` also widens the room the renderer
#: leaves for labels, which is why the measure below happens inside the same
#: block -- scaling only the drawing would overrun the panel it was measured
#: for, and the frames would stop being the same size.
FONT_SCALE = 1.5
#: Pixels across the finished film. The record figure is about five inches
#: wide, which at matplotlib's screen dpi is a 519 px frame -- too small to
#: read a version id in, and h264 rejects odd dimensions outright. So the dpi
#: is derived from this instead of fixed, and the frame is padded to even.
TARGET_WIDTH = 1440


def _rounds(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "tag_rounds.json"
    if not path.exists():
        raise SystemExit(f"not a tag run: {path} is missing")
    return json.loads(path.read_text())


def _calls(run_dir: Path) -> List[Dict[str, Any]]:
    """Every tag action of the run, in the order it was issued.

    Read from the rounds rather than from ``tag.json``: the graph keeps only
    the calls it accepted, while the rounds keep the rejected ones too, and a
    rejection is a thing worth watching -- it is the graph refusing an action
    against a version some other team has already moved past.
    """
    out: List[Dict[str, Any]] = []
    for record in _rounds(run_dir):
        for action in record.get("tag_actions") or []:
            out.append({
                "tool": action.get("tool"),
                "args": action.get("args") or {},
                "result": action.get("result"),
                "team": record.get("team"),
                "env_step": record.get("env_step"),
                "stage": record.get("stage"),
                "reasoning": record.get("reasoning") or "",
                "observed": record.get("observed") or [],
                "notify": (record.get("notify") or {}).get("sent") or [],
            })
    return out


def _fresh():
    from coop2.dig_tag.tag import TAG_TOOLS, TAGParallelInterface, observation_with_history

    return TAGParallelInterface(returns={tool: observation_with_history for tool in TAG_TOOLS})


def _replay_all(calls: List[Dict[str, Any]]):
    """The finished graph, for measuring. Same path every frame takes."""
    tag = _fresh()
    for call in calls:
        if call["result"] != "applied":
            continue
        try:
            tag.step(call["team"], call["tool"], call["args"])
        except Exception:  # noqa: BLE001 - a replay that diverges must not stop the film
            pass
    return tag


def _wrap(text: str, width: float, fontsize: float) -> List[str]:
    """Wrap to @width INCHES, measured in the font it will be drawn in.

    A character count cannot do this job here: `FONT_SCALE` multiplies the type
    without changing the panel, so the same count that fitted at the style's
    size runs off the right edge at 1.5x. It did.
    """
    from coop2.dig_tag.render import canvas

    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if canvas.text_width(trial, fontsize) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _caption(fig, width: float, call: Dict[str, Any], index: int, total: int) -> None:
    from coop2.dig_tag.render import canvas
    from coop2.dig_tag.render import style as S

    applied = call["result"] == "applied"
    y = CAPTION_H - 0.22
    head = (f"env_step {call['env_step']}   {call['team']}   {call['stage']}   "
            f"call {index + 1}/{total}")
    canvas.figure_text(fig, 0.12, y, head, fontsize=S.FONT_LABEL, ha="left", weight="bold")

    y -= 0.30
    verb = f"{call['tool']}"
    target = call["args"].get("task") or call["args"].get("target") or call["args"].get("identity")
    if target:
        verb += f"({target})"
    mark = verb if applied else f"{verb}   REJECTED: {call['result']}"
    canvas.figure_text(fig, 0.12, y, mark, fontsize=S.FONT_LABEL, ha="left",
                       color=S.INK if applied else "#c02020")
    if call["notify"]:
        canvas.figure_text(fig, width - 0.12, y, "notify -> " + ", ".join(call["notify"]),
                           fontsize=S.FONT_MIN, ha="right", color="#9a6b00")

    y -= 0.28
    # Three lines, and a mark when there was more: a sentence that stops
    # mid-clause reads as the model having trailed off, which it did not.
    lines = _wrap(call["reasoning"], width - 0.24, S.FONT_MIN)
    shown, rest = lines[:3], lines[3:]
    if rest:
        shown[-1] = shown[-1] + " ..."
    for line in shown:
        canvas.figure_text(fig, 0.12, y, line, fontsize=S.FONT_MIN, ha="left", style="italic")
        y -= 0.24


def _fit(image: np.ndarray, size: Optional[tuple]) -> np.ndarray:
    """Letterbox @image into exactly @size, or hand it back unchanged.

    For a pair on one slide: the robot videos are 960x720 and this record is
    naturally much wider, so scaling to a common WIDTH would leave two clips of
    different heights. Fitting inside a fixed canvas instead -- scale to
    whichever axis binds, centre, pad with the figure's own white -- makes the
    two files literally the same shape, which is what lets them sit side by
    side without either being stretched.
    """
    if size is None:
        return image
    from PIL import Image  # noqa: PLC0415

    want_w, want_h = size
    h, w = image.shape[:2]
    scale = min(want_w / w, want_h / h)
    new = Image.fromarray(image).resize((max(1, int(round(w * scale))),
                                         max(1, int(round(h * scale)))), Image.LANCZOS)
    canvas_out = np.full((want_h, want_w, 3), 255, dtype=image.dtype)
    x, y = (want_w - new.width) // 2, (want_h - new.height) // 2
    canvas_out[y:y + new.height, x:x + new.width] = np.asarray(new)
    return canvas_out


def _even(image: np.ndarray) -> np.ndarray:
    """h264 wants both dimensions even; pad with the figure's own white."""
    h, w = image.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return image
    out = np.full((h + h % 2, w + w % 2, 3), 255, dtype=image.dtype)
    out[:h, :w] = image
    return out


def _caption_width(calls: List[Dict[str, Any]], total: int) -> float:
    """Inches the widest caption needs, so no frame clips its own text.

    The caption is laid out in figure inches, and the figure is as wide as the
    RECORD. That is fine for a whole run, whose record is wide; a five-call
    window's record is not, and the header ran off the right edge -- "call 6/"
    with the rest gone. Measuring the longest line once and widening the figure
    to it fixes every frame rather than the one that happened to be looked at.
    """
    from coop2.dig_tag.render import canvas
    from coop2.dig_tag.render import style as S

    widest = 0.0
    for index, call in enumerate(calls):
        head = (f"env_step {call['env_step']}   {call['team']}   {call['stage']}   "
                f"call {index + 1}/{total}")
        widest = max(widest, canvas.text_width(head, S.FONT_LABEL))
    # The caption starts 0.12 in from the left and the notify badge ends 0.12 in
    # from the right; the rest is slack so a long line never touches the edge.
    return widest + 0.40


def _frame(tag, size, call, index, total, pitch, dpi, fig_w) -> np.ndarray:
    from coop2.dig_tag.render import canvas, render_record

    fig = canvas.new_figure(fig_w, size.h + 0.2 + CAPTION_H)
    fig.set_dpi(dpi)
    fig.patch.set_facecolor("white")
    render_record(canvas.add_panel(fig, 0.1, 0.1 + CAPTION_H, size.w, size.h), tag,
                  bands=("T",), ids=True, axis=True, pitch=pitch)
    _caption(fig, fig_w, call, index, total)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    matplotlib.pyplot.close(fig)
    return _even(image)


def animate(run_dir: Path, out: Path, hold: float = DEFAULT_HOLD, fps: int = DEFAULT_FPS,
            pitch: Optional[float] = None, first: int = 1, last: Optional[int] = None,
            seconds: Optional[float] = None, size_px: Optional[tuple] = None) -> Path:
    """Film the graph growing, optionally only over calls @first..@last.

    A range shows the frames for those calls and nothing else, but the replay
    still starts at call 1: what makes call 4 legible is the three versions
    already on the board when it lands, so a range that replayed only its own
    calls would draw a different graph, not a window onto this one.

    The panel is measured against the graph as it stands at the END of the
    window, not at the end of the run. Measuring against the whole run keeps a
    clip's nodes where a full film would put them, which sounds right and is
    not: a five-call window of a twenty-five-call run then occupies a fifth of
    a panel sized for all of it, and letterboxing that into a slide-sized
    canvas leaves the version ids too small to read. Within one clip the
    measure is still taken once, before any frame, so nothing moves as the
    graph grows -- which is the property that actually matters in motion.
    """
    from coop2.comm_topology.llm_tag import TAG_FIGURE_PITCH
    from coop2.dig_tag.render import canvas, measure_record

    pitch = TAG_FIGURE_PITCH if pitch is None else pitch
    calls = _calls(run_dir)
    if not calls:
        raise SystemExit(f"{run_dir} records no tag actions")
    lo = max(1, first) - 1
    hi = len(calls) if last is None else min(len(calls), last)
    if lo >= hi:
        raise SystemExit(f"call range {first}-{last} is empty; the run has {len(calls)}")
    shown = hi - lo
    if seconds is not None:
        hold = seconds / shown

    with matplotlib.rc_context(), canvas.font_scale(FONT_SCALE):
        canvas.setup()
        # Measured once, on the graph as the window leaves it: see above.
        size = measure_record(_replay_all(calls[:hi]), bands=("T",), ids=True, axis=True,
                              pitch=pitch)
        fig_w = max(size.w + 0.2, _caption_width(calls[lo:hi], len(calls)))
        dpi = int(round(TARGET_WIDTH / fig_w))
        tag = _fresh()
        writer = imageio.get_writer(str(out), fps=fps, macro_block_size=1, quality=8)
        repeat = max(1, int(round(hold * fps)))
        shape = None
        try:
            for index, call in enumerate(calls):
                if index >= hi:
                    break
                if call["result"] == "applied":
                    try:
                        tag.step(call["team"], call["tool"], call["args"])
                    except Exception as error:  # noqa: BLE001
                        print(f"[warn] call {index + 1} ({call['tool']}) would not replay: {error}")
                if index < lo:
                    continue                    # replayed, not filmed
                image = _fit(_frame(tag, size, call, index, len(calls), pitch, dpi, fig_w),
                             size_px)
                if shape is None:
                    shape = image.shape
                elif image.shape != shape:
                    # Only possible if the measure above was wrong; say so rather
                    # than let imageio fail on frame 40 of 60.
                    raise SystemExit(f"frame {index} is {image.shape}, expected {shape}")
                for _ in range(repeat):
                    writer.append_data(image)
        finally:
            writer.close()
    total = shown * repeat / fps
    print(f"{out}  (calls {lo + 1}-{hi} of {len(calls)}, {total:.1f} s, "
          f"{shape[1]}x{shape[0]}, {hold:.2f} s per call)")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="output file (default: <run_dir>/tag_growing.mp4)")
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD,
                        help=f"seconds per call (default {DEFAULT_HOLD})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--pitch", type=float, default=None,
                        help="room per call in the drawing; the run's own value by default")
    parser.add_argument("--calls", default=None, metavar="A-B",
                        help="film only calls A..B (1-based, inclusive). The graph is still "
                             "replayed from call 1 and the panel still measured against the "
                             "whole run, so a window looks like the same drawing.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="total length; overrides --hold, which is then per call")
    parser.add_argument("--size", default=None, metavar="WxH",
                        help="letterbox each frame into exactly this canvas, e.g. 960x720 to "
                             "match the robot videos so the two sit side by side")
    args = parser.parse_args(argv)
    first, last = 1, None
    if args.calls:
        bits = args.calls.split("-")
        first = int(bits[0])
        last = int(bits[1]) if len(bits) > 1 and bits[1] else None
    size_px = None
    if args.size:
        w, h = (int(v) for v in args.size.lower().split("x"))
        size_px = (w, h)
    animate(args.run_dir, args.out or args.run_dir / "tag_growing.mp4",
            args.hold, args.fps, args.pitch, first, last, args.seconds, size_px)


if __name__ == "__main__":
    main()
