"""Draw a run's plan record onto one robot's episode video, frame by frame.

A recorded episode shows what a robot *did*; `plan_logs.json` holds what it was
*told to do* and why that failed. Reading them side by side means matching a
wall-clock moment in a video against an env_step in a JSON file by eye, which
is exactly the kind of thing that gets a diagnosis wrong. This puts the second
next to the first, aligned:

  * the plan the robot is executing, every action of it, the current one
    marked and the finished ones marked done or failed;
  * which action is running right now;
  * the failure text, in full, from the moment the action fails.

The alignment is arithmetic, not a guess. ``MultiViewRecorder`` captures on
``env_step % every == 0`` (``coop_env`` sets ``every=4``), so frame *n* is
env_step ``n * every``, and every action in the log carries its own
``start_step`` / ``end_step``.

    python -m coop2.experiment.annotate_video <run_dir> <robot> [--aspect 4:3]

The text is drawn **inside** the frame, on translucent bands, and the frame is
centre-cropped to ``--aspect`` (4:3 by default). Both are for the same reason:
a slide. A 16:9 page is 13.33 x 7.5 in, so two videos side by side get about
6.2 in of width each -- at 16:9 that is 3.5 in tall and wastes half the page,
and a panel beside the frame (the first version of this, 1740 x 720) is worse
still. Cropped to 4:3 the pair is 6.2 x 4.65 in each, which fills the page and
leaves room for a heading. The camera is a chase view with its robot near the
centre, so a centre crop costs edges rather than subject.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#: Ticks per captured frame -- `coop_env._start_recording`'s `every`.
DEFAULT_EVERY = 4

#: How long a failed plan keeps the panel after it ends, in env_steps.
#:
#: A plan that fails is replaced in the same tick -- the team hands its member
#: a hold immediately -- so without this the failure text is on screen for one
#: frame out of five hundred, which is the same as not drawing it. 400 ticks is
#: about three seconds of playback at the recorded cadence.
FAILURE_LINGER = 400

#: Most actions drawn before the list is elided around the current one. A team
#: hold is 64 identical `wait`s and would otherwise fill the panel and push the
#: failure off the bottom.
MAX_ACTION_LINES = 9

PAD = 16
#: Line height, and with it the type size. Sized for 480p: these are rendered
#: at 960 x 720 and a slide or a downscale shows them at 640 x 480, so a 26 px
#: glyph arrives as 17 px -- about the smallest that survives projection. It is
#: also why the action text drops the `target=` key and why MAX_ACTION_LINES is
#: what it is: every line costs picture.
LINE = 32
#: The text is drawn straight onto the frame with no panel behind it, so each
#: glyph carries its own dark outline instead. That is what makes white on a
#: pale kitchen floor readable, and it costs no picture at all -- which a band
#: does: at this type size the two bands took two thirds of the frame.
STROKE = 3
STROKE_FILL = (0, 0, 0)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]

BG = (16, 16, 18)
DIM = (198, 200, 206)
TEXT = (255, 255, 255)
LIVE = (255, 205, 60)
GOOD = (110, 215, 135)
BAD = (255, 105, 105)


def _font(size: int, bold: bool = False):
    path = FONT_PATHS[1 if bold else 0]
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def plans_for(run_dir: Path, robot: str) -> List[Dict[str, Any]]:
    """Every plan this robot executed, in the order it executed them.

    Both halves of the log are read: `plan_history` holds the plans that
    terminated, `current_plans` the one still running when the episode ended,
    and a run that stops on its step budget leaves its last plan in the second.
    """
    log = json.loads((run_dir / "plan_logs.json").read_text())
    everything = list(log.get("plan_history", [])) + list(log.get("current_plans", {}).values())
    mine = [p for p in everything if p.get("agent_id") == robot and p.get("start_step") is not None]
    mine.sort(key=lambda p: (p["start_step"], p.get("plan_id") or 0))
    # A plan holds the screen until the next one starts, so a failure stays
    # readable past the tick it happened on.
    for plan, following in zip(mine, mine[1:]):
        plan["_until"] = following["start_step"]
    if mine:
        mine[-1]["_until"] = float("inf")
    return mine


def active(plans: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    """The plan whose story this frame tells.

    A plan that just failed wins over the one that replaced it, for
    FAILURE_LINGER ticks: the replacement is the team's hold, which says
    nothing, and the failure is the whole reason to be looking.
    """
    for plan in plans:
        if plan.get("status") != "failed" or plan.get("end_step") is None:
            continue
        if plan["end_step"] <= step < plan["end_step"] + FAILURE_LINGER:
            return plan
    for plan in plans:
        if plan["start_step"] <= step < plan["_until"]:
            return plan
    return None


def current_action(plan: Dict[str, Any], step: int) -> Optional[int]:
    """Index of the action running at @step, or None once the plan is over."""
    for index, action in enumerate(plan.get("actions") or []):
        start, end = action.get("start_step"), action.get("end_step")
        if start is None:
            continue
        if start <= step < (end if end is not None else float("inf")):
            return index
    return None


def _wrap(draw, text: str, font, width: int) -> List[str]:
    """Greedy wrap measured against the font, not counted in characters."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words, line = paragraph.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=font) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _action_text(action: Dict[str, Any]) -> str:
    """``navigate_to(notebook.n.01_1)`` -- the value, not the keyword.

    Every one of these verbs takes exactly one argument, so `target=` and
    `ticks=` name what the verb already says and cost seven characters of a
    line that has to stay readable at 480p.
    """
    args = action.get("args") or {}
    return f"{action.get('action_type')}({', '.join(str(v) for v in args.values())})"


def _rows(plan: Dict[str, Any], index: Optional[int]) -> List[Dict[str, Any]]:
    """The action list as it should be read: runs folded, length capped.

    Consecutive identical actions are one row with a count -- a team hold is
    sixty-four identical `wait`s, and drawing them one per line says nothing
    sixty-four times. What is left is then elided around the running action,
    which is the row the reader is looking for.
    """
    folded: List[Dict[str, Any]] = []
    for i, action in enumerate(plan.get("actions") or []):
        text = _action_text(action)
        # An action the plan has not reached yet is PENDING, whatever the log
        # says it eventually became. The record holds final statuses, so
        # colouring straight from it drew a green "done" tick beside an action
        # that had not run -- on the frame where navigate_to was still in
        # flight, the place_on_top under it already read as succeeded.
        status = action.get("status") if (index is None or i < index) else (
            "running" if i == index else None)
        last = folded[-1] if folded else None
        same = last is not None and last["text"] == text and last["status"] == status
        if same and index not in last["indices"] and i != index:
            last["indices"].append(i)
            continue
        folded.append({"text": text, "status": status, "indices": [i]})
    for row in folded:
        row["count"] = len(row["indices"])
        row["live"] = index is not None and index in row["indices"]
    if len(folded) <= MAX_ACTION_LINES:
        return folded
    live = next((i for i, row in enumerate(folded) if row["live"]), 0)
    half = MAX_ACTION_LINES // 2
    start = max(0, min(live - half, len(folded) - MAX_ACTION_LINES))
    kept = folded[start:start + MAX_ACTION_LINES]
    if start:
        kept = [{"text": f"... {start} earlier", "status": None, "indices": [],
                 "count": 1, "live": False}] + kept[1:]
    tail = len(folded) - (start + MAX_ACTION_LINES)
    if tail > 0:
        kept = kept[:-1] + [{"text": f"... {tail + 1} more", "status": None, "indices": [],
                             "count": 1, "live": False}]
    return kept


def _text(draw, xy, line, font, fill) -> None:
    """One line, outlined so it reads over whatever is behind it."""
    draw.text(xy, line, font=font, fill=fill, stroke_width=STROKE, stroke_fill=STROKE_FILL)


def draw_overlay(image: Image.Image, robot: str, step: int, plan, index, fonts) -> None:
    """The three sections, drawn into the frame itself.

    Top: who this is, and the whole plan with every action marked.
    Bottom: the action running right now, and the failure text when there is
    one. Nothing is drawn between them, so the middle of the frame -- where
    the chase camera keeps the robot -- stays clear.
    """
    small, body, bold = fonts
    draw = ImageDraw.Draw(image)
    inner = image.width - 2 * PAD

    # -- what goes in the top band, measured before anything is drawn --------
    head: List[tuple] = [(robot, bold, TEXT)]
    if plan is None:
        head.append(("no plan", body, DIM))
        rows = []
    else:
        for line in _wrap(draw, plan.get("specification") or "", body, inner):
            head.append((line, body, TEXT))
        rows = _rows(plan, index)

    lines: List[tuple] = list(head)
    for row in rows:
        if row["live"]:
            colour, mark = LIVE, ">"
        elif row["status"] == "failed":
            colour, mark = BAD, "x"
        elif row["status"] == "success":
            colour, mark = GOOD, "v"
        else:
            colour, mark = DIM, " "
        text = row["text"] + (f"  x{row['count']}" if row["count"] > 1 else "")
        for j, line in enumerate(_wrap(draw, text, body, inner - 26)):
            lines.append(((mark if j == 0 else " "), line, colour))

    y = PAD
    for item in lines:
        if len(item) == 3 and isinstance(item[1], str):     # marked action row
            mark, line, colour = item
            _text(draw, (PAD, y), mark, bold, colour)
            _text(draw, (PAD + 24, y), line, body, colour)
        else:
            line, font, colour = item
            _text(draw, (PAD, y), line, font, colour)
        y += LINE
    stamp = f"env_step {step}"
    _text(draw, (image.width - PAD - draw.textlength(stamp, font=small), PAD + 6),
          stamp, small, DIM)

    # -- and what goes in the bottom one -------------------------------------
    tail: List[tuple] = []
    if plan is not None:
        failed = next((a for a in (plan.get("actions") or [])
                       if a.get("status") == "failed"
                       and a.get("end_step") is not None and step >= a["end_step"]), None)
        reason = (failed or {}).get("failure_reason") or (
            plan.get("failure_reason") if plan.get("status") == "failed" and index is None else None)
        # The engine's reason code is dropped rather than shown. It is an
        # internal label -- PRE_CONDITION_ERROR, PLANNING_ERROR -- and what the
        # frame is showing is the sentence the agent is handed back, so the
        # heading says that instead. It also buys a line of picture: the code
        # was 21 characters of a 59-character line.
        text = reason or ""
        if reason and ": " in reason[:40]:
            text = reason.split(": ", 1)[1]
        if reason:
            tail.append(("ERROR FEEDBACK", bold, BAD))
        elif index is None:
            over = plan.get("status") or "ended"
            tail.append((f"NOW  plan {over}", body, BAD if over == "failed" else GOOD))
        else:
            tail.append((f"NOW  {_action_text(plan['actions'][index])}", body, LIVE))
        if reason:
            for line in _wrap(draw, text, body, inner):
                tail.append((line, body, BAD))
    if not tail:
        return
    y = image.height - PAD - len(tail) * LINE
    for line, font, colour in tail:
        _text(draw, (PAD, y), line, font, colour)
        y += LINE


def _crop_box(width: int, height: int, aspect: Optional[str]) -> tuple:
    """Centre crop to @aspect ("4:3", "16:9", "1:1"), or the whole frame."""
    if not aspect:
        return (0, 0, width, height)
    w, h = (float(part) for part in aspect.split(":"))
    want = w / h
    if width / height > want:                       # too wide: take the middle
        keep = int(round(height * want))
        left = (width - keep) // 2
        return (left, 0, left + keep, height)
    keep = int(round(width / want))                 # too tall
    top = (height - keep) // 2
    return (0, top, width, top + keep)


def annotate(run_dir: Path, robot: str, out: Path, every: int = DEFAULT_EVERY,
             aspect: Optional[str] = "4:3") -> Path:
    source = run_dir / f"episode_{robot}.mp4"
    if not source.exists():
        raise SystemExit(f"no video for {robot}: {source}")
    plans = plans_for(run_dir, robot)
    if not plans:
        print(f"[warn] {robot} has no plans with a start_step in this run")

    fonts = (_font(20), _font(26), _font(28, bold=True))
    reader = imageio.get_reader(str(source))
    fps = reader.get_meta_data().get("fps", 30)
    writer = imageio.get_writer(str(out), fps=fps, macro_block_size=1, quality=8)
    frames, size = 0, None
    try:
        for n, frame in enumerate(reader):
            step = n * every
            image = Image.fromarray(frame).convert("RGBA")
            image = image.crop(_crop_box(image.width, image.height, aspect))
            size = image.size
            plan = active(plans, step)
            draw_overlay(image, robot, step, plan,
                         current_action(plan, step) if plan else None, fonts)
            writer.append_data(np.asarray(image.convert("RGB")))
            frames += 1
    finally:
        reader.close()
        writer.close()
    print(f"{out}  ({frames} frames, {frames / fps:.1f} s, env_step 0-{(frames - 1) * every}, "
          f"{size[0]}x{size[1]})")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("robots", nargs="+", help="robot names, as in episode_<name>.mp4")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the annotated files go (default: the run directory)")
    parser.add_argument("--every", type=int, default=DEFAULT_EVERY,
                        help=f"ticks per captured frame (default {DEFAULT_EVERY}, "
                             "which is what coop_env records at)")
    parser.add_argument("--aspect", default="4:3", metavar="W:H",
                        help="centre-crop to this aspect (default 4:3, which puts two "
                             "of these side by side on a 16:9 slide); 'none' keeps the "
                             "recorded 16:9 frame")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for robot in args.robots:
        annotate(args.run_dir, robot, out_dir / f"annotated_{robot}.mp4", args.every,
                 None if args.aspect.lower() == "none" else args.aspect)


if __name__ == "__main__":
    main()
