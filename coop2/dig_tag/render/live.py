"""A figure that follows a record as it changes."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .canvas import add_panel, setup
from .record import measure_record, render_record, timeline_for


class LiveFigure:
    """One window that follows a record while agents change it: a
    `TAGParallelInterface` (its T band), a `DIGParallelInterface` (D and Z), or a `DIGTAG` (all
    three), drawn whole in the paper's convention and fitted to the window.

    The record's `subscribe` hook marks the figure dirty from whatever
    thread acted; `refresh()`, on the main thread (the only one matplotlib
    draws on), redraws when it is dirty, holding a threaded interface
    still while it reads. `watch()` is the loop a script runs: refresh,
    pause, until the window closes, the time is up, or `until()` says the
    run is over; `watch_async()` is the same loop for a record driven by
    asyncio on this thread (the DIG's own runtime), yielding to the event
    loop instead of pausing, so the agents run while the window follows;
    `inline=True` is for a run that is synchronous on the main thread with
    no loop to call from (a LangGraph run recorded through the adapter):
    the figure redraws inside the record's own notification, at every
    change. An activation still open is drawn reaching to now, so a live
    D band moves with the clock. With `capture`, every redraw is kept as a frame
    and `save()` writes the frames as a GIF.

    The axis is wall-clock time from the moment the figure was made: it
    covers `span` seconds to begin with (5), and when the run outgrows it
    the span doubles, by at most `growth` seconds (60) at a time, so the
    frame stays put and moments are added to it rather than the figure
    redrawn around them. When the run is over (`finish()`, which `watch`
    calls when `until()` says so) the axis is fitted to the run and drawn
    once more, and stops growing. `logical` draws the paper's order
    instead, one slot per moment. `pitch` is the room per slot in the record's own
    inches, wider than the paper's so that adjacent labels stay apart on
    a screen.

    The record is drawn at its natural size, offscreen, at the pixel
    density that fits it into the window as the window is at that moment,
    and the window shows that image: zoomed as a whole, type included, and
    following the window when it is resized."""

    AXIS_SLOTS = 16        # a clock axis is this many slots long, whatever its span

    def __init__(self, record: Any, *, bands: Optional[Sequence[str]] = None, width: float = 9.0,
                 height: float = 4.0, ids: bool = True, action_labels: bool = True, context: bool = False,
                 span: float = 5.0, growth: float = 60.0, logical: bool = False, pitch: float = 0.3,
                 capture: bool = False, inline: bool = False, title: Optional[str] = None) -> None:
        setup()
        self.record = record
        self.inline = inline
        self._measure = dict(bands=bands, ids=ids, pitch=pitch)
        self._render = dict(bands=bands, ids=ids, action_labels=action_labels, context=context, pitch=pitch)
        self.logical = logical
        self.span, self.growth = float(span), float(growth)
        self.origin = time.time()
        self.finished = False
        self.capture = capture
        self.frames: List[Any] = []
        self._dirty = True
        self._shown = False
        self.fig = plt.figure(figsize=(width, height))
        manager = getattr(self.fig.canvas, "manager", None)
        if title and manager is not None:
            manager.set_window_title(title)
        record.subscribe(self._changed)

    def _changed(self) -> None:
        """Called by the record after every change, on the acting thread;
        inline, and on the main thread, that is where the redraw happens."""
        self._dirty = True
        if self.inline and threading.current_thread() is threading.main_thread():
            self._show()
            self.refresh()
            self.fig.canvas.flush_events()

    def _show(self) -> None:
        """Put the window up, once, when there is a window to put up."""
        if not self._shown and matplotlib.get_backend().lower() != "agg":
            plt.show(block=False)
        self._shown = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    MIN_SPAN = 1.0     # seconds: a fitted axis is at least this long

    def finish(self) -> None:
        """The run is over: fit the axis to it, draw it once more, and stop
        growing. A change after that still draws, on the fitted axis."""
        self.finished = True
        self.refresh(force=True)

    def _grow(self) -> bool:
        """Extend the span past now, doubling it by at most `growth` at a
        time; returns whether it grew."""
        grown = False
        if self.finished:
            return grown
        while time.time() - self.origin > self.span:
            self.span += min(self.span, self.growth)
            grown = True
        return grown

    def _moving(self) -> bool:
        """Whether an activation is still open: on a clock axis its box
        reaches to now, so the figure moves without a change."""
        dig = getattr(self.record, "dig", None)
        return (dig is not None and not self.logical and not self.finished
                and any(h.is_open for h in dig.activations.values()))

    def refresh(self, force: bool = False) -> bool:
        """Redraw if the record changed since the last redraw, the axis has
        to grow, an open activation reaches further, or `force`; returns
        whether it did. Main thread only."""
        grown = not self.logical and self._grow()
        if not (self._dirty or grown or force or self._moving()):
            return False
        self._dirty = False
        hold = getattr(self.record, "hold", None)          # a threaded interface stands still while read
        with (hold() if hold is not None else nullcontext()):
            self._draw()
        if self.capture:
            self.frames.append(self._frame())
        return True

    MARGIN = 0.15      # inches of the record's own around it

    def _draw(self) -> None:
        window = self.fig
        w_px, h_px = window.canvas.get_width_height()                  # the window as it is now
        ratio = getattr(window.canvas, "device_pixel_ratio", 1.0) or 1.0
        timeline = None
        if not self.logical:
            reach = timeline_for(self.record)                          # the moments' first and last times
            if reach.first is not None and reach.first < self.origin:  # moments from before the figure
                self.origin = reach.first
            if self.finished and reach.last is not None:
                self.span = max(reach.last - self.origin, self.MIN_SPAN)
            timeline = timeline_for(self.record, clock=self.span / self.AXIS_SLOTS, origin=self.origin,
                                    horizon=self.origin + self.span)
        size = measure_record(self.record, timeline=timeline, **self._measure)
        w_in, h_in = size.w + 2 * self.MARGIN, size.h + 2 * self.MARGIN
        dpi = max(1.0, min(w_px / w_in, h_px / h_in)) * ratio         # the whole record fitted, at the screen's density
        offscreen = Figure(figsize=(w_in, h_in), dpi=dpi)
        FigureCanvasAgg(offscreen)
        render_record(add_panel(offscreen, self.MARGIN, self.MARGIN, size.w, size.h), self.record,
                      timeline=timeline, **self._render)
        offscreen.canvas.draw()
        self._image = np.asarray(offscreen.canvas.buffer_rgba()).copy()
        window.clear()
        ax = window.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(self._image)                                          # centered, aspect kept
        window.canvas.draw_idle()

    def _frame(self) -> Any:
        return self._image

    def watch(self, seconds: Optional[float] = None, *, interval: float = 0.1,
              until: Optional[Callable[[], bool]] = None) -> None:
        """Show the window and keep it current: refresh and pause, until the
        window is closed, `seconds` have passed, or `until()` is true, which
        ends the run: the axis is fitted to it and drawn before returning.
        Main thread only."""
        self._show()
        start = time.time()
        while plt.fignum_exists(self.fig.number):
            self.refresh()
            if until is not None and until():
                self.finish()
                break
            if seconds is not None and time.time() - start >= seconds:
                break
            plt.pause(interval)

    async def watch_async(self, *, interval: float = 0.1, until: Optional[Callable[[], bool]] = None) -> None:
        """`watch` for a record driven by asyncio on this thread: refresh,
        let the window process its events, yield to the event loop for
        `interval` seconds, until the window is closed or `until()` is
        true, which ends the run: the axis is fitted to it and drawn."""
        import asyncio

        self._show()
        while plt.fignum_exists(self.fig.number):
            self.refresh()
            self.fig.canvas.flush_events()
            if until is not None and until():
                self.finish()
                break
            await asyncio.sleep(interval)

    def save(self, path: "str | Path", fps: float = 4.0) -> Path:
        """Write the captured frames as a GIF."""
        from PIL import Image

        if not self.frames:
            raise ValueError("no frames: make the figure with capture=True and refresh it")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in self.frames]
        first = images[0].size
        images = [im if im.size == first else im.resize(first) for im in images]   # the window may have been resized
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
        return path

    def close(self) -> None:
        plt.close(self.fig)
