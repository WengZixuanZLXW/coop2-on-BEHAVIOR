"""Drawing primitives on a canvas measured in inches.

A `Panel` is an axes whose data coordinates are inches from its lower-left
corner, so every renderer places marks in inches and the style's sizes hold
everywhere. Nothing here knows about DIG-TAG. The backend is the caller's:
a batch build selects Agg before drawing, a live figure takes the
interactive one."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from . import style as S

Point = Tuple[float, float]

LINESTYLES = {"solid": "-", "dashed": (0, (2.6, 1.6)), "dotted": (0, (1, 1.6))}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": S.FONT_LABEL,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


_FONT_SCALE = [1.0]


@contextmanager
def font_scale(factor: float):
    """Scale every font drawn inside the block: for a figure the paper
    includes below its native size, so its labels print like the others."""
    _FONT_SCALE.append(factor)
    try:
        yield
    finally:
        _FONT_SCALE.pop()


def pt(fontsize: float) -> float:
    return max(fontsize, S.FONT_MIN) * _FONT_SCALE[-1]


def font_ratio() -> float:
    """How much larger type is drawn here than the style says. A renderer
    multiplies the room it leaves for a label by this, so the geometry
    follows the type instead of being outgrown by it."""
    return _FONT_SCALE[-1]


def text_width(label: str, fontsize: float) -> float:
    """A serviceable estimate of rendered text width, inches."""
    return 0.5 * pt(fontsize) / 72.0 * len(label)


def text_extent(label: str, fontsize: float) -> float:
    """The rendered width of a label, inches, for a cell sized to its text
    (the estimate is loose on capitals and mathtext)."""
    scratch = plt.figure(figsize=(1, 1))
    try:
        artist = scratch.text(0, 0, label, fontsize=pt(fontsize))
        return artist.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi
    finally:
        plt.close(scratch)


@dataclass
class Panel:
    ax: plt.Axes
    w: float
    h: float

    # -- marks --------------------------------------------------------------

    def node(self, xy: Point, label: str, *, kind: str = "agent", faint: bool = False,
             fontsize: float = S.FONT_NODE, r: float = S.NODE_R) -> float:
        """An agent node: a circle, widened into a pill when the label needs
        it. Returns the half-width used (for arrow shrinking)."""
        fill, edge, color = {
            "agent": (S.AGENT_FILL, S.AGENT_EDGE, S.INK),
            "gray": ("#ffffff", S.GRAY, "#555555"),
        }[kind]
        alpha = 0.35 if faint else 1.0
        half_w = max(r, text_width(label, fontsize) / 2 + 0.045)
        x, y = xy
        self.ax.add_patch(FancyBboxPatch((x - half_w, y - r), 2 * half_w, 2 * r,
                                         boxstyle=f"round,pad=0,rounding_size={r}",
                                         facecolor=fill, edgecolor=edge, linewidth=0.7, alpha=alpha, zorder=4))
        if label:
            self.ax.text(x, y, label, ha="center", va="center", fontsize=pt(fontsize), color=color, alpha=alpha, zorder=5)
        return half_w

    def square(self, xy: Point, fill: str, side: float = S.SQ, z: int = 5, lw: float = 0.5) -> None:
        self.ax.add_patch(Rectangle((xy[0] - side / 2, xy[1] - side / 2), side, side, facecolor=fill,
                                    edgecolor=S.INK if lw > 0.5 else S.MARK_EDGE, linewidth=lw, zorder=z))

    def circle(self, xy: Point, fill: str, r: float = S.R_EVENT, z: int = 5, edge: str = S.MARK_EDGE,
               lw: float = 0.5) -> None:
        self.ax.add_patch(Circle(xy, r, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))

    def box(self, xy: Point, w: float, h: float, *, fill: str = S.ACT_FILL, edge: str = S.ACT_EDGE,
            z: int = 3, radius: float = 0.03, style: str = "solid") -> None:
        """A rounded box centered at xy."""
        self.ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                         boxstyle=f"round,pad=0,rounding_size={radius}",
                                         facecolor=fill, edgecolor=edge, linewidth=0.6, zorder=z,
                                         linestyle=LINESTYLES[style]))

    def band(self, x: float, y: float, w: float, h: float, fill: str) -> None:
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="none", zorder=0))

    def cross(self, xy: Point, size: float = 0.03, color: str = S.MARK_EDGE) -> None:
        x, y = xy
        self.ax.plot([x - size, x + size], [y - size, y + size], color=color, linewidth=0.7, zorder=6)
        self.ax.plot([x - size, x + size], [y + size, y - size], color=color, linewidth=0.7, zorder=6)

    # -- lines --------------------------------------------------------------

    def arrow(self, a: Point, b: Point, *, style: str = "solid", color: str = S.INK, lw: float = 0.8,
              rad: float = 0.0, shrink_a: float = 0.0, shrink_b: float = 0.0, head: float = 5.0, z: int = 2) -> None:
        """An arrow from a to b (inches), ends shrunk by inches."""
        self.ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle=f"-|>,head_length={head * 0.7},head_width={head * 0.35}", mutation_scale=1,
            connectionstyle=f"arc3,rad={rad}", linestyle=LINESTYLES[style], linewidth=lw, color=color,
            shrinkA=shrink_a * 72, shrinkB=shrink_b * 72, zorder=z,
        ))

    def line(self, a: Point, b: Point, *, style: str = "solid", color: str = S.INK, lw: float = 0.6, z: int = 1) -> None:
        self.ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=LINESTYLES[style], color=color, linewidth=lw, zorder=z)

    def lines(self, segments: Sequence[Sequence[Point]], *, color: str, lw: float, alpha: float, z: int = 1) -> None:
        from matplotlib.collections import LineCollection
        self.ax.add_collection(LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=z))

    # -- text ---------------------------------------------------------------

    def text(self, xy: Point, s: str, *, fontsize: float = S.FONT_LABEL, ha: str = "center", va: str = "center",
             color: str = S.INK, style: str = "normal", weight: str = "normal", z: int = 7,
             rotation: float = 0.0, backdrop: Optional[str] = None) -> None:
        """Text at a point. `rotation` is degrees counterclockwise, for a
        label that has to run alongside something tall and narrow;
        `backdrop` fills a tight box behind the text, for a label a line
        would otherwise cross."""
        bbox = dict(facecolor=backdrop, edgecolor="none", pad=0.4) if backdrop else None
        self.ax.text(xy[0], xy[1], s, fontsize=pt(fontsize), ha=ha, va=va, color=color, style=style,
                     weight=weight, zorder=z, rotation=rotation, bbox=bbox)


def new_figure(w: float, h: float) -> plt.Figure:
    return plt.figure(figsize=(w, h))


def add_panel(fig: plt.Figure, x: float, y: float, w: float, h: float, scale: float = 1.0) -> Panel:
    """A panel of w x h inches with its lower-left corner at (x, y) inches.
    With `scale`, the panel is drawn at that fraction of its size: its
    marks and lines shrink, its type does not."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x / fw, y / fh, w * scale / fw, h * scale / fh])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    return Panel(ax, w, h)


def figure_text(fig: plt.Figure, x: float, y: float, s: str, *, fontsize: float = S.FONT_LABEL, ha: str = "center",
                va: str = "center", weight: str = "normal", style: str = "normal", color: str = S.INK) -> None:
    """Text placed in figure inches."""
    fw, fh = fig.get_size_inches()
    fig.text(x / fw, y / fh, s, fontsize=pt(fontsize), ha=ha, va=va, weight=weight, style=style,
             color=color)


def legend_row(fig: plt.Figure, x: float, y: float, width: float, items: Iterable[Tuple[str, str, str]]) -> None:
    """A row of legend entries (kind, fill, label), each as wide as its
    label, spread evenly over `width` inches."""
    items = list(items)
    widths = [0.24 + text_width(label, S.FONT_MIN) for _, _, label in items]
    slack = max(0.0, width - sum(widths)) / max(1, len(items) - 1)
    lx = x
    for (kind, fill, label), w in zip(items, widths):
        panel = add_panel(fig, lx, y - 0.09, 0.18, 0.18)
        if kind == "box":
            panel.box((0.09, 0.09), 0.16, 0.1, fill=fill, radius=0.02)
        elif kind == "square":
            panel.square((0.09, 0.09), fill)
        else:
            panel.circle((0.09, 0.09), fill, r=0.04)
        figure_text(fig, lx + 0.22, y, label, fontsize=S.FONT_MIN, ha="left")
        lx += w + slack


def save(fig: plt.Figure, stem: str, out_dir, preview: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in (("pdf", "png") if preview else ("pdf",)):
        path = out_dir / f"{stem}.{ext}"
        # no creation date in the PDF: the same record renders the same bytes, so a committed
        # figure changes in git only when the figure does
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
        paths[ext] = str(path)
    plt.close(fig)
    return paths
