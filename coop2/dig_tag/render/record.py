"""A record in the paper's convention: the T, D, and Z bands on one time
axis, time running left to right in the order of opens, calls, and closes.
Every call is a square, every returned event a circle, every delivery a
dashed arrow into the activation it was presented to; ids are optional.
A T band's square is lettered by its tool (o, e, u, s, j, c), a root
version i, so a dense record stays legible.
The record may be a DIG-TAG (all three bands), a DIG on its own (D and
Z), or a TAG on its own -- a `TAGGraph` or the `TAGParallelInterface` over it
(T alone, versions at the actions that returned them). `render_bars` is
the D band for hundreds of agents: activations as bars at their logical
step, deliveries as lines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from matplotlib.patches import Rectangle

from ..base import ToolCall
from ..dig import DIGParallelInterface, DIGEvent
from ..dig.views import delivery, presented_to, roots
from ..interface import TAG_ORIGIN
from ..tag import TAGParallelInterface, TAGGraph, TaskAction, Version, close_target
from ..tag.views import closed_identities
from . import style as S
from .canvas import Panel, Point, text_width
from .timeline import ActionTimeline, ActivationTimeline, Key, Timeline

# a record's geometry (inches)
PITCH = 0.115            # per call in D; an activation's open takes half of it, its end three quarters
T_PITCH = 0.13           # per call when T is drawn: an action's square at its slot, its versions ACTION_GAP after
ACTION_GAP = 0.065       # from an action's square to the versions it returned
LANE_T, LANE_D, LANE_D_COMPACT = 0.2, 0.26, 0.2
BOX_H = 0.17
LABEL_PAD = 0.15         # before a lane label
BAND_GAP = 0.1
Z_H = 0.36
AXIS_H = 0.22
RIGHT = 0.2              # room for a trailing label
LABEL_UP = 0.04          # a letter's baseline above its lane; an id hangs the same way below
ARC_H = 0.12             # the height of an arc over the nodes between an edge's ends

ROUTED = ("send", "root", "start")     # deliveries that a send or the entry routed, drawn dashed


def letter(name: str) -> str:
    """A mark's label, the first letter of what it is: o, e, u, s, j, c for
    the task tool an action applied, i for an initial version. An
    attachment is labeled by its evidence id instead."""
    return name[0]


def tool_fill(call: ToolCall) -> str:
    return {ToolCall.Label.TASK: S.TASK_TOOL, ToolCall.Label.ENVIRONMENT: S.ENV_TOOL,
            ToolCall.Label.INFORMATION: S.COMM_TOOL}[call.label]


def event_fill(event: DIGEvent) -> str:
    if event.origin == TAG_ORIGIN:
        return S.TASK_EVENT
    return S.ENV_EVENT if event.origin == "environment" else S.COMM_EVENT


def lanes_of(dig: DIGParallelInterface, agents: Optional[Sequence[str]] = None) -> List[str]:
    """Agents in a given order, or in order of first activation."""
    if agents is not None:
        return list(agents)
    return list(dict.fromkeys(a.agent_id for a in dig.dig.activations.values()))


def identities_of(tag: TAGGraph) -> List[str]:
    return list(dict.fromkeys(t.identity for t in tag.tasks.values()))


def steps(dig: DIGParallelInterface) -> Dict[str, int]:
    """Logical time: one past the deepest step of the activations whose
    returns an activation received (roots are step 0)."""
    step: Dict[str, int] = {}
    for activation in dig.dig.activations.values():
        depth = 0
        for event in dig.received(activation):
            if event.producer is not None:
                depth = max(depth, step[event.producer.activation_id])
        step[activation.id] = depth + 1
    return step


def first_presentation(dig: DIGParallelInterface, event: DIGEvent, activation) -> bool:
    return next((a for a in presented_to(dig, event) if a.agent_id == activation.agent_id), None) is activation


@dataclass
class Sources:
    """What a record offers to draw: its DIG (D and Z), its TAG (T), the
    timeline placing their moments, and the bands asked for."""

    dig: Optional[DIGParallelInterface]
    tag: Optional[TAGGraph]
    timeline: Timeline
    bands: Tuple[str, ...]


def timeline_for(record: Any, **options: Any) -> Timeline:
    """The timeline a record's moments take: a TAG's own actions, or a
    DIG's activations. `options` are the timeline's (clock, origin, horizon)."""
    if isinstance(record, TAGParallelInterface):
        record = record.tag
    if isinstance(record, TAGGraph):
        return ActionTimeline(record, **options)
    if isinstance(record, DIGParallelInterface):
        return ActivationTimeline(record, getattr(record, "tag", None), **options)
    raise TypeError(f"a record to draw is a DIGParallelInterface, a DIGTAG, a TAGGraph, or a TAGParallelInterface; got {record!r}")


def sources(record: Any, bands: Optional[Sequence[str]] = None, *, timeline: Optional[Timeline] = None) -> Sources:
    """A DIG-TAG offers T, D, and Z; a DIG on its own D and Z; a TAG on
    its own (a `TAGGraph`, or the `TAGParallelInterface` over one) T. `bands`
    selects among those, in any order; None takes all the record offers.
    The timeline is the record's own, in logical order, unless one is
    given (`timeline_for(record, clock=...)` for a wall-clock one)."""
    timeline = timeline if timeline is not None else timeline_for(record)
    if isinstance(record, TAGParallelInterface):
        record = record.tag
    if isinstance(record, TAGGraph):
        dig, tag, offered = None, record, ("T",)
    else:
        dig, tag = record, getattr(record, "tag", None)
        offered = (("T",) if tag is not None else ()) + ("D", "Z")
    wanted = tuple(bands) if bands is not None else offered
    missing = [band for band in wanted if band not in offered]
    if missing:
        raise ValueError(f"this record has no {', '.join(missing)} band to draw; it offers {', '.join(offered)}")
    return Sources(dig, tag, timeline, wanted)


def label_margin(src: Sources, agents: Optional[Sequence[str]]) -> float:
    """Room for the longest lane label."""
    labels: List[str] = []
    if "D" in src.bands:
        active = {a.agent_id for a in src.dig.dig.activations.values()}
        labels += [a if a in active else f"{a}: none" for a in lanes_of(src.dig, agents)]
    if "T" in src.bands:
        labels += identities_of(src.tag)
    return LABEL_PAD + max([text_width(s, S.FONT_MIN) for s in labels] + [0.2]) + 0.08


@dataclass
class RecordSize:
    w: float
    h: float
    bands: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # band -> (y0, height)


def pitch_for(src: Sources, pitch: Optional[float]) -> float:
    """The room per call: what the caller asks, else T_PITCH when T is
    drawn (an action's square and its versions per call), else PITCH."""
    if pitch is not None:
        return pitch
    return T_PITCH if "T" in src.bands else PITCH


def measure_record(record: Any, *, bands: Optional[Sequence[str]] = None,
                   agents: Optional[Sequence[str]] = None, ids: bool = True, axis: bool = True,
                   pitch: Optional[float] = None, timeline: Optional[Timeline] = None) -> RecordSize:
    """The panel a record needs, and where its bands go: from the bottom,
    the time axis (when drawn), then Z, D, T. Without ids the agent lanes
    are denser; `pitch` is the room per call (`pitch_for`), for a sparse
    record whose labels need more than the default."""
    src = sources(record, bands, timeline=timeline)
    return _measure(src, agents=agents, ids=ids, axis=axis, pitch=pitch_for(src, pitch))


def _measure(src: Sources, *, agents: Optional[Sequence[str]], ids: bool, axis: bool, pitch: float) -> RecordSize:
    lane_d = LANE_D if ids else LANE_D_COMPACT
    heights = {
        "T": (len(identities_of(src.tag)) if src.tag is not None else 0) * LANE_T + 0.19,
        "D": (len(lanes_of(src.dig, agents)) if src.dig is not None else 0) * lane_d + 0.12,
        "Z": Z_H,
    }
    width = label_margin(src, agents) + (src.timeline.extent + 0.5) * pitch
    width += ACTION_GAP + RIGHT if "T" in src.bands else 0.12
    y = AXIS_H if axis else 0.05
    placed: Dict[str, Tuple[float, float]] = {}
    for band in ("Z", "D", "T"):
        if band in src.bands:
            placed[band] = (y, heights[band])
            y += heights[band] + BAND_GAP
    return RecordSize(width, y - BAND_GAP + 0.05, placed)


def render_record(panel: Panel, record: Any, *, bands: Optional[Sequence[str]] = None, ids: bool = True,
                  context: bool = False, agents: Optional[Sequence[str]] = None,
                  highlight: Optional[Dict[str, Set[str]]] = None, axis: bool = True,
                  action_labels: bool = True, pitch: Optional[float] = None,
                  timeline: Optional[Timeline] = None) -> None:
    """`ids` labels versions, activations, and attachments; `action_labels`
    letters each action above its square (`letter`); `context`
    also draws, dotted, the events an activation had from earlier;
    `highlight` = {"versions": ids, "evidence": ids, "actions": ids} draws
    the named versions, attachments, and actions bold -- an action's
    arrows also when both its ends are named, so a set of versions is one
    lineage -- and the other task relations gray; `axis` draws the time
    arrow under the bands, with seconds ticked on a clock timeline;
    `timeline` places the moments (the record's own logical order when
    not given; `timeline_for(record, clock=...)` for wall-clock time)."""
    src = sources(record, bands, timeline=timeline)
    _RecordDrawing(panel, src, ids=ids, context=context, agents=agents, highlight=highlight, axis=axis,
                   action_labels=action_labels, pitch=pitch_for(src, pitch)).draw()


class _RecordDrawing:
    """One record being drawn: the geometry the bands share, then the bands
    in order -- D's activations and roots, T's actions and versions, its
    edges and attachments, the deliveries between the two, Z's
    transitions, and the axis. The positions each step leaves
    (`pos_event`, `box_left`, `pos_action`, `pos_task`) are what the later
    steps draw against."""

    def __init__(self, panel: Panel, src: Sources, *, ids: bool, context: bool,
                 agents: Optional[Sequence[str]], highlight: Optional[Dict[str, Set[str]]], axis: bool,
                 action_labels: bool, pitch: float) -> None:
        self.panel, self.src = panel, src
        self.dig, self.tag, self.timeline = src.dig, src.tag, src.timeline
        self.ids, self.context, self.agents, self.axis, self.action_labels = ids, context, agents, axis, action_labels
        self.size = _measure(src, agents=agents, ids=ids, axis=axis, pitch=pitch)
        self.lane_pitch = LANE_D if ids else LANE_D_COMPACT
        marked = highlight or {}
        self.highlight = bool(highlight)
        self.hl_versions = set(marked.get("versions", ()))
        self.hl_evidence = set(marked.get("evidence", ()))
        self.hl_actions = set(marked.get("actions", ()))
        evidence = self.tag.evidence.values() if self.tag is not None else ()
        self.hl_events = {e.source for e in evidence if e.id in self.hl_evidence}
        self.dim = "#8c8c8c" if highlight else S.INK
        self.hl_calls = set()
        if self.tag is not None and self.dig is not None:
            for action in self.tag.actions.values():
                if self.on_lineage(action):
                    self.hl_calls.add((action.issuer, action.call.index))
        self.closed: Set[str] = set(closed_identities(self.tag)) if self.tag is not None else set()   # drawn gray
        self.pitch = pitch
        self.label_w = label_margin(src, agents)
        self.x_root = self.label_w
        self.lane_t: Dict[str, float] = {}
        self.lane_d: Dict[str, float] = {}
        self.pos_event: Dict[str, Point] = {}
        self.box_left: Dict[str, Point] = {}
        self.pos_action: Dict[str, Point] = {}
        self.pos_task: Dict[str, Point] = {}

    def on_lineage(self, action: TaskAction) -> bool:
        """Whether an action is drawn bold: named in the highlight, or
        between highlighted versions (its output alone for an open)."""
        if action.id in self.hl_actions:
            return True
        returned = any(q in self.hl_versions for q in action.output_ids)
        return returned and (not action.input_ids or any(q in self.hl_versions for q in action.input_ids))

    def x_at(self, slots: float) -> float:
        """Inches from the panel's left for a timeline position."""
        return self.label_w + slots * self.pitch

    def x_of(self, key: Key) -> float:
        return self.x_at(self.timeline.x[key])

    def draw(self) -> None:
        self.bands_and_lanes()
        if "D" in self.size.bands:
            self.activations()
            self.roots()
        if "T" in self.size.bands:
            self.task_nodes()
            self.task_edges()
            self.attachments()
        self.deliveries()
        if "Z" in self.size.bands:
            self.transitions()
        if self.axis:
            self.time_axis()

    def bands_and_lanes(self) -> None:
        """The tinted bands with their letters, and the lane labels: task
        identities in T, agents in D (an agent with no activation says so)."""
        panel, size = self.panel, self.size
        for band, (y0, h) in size.bands.items():
            panel.band(0.1, y0, size.w - 0.15, h, {"T": S.BAND_T, "D": S.BAND_D, "Z": S.BAND_Z}[band])
            panel.text((0.05, y0 + h / 2), band, fontsize=S.FONT_TITLE, style="italic", weight="bold")
        if "T" in size.bands:
            y0, h = size.bands["T"]
            self.lane_t = {k: y0 + h - 0.13 - i * LANE_T for i, k in enumerate(identities_of(self.tag))}
            for k, y in self.lane_t.items():
                panel.text((LABEL_PAD, y), k, fontsize=S.FONT_MIN, ha="left",
                           color=S.CLOSED if k in self.closed else "#666666", style="italic")
                panel.line((self.label_w - 0.05, y), (size.w - 0.1, y), color="#e8d9c4", lw=0.4, z=1)
        if "D" in size.bands:
            y0, h = size.bands["D"]
            self.lane_d = {a: y0 + h - 0.12 - i * self.lane_pitch for i, a in enumerate(lanes_of(self.dig, self.agents))}
            active = {a.agent_id for a in self.dig.dig.activations.values()}
            for a, y in self.lane_d.items():
                panel.text((LABEL_PAD, y), a if a in active else f"{a}: none", fontsize=S.FONT_MIN, ha="left",
                           color="#555555" if a in active else S.LOOP, style="italic")

    def activations(self) -> None:
        """D: activations as boxes spanning their calls, each call a square
        with the events it returned stacked beside it."""
        panel, dig = self.panel, self.dig
        for activation in dig.dig.activations.values():
            y = self.lane_d[activation.agent_id]
            left, right = self.x_of(("open", activation.id)) - 0.03, self.x_of(("end", activation.id)) + 0.03
            panel.box(((left + right) / 2, y), right - left, BOX_H)
            self.box_left[activation.id] = (left, y)
            if self.ids:
                panel.text((left + 0.01, y + BOX_H / 2 + 0.005), activation.id, fontsize=S.FONT_MIN, ha="left", va="bottom")
            for call in activation.calls:
                x = self.x_of(("call", (activation.id, call.index)))
                panel.square((x, y), tool_fill(call))
                if (activation.id, call.index) in self.hl_calls:
                    panel.square((x, y), "none", z=6, lw=1.0)
                n = len(call.returned)
                gap = min(0.05, 0.15 / max(1, n - 1))          # returned events stacked in the box
                for k, event in enumerate(call.returned):
                    self.pos_event[event.id] = (x + 0.06, y + ((n - 1) / 2 - k) * gap)
                    panel.circle(self.pos_event[event.id], event_fill(event))
                    if event.id in self.hl_events:
                        panel.circle(self.pos_event[event.id], "none", z=6, edge=S.INK, lw=1.0)

    def roots(self) -> None:
        """The root events at the left of D, each between the lanes of the
        agents it entered through; a task root is drawn in T when T is."""
        panel, dig = self.panel, self.dig
        for event in roots(dig):
            presented = presented_to(dig, event)
            entry = list(dict.fromkeys(a.agent_id for a in presented if delivery(dig, a, event) in ("root", "start")))
            entry = entry or list(dict.fromkeys(a.agent_id for a in presented))
            if entry:
                self.pos_event[event.id] = (self.x_root, sum(self.lane_d[a] for a in entry) / len(entry))
                panel.circle(self.pos_event[event.id], event_fill(event))

    def _agent_of(self, activation_id: str) -> str:
        return self.dig.dig.activations[activation_id].agent_id

    def _identity_of_action(self, action: TaskAction) -> str:
        """The lane an action's square sits on: the identity it closes,
        else its first output's, so an action on a consumed version sits
        on the line it branched and its input arrives on a diagonal."""
        if action.label is TaskAction.Label.CLOSE:
            return close_target(action.call)
        return action.outputs[0].identity

    def _closed_action(self, action: TaskAction) -> bool:
        """An action is in the closed part of the graph once every
        identity it returned is closed (a close, always)."""
        if action.label is TaskAction.Label.CLOSE:
            return True
        return all(q.identity in self.closed for q in action.outputs)

    def _label_color(self, bold: bool, closed: bool) -> str:
        if bold:
            return S.LABEL_RED
        if closed:
            return S.CLOSED
        return S.LABEL_RED if not self.highlight else self.dim

    def task_nodes(self) -> None:
        """T: the bipartite graph's nodes. Each action is a red square at
        its call on its first output's lane, lettered above; each version is a circle
        ACTION_GAP after the action that returned it, on its identity's
        lane, a root version at the left. What is closed is gray: the
        versions of a closed identity and the actions whose every output
        is closed, the whole task line. When D is drawn a dotted vertical
        ties each square to its call."""
        panel, tag = self.panel, self.tag
        produced = {q for action in tag.actions.values() for q in action.output_ids}
        for task in tag.tasks.values():
            if task.id not in produced:
                self._version(task, self.x_root, root=True)
        for action in tag.actions.values():
            identity = self._identity_of_action(action)
            x, y = self.x_at(self.timeline.action(action)), self.lane_t[identity]
            self.pos_action[action.id] = (x, y)
            bold, closed = self.on_lineage(action), self._closed_action(action)
            panel.square((x, y), S.CLOSED if closed else S.TASK_TOOL, side=S.T_SQ, z=6, lw=1.0 if bold else 0.5)
            if self.action_labels:                  # the tool's letter, centered above the square
                panel.text((x, y + LABEL_UP), letter(action.label.value), fontsize=S.FONT_MIN, va="bottom",
                           color=self._label_color(bold, closed))
            if "D" in self.size.bands:
                panel.line((x, self.lane_d[self._agent_of(action.issuer)] + S.SQ / 2), (x, y - S.T_SQ / 2),
                           style="dotted", color="#8a8a8a", lw=0.5)
            for task in action.outputs:
                self._version(task, x + ACTION_GAP)

    def _version(self, task: Version, x: float, *, root: bool = False) -> None:
        panel = self.panel
        p = (x, self.lane_t[task.identity])
        self.pos_task[task.id] = p
        closed, bold = task.identity in self.closed, task.id in self.hl_versions
        panel.circle(p, S.CLOSED if closed else S.TASK_EVENT, r=S.R_TASK, z=6,
                     edge=S.INK if bold else S.MARK_EDGE, lw=1.1 if bold else 0.5)
        if root:
            panel.text((x, p[1] + LABEL_UP), letter("initial"), fontsize=S.FONT_MIN, va="bottom",
                       color=S.CLOSED if closed else S.LABEL_RED)
        if self.ids:                            # centered under the circle, clear of the next call's letter on the lane below
            panel.text((x, p[1] - LABEL_UP), task.id, fontsize=S.FONT_MIN, va="top",
                       color=S.CLOSED if closed else "#7a4a1a", backdrop=S.BAND_T)

    def task_edges(self) -> None:
        """T: the bipartite graph's edges, arrows from the versions an
        action consumed to its square and from its square to the versions
        it returned; bold along the highlighted lineage, gray elsewhere
        when there is one. A close consumes nothing, so a dotted arrow
        from the identity's latest version names what it closes."""
        for action in self.tag.actions.values():
            square, bold = self.pos_action[action.id], self.on_lineage(action)
            if action.label is TaskAction.Label.CLOSE:
                before = [self.pos_task[q.id] for q in self.tag.tasks.values()
                          if q.identity == close_target(action.call) and self.pos_task[q.id][0] < square[0]]
                if before:
                    latest = max(before, key=lambda p: p[0])
                    self.panel.arrow(latest, square, style="dotted", lw=1.2 if bold else 0.6, head=4,
                                     color=S.INK if bold or not self.highlight else self.dim,
                                     shrink_a=S.R_TASK, shrink_b=S.T_SQ / 2, z=5 if bold else 4)
                continue
            for q in action.input_ids:
                self._task_edge(self.pos_task[q], square, bold and q in self.hl_versions, into_square=True)
            for q in action.output_ids:
                self._task_edge(square, self.pos_task[q], bold and q in self.hl_versions, into_square=False)

    def _task_edge(self, a: Point, b: Point, bold: bool, *, into_square: bool) -> None:
        same_lane = abs(a[1] - b[1]) < 1e-9
        between = same_lane and any(          # a node between them on the lane: arc over it
            abs(p[1] - a[1]) < 1e-9 and min(a[0], b[0]) < p[0] < max(a[0], b[0])
            for p in (*self.pos_task.values(), *self.pos_action.values())
        )
        over = -min(0.45, ARC_H / max(abs(b[0] - a[0]), 1e-9))      # an arc of at most ARC_H, however long the edge
        self.panel.arrow(a, b, lw=1.2 if bold else 0.6, color=S.INK if bold or not self.highlight else self.dim,
                         head=4, rad=over if between else (0.0 if same_lane else -0.15),
                         shrink_a=S.R_TASK if into_square else S.T_SQ / 2,
                         shrink_b=S.T_SQ / 2 if into_square else S.R_TASK, z=5 if bold else 4)

    def attachments(self) -> None:
        """T: each attachment as a link from the attaching call up to its
        target's lane -- from the call's square when D is drawn, else from
        the foot of the band -- labeled by its evidence id."""
        panel = self.panel
        for at, phi, agent in self.timeline.attachments():
            x, target = self.x_at(at), self.pos_task[phi.target]
            bold = phi.id in self.hl_evidence
            foot = self.lane_d[agent] + S.SQ / 2 if "D" in self.size.bands else self.size.bands["T"][0] + 0.02
            panel.line((x, foot), (x, target[1] - S.R_TASK),
                       style="solid" if bold else "dotted", color=S.EVIDENCE, lw=1.0 if bold else 0.5, z=4)
            if self.ids:
                where = (x + 0.02, target[1] - 0.1) if "D" in self.size.bands else (x + 0.02, foot)
                panel.text(where, phi.id, fontsize=S.FONT_MIN, ha="left",
                           va="top" if "D" in self.size.bands else "bottom",
                           color=S.EVIDENCE if bold or not self.highlight else self.dim)

    def deliveries(self) -> None:
        """Each routed delivery as a dashed arrow into the activation the
        event was first presented to; with `context`, what the activation
        had from earlier dotted."""
        if self.dig is None:
            return
        panel, dig = self.panel, self.dig
        for activation in dig.dig.activations.values():
            if activation.id not in self.box_left:
                continue
            left, y = self.box_left[activation.id]
            for event in activation.inputs:
                if event.id not in self.pos_event:
                    continue
                src = self.pos_event[event.id]
                if delivery(dig, activation, event) in ROUTED and first_presentation(dig, event, activation):
                    panel.arrow((src[0] + S.R_EVENT, src[1]), (left, y), style="dashed", color=S.SEND, lw=0.55, head=4,
                                rad=-0.12 if src[1] > y else 0.12, z=4)
                elif self.context:      # what the runtime's state carried, or the agent's own earlier return
                    panel.arrow((src[0] + S.R_EVENT, src[1]), (left, y), style="dotted", color=S.CONTEXT, lw=0.45,
                                head=3.5, rad=-0.18 if src[1] > y else 0.18, z=1)

    def transitions(self) -> None:
        """Z: one transition per environment call, tied to the call's square."""
        panel, dig, size = self.panel, self.dig, self.size
        y0, h = size.bands["Z"]
        zy = y0 + h / 2
        env_calls = sorted(((a, c) for a in dig.dig.activations.values() for c in a.calls
                            if c.label is ToolCall.Label.ENVIRONMENT),
                           key=lambda ac: self.timeline.x[("call", (ac[0].id, ac[1].index))])
        zx = [self.x_root] + [self.x_of(("call", (a.id, c.index))) for a, c in env_calls]
        for k, x in enumerate(zx):
            panel.circle((x, zy), S.ENV_EVENT, r=0.055)
            panel.text((x, zy), f"z{k}", fontsize=S.FONT_MIN, color="white")
            if k > 0:
                panel.arrow((zx[k - 1] + 0.06, zy), (x - 0.06, zy), color=S.ENV_EVENT, lw=0.6, head=4)
                if "D" in size.bands:
                    panel.line((x, self.lane_d[env_calls[k - 1][0].agent_id] - S.SQ / 2), (x, zy + 0.06),
                               style="dotted", color=S.ENV_TOOL, lw=0.5)
        panel.text((size.w - 0.08, zy), "...", fontsize=S.FONT_LABEL, ha="right", color=S.ENV_EVENT)

    TICK_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)

    def time_axis(self) -> None:
        """The time arrow under the bands; on a clock timeline, ticked in
        seconds (minutes past two of them), at most six ticks."""
        panel, size, line = self.panel, self.size, self.timeline
        panel.arrow((self.label_w - 0.05, 0.1), (size.w - 0.06, 0.1), lw=0.6, head=5)
        span = line.span
        if span is None:
            panel.text(((self.label_w + size.w) / 2, 0.02), "Time", fontsize=S.FONT_MIN, va="bottom", style="italic")
            return
        step = next((s for s in self.TICK_STEPS if span / s <= 6), self.TICK_STEPS[-1])
        t = 0.0
        while t <= span + 1e-9:
            x = self.x_at(line.at(line.origin + t))
            panel.line((x, 0.08), (x, 0.12), lw=0.5, color=S.MARK_EDGE)
            label = f"{t / 60:g}m" if step >= 120 else f"{t:g}s"
            panel.text((x, 0.02), label, fontsize=S.FONT_MIN, va="bottom", color="#777777")
            t += step


def render_bars(panel: Panel, dig: DIGParallelInterface) -> None:
    """Hundreds of agents: activations as bars in thin lanes at their
    logical step, deliveries as lines from the producing activation to the
    receiving one; lanes and steps are fitted to the panel."""
    step = steps(dig)
    lanes = lanes_of(dig)
    y0, h = 0.05, panel.h - 0.1
    panel.band(0.1, y0, panel.w - 0.15, h, S.BAND_D)
    panel.text((0.05, y0 + h / 2), "D", fontsize=S.FONT_TITLE, style="italic", weight="bold")
    n_steps = max(step.values())
    x_left, x_right = 0.42, panel.w - 0.12
    pitch = (x_right - x_left) / n_steps
    bar_w = min(0.3, pitch * 0.4)
    lane_pitch = min(0.02, (h - 0.22) / max(1, len(lanes) - 1))
    lane = {a: y0 + h - 0.16 - i * lane_pitch for i, a in enumerate(lanes)}
    pos = {a.id: (x_left + (step[a.id] - 0.5) * pitch, lane[a.agent_id]) for a in dig.dig.activations.values()}
    x_root = x_left - 0.12
    segments = []
    for activation in dig.dig.activations.values():
        x, y = pos[activation.id]
        for event in activation.inputs:
            if delivery(dig, activation, event) not in ROUTED or not first_presentation(dig, event, activation):
                continue
            px, py = (x_root + 0.03, y) if event.producer is None else pos[event.producer.activation_id]
            segments.append([(px + (0.0 if event.producer is None else bar_w / 2), py), (x - bar_w / 2, y)])
    panel.lines(segments, color=S.SEND, lw=0.35, alpha=0.35)
    bar_h = max(0.006, min(0.014, lane_pitch * 0.7))
    for x, y in pos.values():
        panel.ax.add_patch(Rectangle((x - bar_w / 2, y - bar_h / 2), bar_w, bar_h, facecolor=S.ACT_FILL,
                                     edgecolor=S.ACT_EDGE, linewidth=0.3, zorder=3))
    for root in roots(dig):
        ys = [pos[a.id][1] for a in presented_to(dig, root)]
        if ys:
            panel.circle((x_root, sum(ys) / len(ys)), event_fill(root), r=0.03)
    panel.text((x_root, y0 + h - 0.04), "step", fontsize=S.FONT_MIN, color="#777777", va="top")
    for s in range(1, n_steps + 1):
        panel.text((x_left + (s - 0.5) * pitch, y0 + h - 0.04), str(s), fontsize=S.FONT_MIN, color="#777777", va="top")
