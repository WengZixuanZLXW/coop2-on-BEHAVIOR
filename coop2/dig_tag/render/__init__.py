"""Drawing a record, and following one as it changes. The one place the
package imports matplotlib (`pip install dig-tag[render]`); nothing below
imports this.

  style.py     one style: fonts in points, geometry in inches, the paper's palette
  canvas.py    drawing primitives on panels measured in inches
  timeline.py  where on the time axis each moment of a record goes: a DIG's activations, or a TAG's actions
  record.py    a record in the paper's convention: the T, D, and Z bands a DIG-TAG, a DIG, or a TAG alone offers
  live.py      LiveFigure, a window that follows a record while agents change it
"""

from .live import LiveFigure
from .record import PITCH, RecordSize, lanes_of, measure_record, render_bars, render_record
from .timeline import ActionTimeline, ActivationTimeline, Timeline

__all__ = [
    "PITCH", "RecordSize", "lanes_of", "measure_record", "render_record", "render_bars",
    "Timeline", "ActivationTimeline", "ActionTimeline",
    "LiveFigure",
]
