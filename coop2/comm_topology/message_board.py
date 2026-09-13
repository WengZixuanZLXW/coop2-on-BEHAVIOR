"""One board every team reads and writes -- the medium of `decentralized_messageboard`.

The mode is `individual` with a shared record: no team addresses another, no
team waits for another, nothing interrupts anyone. What changes is that every
team, each time it returns to reasoning, is shown the board -- every post any
team has made, oldest first, the ones since it last looked marked ``(new)`` --
and posts one entry of its own in the same model call that produces its plans
(``LLMMessageboardPlanResponse.board_post``). The post is the team's public
commitment for the round: which robot goes for which cargo or leg, what it is
leaving to others. A team that reads it before planning can avoid the
collision `individual` mode has no way to avoid (three drones for one die, in
every individual run so far).

A post is not a message. It has no recipient, it reaches nobody's inbox, and
it stops nobody: a team learns of it only when it next plans. That is the
whole difference from the chain and the leader, which both interrupt. The
**notify tool** is the reserved way to cross that line -- a team choosing
which other teams to interrupt with what -- and this module holds its seam:
:meth:`MessageBoard.notify` records the request and hands it to a deliverer
if one is installed. None is installed yet, and the response schema does not
offer the field until ``MessageboardTeamBrain.NOTIFY_TOOL_ENABLED`` is set,
so today a notify is a record and nothing else. See the brain for the rest of
the reservation.

Thread-safety: brains call in from their own agents' threads, so every
mutation is under one lock. Posts are sequenced, and "new since I last
looked" is a sequence number a reader keeps, not a per-reader flag on the
board.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

__all__ = ["BoardPost", "MessageBoard", "NotifyRecord", "render_board"]


@dataclass
class BoardPost:
    seq: int
    team: str
    content: str
    env_step: Optional[int]
    timestamp: float
    #: Which planning round of the posting team this was written in.
    round: int = 0


@dataclass
class NotifyRecord:
    """A team's request to interrupt other teams -- recorded, not yet delivered.

    Reserved for the notify tool. When it is delivered, ``delivered`` names the
    teams that were actually interrupted; until then it stays empty and the
    record is the only trace the request leaves.
    """

    seq: int
    sender: str
    targets: List[str]
    content: str
    env_step: Optional[int]
    timestamp: float
    delivered: List[str] = field(default_factory=list)


#: A deliverer takes a NotifyRecord and returns the teams it interrupted. The
#: brain that wires the notify tool installs one; without it, notify() only
#: records.
Deliverer = Callable[[NotifyRecord], Sequence[str]]


class MessageBoard:
    """The shared board: append-only posts, plus the reserved notify seam."""

    def __init__(self):
        self._lock = threading.RLock()
        self._posts: List[BoardPost] = []
        self._notifies: List[NotifyRecord] = []
        self._seq = 0
        #: Installed by whoever implements the notify tool's delivery. None
        #: means "record only", which is the reserved state.
        self.deliverer: Optional[Deliverer] = None

    # -- posts ---------------------------------------------------------------

    def post(self, team: str, content: str, env_step: Optional[int] = None,
             round: int = 0, timestamp: Optional[float] = None) -> Optional[BoardPost]:
        """Append @team's post. Blank content posts nothing and returns None."""
        content = " ".join(str(content or "").split())
        if not content:
            return None
        with self._lock:
            self._seq += 1
            entry = BoardPost(
                seq=self._seq, team=team, content=content, env_step=env_step,
                timestamp=time.time() if timestamp is None else timestamp, round=round,
            )
            self._posts.append(entry)
            return entry

    def posts(self, after: int = 0) -> List[BoardPost]:
        """Every post with seq > @after, oldest first."""
        with self._lock:
            return [p for p in self._posts if p.seq > after]

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def __len__(self) -> int:
        with self._lock:
            return len(self._posts)

    # -- the reserved notify seam ------------------------------------------

    def notify(self, sender: str, targets: Sequence[str], content: str,
               env_step: Optional[int] = None, timestamp: Optional[float] = None) -> NotifyRecord:
        """Record @sender's request to interrupt @targets with @content.

        Delivery is the deliverer's job, and there is none by default: the
        request is kept (``notifies()``) so a run's record shows what a team
        *wanted* to interrupt, which is the evidence the tool's design needs
        before it interrupts anything.
        """
        content = " ".join(str(content or "").split())
        with self._lock:
            self._seq += 1
            record = NotifyRecord(
                seq=self._seq, sender=sender, targets=[t for t in targets if t and t != sender],
                content=content, env_step=env_step,
                timestamp=time.time() if timestamp is None else timestamp,
            )
            self._notifies.append(record)
            deliverer = self.deliverer
        if deliverer is not None and record.targets:
            record.delivered = list(deliverer(record))
        return record

    def notifies(self) -> List[NotifyRecord]:
        with self._lock:
            return list(self._notifies)

    # -- persistence ---------------------------------------------------------

    def to_records(self) -> Dict[str, List[Dict]]:
        """JSON-ready: the posts and the (reserved) notify requests."""
        with self._lock:
            return {
                "posts": [asdict(p) for p in self._posts],
                "notifies": [asdict(n) for n in self._notifies],
            }


def render_board(posts: Sequence[BoardPost], reader: str, new_after: int = 0,
                 heading: str = "## 7. MESSAGE BOARD (shared by every team; oldest first)",
                 limit: int = 12, old_cutoff: int = 240) -> str:
    """Section 7 for the board mode: the posts as the reader should see them.

    The reader's own posts are labelled ``You (team)`` so it does not answer
    itself; posts after @new_after are marked ``(new)`` -- what appeared since
    the reader last planned is what it has to react to, the rest is context
    and is cut short. Empty board -> a one-line note, so the model knows the
    board exists and is empty rather than absent.
    """
    lines = [heading]
    if not posts:
        lines.append("  (nothing posted yet -- you are among the first to plan; "
                     "your board_post this round is what the others will read)")
        return "\n".join(lines)
    for post in list(posts)[-limit:]:
        who = f"You ({post.team})" if post.team == reader else post.team
        is_new = post.seq > new_after and post.team != reader
        content = post.content
        if not is_new and len(content) > old_cutoff:
            content = content[:old_cutoff - 3] + "..."
        step = "?" if post.env_step is None else post.env_step
        lines.append(f"  [step {step}] {who}{' (new)' if is_new else ''}: {content}")
    return "\n".join(lines)
