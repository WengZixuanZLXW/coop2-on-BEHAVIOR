"""The shared-message-board ablation (`board`): DIG-TAG's task-graph team with
the task graph replaced by an unstructured shared message board.

Ported from HappyEureka/dig-tag-icra `ma_crafter/comm_topology/llm_message_board.py`,
whose own docstring states the contract this file has to keep:

    The board is a list of posts every teammate reads in full ... Its output's
    space part is `write`, the board's one tool, the message to write or null;
    then `notify`, then `plan`, **exactly as for the task-graph team**, so the
    two differ only in what the shared space is.

That last clause is the ablation, and it is enforced structurally rather than
by care: the round is ``notifying_team.NotifyingTeamBrain``, the same object
`tag` uses -- as upstream shares ``notify.py`` between its two spaces -- so
this file cannot drift from `tag` in anything but the space. Three
substitutions, and there is nowhere else to put a fourth:

    task graph  -> message board          the shared space
    tag_actions -> write                  the space's tools, eight against one
    version ids -> post indices           what "observed" means for the record

The prompt sections, the barrier, the order actions-then-notify-then-plans,
the per-step notify budget and the late-notify path are all the base's.

(The first draft of this file was `llm_tag.py` copied and edited, 120 lines
of it identical; the user asked for the reuse instead, which is how the base
came to exist. A copy is the drift arriving before the second mode has run.)

**One board agent is one team brain**, as one TAG agent is one team brain
(user, 2026-09-13): upstream has no team layer, so its board is shared by
agents and ours by teams. `write` is issued under the team's name and `notify`
names teams.

This replaces `decentralized_messageboard`, which was our own design rather
than theirs (user, 2026-09-15). That one differed from `tag` in four ways at
once -- no notify without editing the source, no write or notify from an
interrupt round, the board truncated to its last 12 posts, and the board
*after* the observation rather than before -- so a `tag` vs board comparison
measured those as much as the shared space.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from coop2.cognitive.agent.llm_client import (
    LLMBoardInterruptResponse,
    LLMBoardPlanResponse,
)
from coop2.comm_topology.notifying_team import (
    DEFAULT_NOTIFY_BUDGET,
    NOTIFY_MESSAGE_TYPE,
    NotifyingTeamBrain,
    rounds_of,
)

__all__ = [
    "DEFAULT_NOTIFY_BUDGET",
    "NOTIFY_MESSAGE_TYPE",
    "BOARD_MANUAL",
    "BoardTeamBrain",
    "SharedBoard",
    "board_rounds_of",
    "format_board_observation",
    "new_shared_board",
    "save_board_outputs",
    "shared_board",
]

# NOTIFY_MESSAGE_TYPE and DEFAULT_NOTIFY_BUDGET are the shared round's, so a
# notification from either mode is the same message under the same budget.

#: Section 5, from upstream's MESSAGE_BOARD_ROLE, said to a team controller
#: rather than to a robot. One tool against the graph's eight: that asymmetry
#: is the ablation, not an omission.
BOARD_MANUAL = """The teams coordinate through one shared message board: every post, in the order it was made, with its author and the environment step. You are shown its complete history each time you plan or are notified.
write, the board's one tool:
- write(text): a message for the board, or null to write nothing.

Say what you are doing, what you found, or what you need from others; every team reads it. Keep it short and useful. The board has no structure and nothing is retracted: a post stays as written, and a claim someone made earlier is only as true as it was then."""

class SharedBoard:
    """An unstructured shared board: posts in order, each with its author and
    the environment step.

    Upstream's `SharedMessageBoard`, kept to the letter -- a lock around the
    writes, because teams plan on threads, and `history()` hands out copies so
    a reader cannot edit what it read.
    """

    def __init__(self) -> None:
        self._posts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, author: str, text: str, env_step: int) -> Dict[str, Any]:
        """The board's one tool: a message, posted with its author and step."""
        with self._lock:
            entry = {"index": len(self._posts), "author": author, "text": text,
                     "env_step": env_step, "at": time.time()}
            self._posts.append(entry)
            return dict(entry)

    def history(self) -> List[Dict[str, Any]]:
        """Every post so far, in order; copies."""
        with self._lock:
            return [dict(entry) for entry in self._posts]

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "message_board.v1", "posts": self.history()}

    def save_json(self, path: "str | Path") -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def new_shared_board() -> SharedBoard:
    """The team's board, empty at the start of the episode."""
    return SharedBoard()


def format_board_observation(history: List[Dict[str, Any]], team_names: List[str],
                             budget_left: int) -> str:
    """The board as the prompt shows it: **the complete history**.

    Not a window. Upstream shows every post and so does this, because the
    board's only structure is that nothing is lost: a team that cannot see an
    old claim cannot honour it, and truncating would hand `tag` an advantage
    the ablation is not supposed to be testing.
    """
    lines = [f"Shared message board (shared by {', '.join(team_names)}), complete history:"]
    if history:
        for entry in history:
            lines.append(f"  [{entry['index']}] step {entry['env_step']} {entry['author']}: {entry['text']}")
    else:
        lines.append("  (nothing posted yet)")
    lines.append(f"Notifications left until the next environment step: {budget_left} "
                 "(one notification may name several teams)")
    return "\n".join(lines)


class BoardTeamBrain(NotifyingTeamBrain):
    """One team on the shared message board -- `board`.

    The round is the base's, shared with `tag`; what is here is the board.
    """

    COOPERATION_MODE = "board"
    SPACE_NAME = "board"
    MANUAL = BOARD_MANUAL
    NOTIFY_TEXT = "read the shared message board"
    PLAN_RESPONSE = LLMBoardPlanResponse
    INTERRUPT_RESPONSE = LLMBoardInterruptResponse

    def __init__(self, *args, board: Optional[SharedBoard] = None, **kwargs):
        super().__init__(*args, **kwargs)
        #: Shared with every other brain in the run by the factory; a brain
        #: built alone gets a board of its own so it still works.
        self.board = board if board is not None else new_shared_board()

    # -- the board, in the six seams ---------------------------------------

    def _observe_space(self) -> List[Dict[str, Any]]:
        return self.board.history()

    def _space_section(self, observation: List[Dict[str, Any]]) -> str:
        return format_board_observation(observation, self._team_names(), self.notify_left)

    def _observed(self, observation: List[Dict[str, Any]]) -> List[str]:
        return [str(entry["index"]) for entry in observation]

    def _record_fields(self) -> Dict[str, Any]:
        return {"write": None}

    def _apply(self, response: Any, record: Dict[str, Any]) -> None:
        """A non-empty write lands on the board. There is nothing to reject:
        the board takes any text, which is the point of the ablation -- the
        graph refuses an action against a stale version and the board cannot.
        """
        text = (getattr(response, "write", None) or "").strip()
        if text:
            record["write"] = self.board.write(self.team_name, text, self._env_step())["text"]

    def _summary(self, record: Dict[str, Any]) -> str:
        return "wrote" if record["write"] else "no write"

    # -- the closing lines -------------------------------------------------

    def _plan_closing(self, members) -> str:
        return (super()._plan_closing(members)
                + " `write`: the message the board needs (null if none); "
                  "`notify`: teams that must read it now (usually empty).")

    def _interrupt_closing(self) -> str:
        return (super()._interrupt_closing()
                + " Also `write` (null if none) and `notify` (usually empty).")


# -- what a run saves ------------------------------------------------------

def shared_board(agents: Dict[str, Any]) -> SharedBoard:
    """The one board a `board` run shares."""
    boards = {id(a.brain.board): a.brain.board for a in agents.values()
              if isinstance(getattr(a, "brain", None), BoardTeamBrain)}
    if len(boards) != 1:
        raise ValueError("not one message-board run: expected every team to share one board")
    return next(iter(boards.values()))


def board_rounds_of(agents: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every team's rounds, in the order they happened."""
    return rounds_of(agents)


def save_board_outputs(agents: Dict[str, Any], output_dir: str) -> List[str]:
    """Write board.json and board_rounds.json; return one line per file for the
    runner to print. The board has no drawing: it is a list, and `tag`'s figure
    exists because a graph is not."""
    board = shared_board(agents)
    lines = []
    path = board.save_json(os.path.join(output_dir, "board.json"))
    lines.append(f"Message board saved to {path}")
    rounds_path = os.path.join(output_dir, "board_rounds.json")
    with open(rounds_path, "w", encoding="utf-8") as handle:
        json.dump(board_rounds_of(agents), handle, indent=2)
    lines.append(f"Message-board rounds saved to {rounds_path}")
    return lines
