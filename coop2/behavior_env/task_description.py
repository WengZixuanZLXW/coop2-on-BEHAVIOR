"""One sentence or three of natural language per activity, beside its BDDL.

``<activity>/description.txt`` next to ``problem0.bddl`` says what the task
is the way a person would put it (user, 2026-09-13): what the cargo is, where
it starts, the supports in order, where it ends. The prompt opens THE TEAM'S
TASK with it; the marked route or the BDDL goal follows in ids. Missing file
-> None, and the block is what it was before.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["description_file_for", "load_task_description"]


def description_file_for(activity: str) -> str:
    """``.../activity_definitions/<activity>/description.txt``, beside problem0.bddl,
    found through the same helper BDDL uses for the problem file."""
    from bddl.config import get_definition_filename  # noqa: PLC0415

    return os.path.join(os.path.dirname(get_definition_filename(activity, 0)), "description.txt")


def load_task_description(activity: Optional[str]) -> Optional[str]:
    if not activity:
        return None
    try:
        path = description_file_for(activity)
    except Exception:  # noqa: BLE001 - an activity BDDL does not know
        return None
    if not os.path.exists(path):
        return None
    text = " ".join(open(path, encoding="utf-8").read().split())
    return text or None
