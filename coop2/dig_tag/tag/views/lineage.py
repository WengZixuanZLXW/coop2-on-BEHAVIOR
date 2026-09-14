"""Lineage analysis of the TAG view: exact-version ancestry through the
task actions. Lineage follows L_T only -- which versions an action consumed
to return this one -- never the DIG's interaction flow."""

from __future__ import annotations

from typing import List

from ..graph import TAGGraph
from ..version import Version


def lineage(graph: TAGGraph, task: Version) -> List[Version]:
    """The versions this one descends from, nearest first: the inputs of the
    action that returned it, then theirs. An injected root has none."""
    producing = {
        task_id: action for action in graph.actions.values() for task_id in action.output_ids
    }
    out: List[Version] = []
    seen = {task.id}
    queue = [task]
    while queue:
        current = queue.pop(0)
        action = producing.get(current.id)
        if action is None:
            continue
        for task_id in action.input_ids:
            if task_id in seen:
                continue
            seen.add(task_id)
            node = graph.tasks[task_id]
            out.append(node)
            queue.append(node)
    return out


def descendants(graph: TAGGraph, task: Version) -> List[Version]:
    """The versions derived from this one, nearest first: the outputs of the
    actions that consumed it, then theirs."""
    consuming: dict = {}
    for action in graph.actions.values():
        for task_id in action.input_ids:
            consuming.setdefault(task_id, []).append(action)
    out: List[Version] = []
    seen = {task.id}
    queue = [task]
    while queue:
        current = queue.pop(0)
        for action in consuming.get(current.id, ()):
            for task_id in action.output_ids:
                if task_id in seen:
                    continue
                seen.add(task_id)
                node = graph.tasks[task_id]
                out.append(node)
                queue.append(node)
    return out
