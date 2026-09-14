"""World AABBs, memoised for the length of one simulation tick.

``support_of`` answers "what is this object resting on?" by scanning **every**
object in the scene and taking each one's world AABB, and it is asked once per
entity per agent while an observation is built. On Merom (654 objects, nine
robots) one macro step recomputed the same few hundred AABBs hundreds of times:
measured 2026-09-14, 2277 ``_aabb_of`` calls per three macro steps, 85% of the
whole cost of building an observation, with 0.32 s of that inside
``torch.tensor`` alone, because ``entity_prim.aabb`` rebuilds collision points
in world frame on every call. Memoised, a macro step for nine robots went from
606 ms to 41 ms.

An AABB cannot change while nothing moves, so the cache is valid for exactly
one tick. :func:`invalidate_aabb_cache` is called by the engine after every
``env.step`` and by the few places that teleport a body between ticks.

This lives in its own module, free of any omnigibson import, so the engine can
clear the cache without depending on the controller stack -- the stubbed CPU
tests replace omnigibson wholesale and would not survive that import.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

__all__ = ["aabb_cache", "invalidate_aabb_cache"]

Box = Tuple[List[float], List[float]]

_AABB_CACHE: Dict[Any, Box] = {}


def aabb_cache() -> Dict[Any, Box]:
    """The live cache, keyed by object name (or ``id(obj)`` when unnamed)."""
    return _AABB_CACHE


def invalidate_aabb_cache() -> None:
    """Drop every cached AABB. Call whenever a body may have moved."""
    _AABB_CACHE.clear()
