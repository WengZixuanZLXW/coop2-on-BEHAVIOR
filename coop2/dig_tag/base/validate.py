"""Value-shaping leaf helpers: type contracts and JSON shaping."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def expect_type(value: Any, cls: type, *, what: str) -> None:
    """Raise the standard TypeError when `value` is not a `cls` instance."""
    if not isinstance(value, cls):
        raise TypeError(
            f"{what} expects a {cls.__name__}; got {type(value).__name__}"
        )


def is_instance_sequence(value: Any, cls: type) -> bool:
    """Whether `value` is a list/tuple whose items are all `cls` instances."""
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, cls) for item in value
    )


def jsonable(value: Any) -> Any:
    """Recursively shape a value into JSON-serializable primitives."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def natural_sort_key(name: str) -> list:
    """Sort key placing agent10 after agent9, not after agent1."""
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]
