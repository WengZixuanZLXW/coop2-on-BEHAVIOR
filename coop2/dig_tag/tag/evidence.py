"""Evidence attachment model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .version import Version


@dataclass(eq=False)
class Evidence:
    """phi = (s, x, q): a declared source, an arbitrary payload, and the
    EXACT task version it targets.

    The source may name an agent, an activation, an environment call, or any
    implementation-defined provenance; it does not certify the payload.
    Evidence stays attached to its exact target: a successor version of the
    task inherits nothing. `target` is the version id; a recorded
    `Version` is accepted and reduced to its id."""

    source: str
    payload: Any
    target: str
    id: Optional[str] = None
    created_at: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.target, Version):
            if self.target.id is None:
                raise ValueError("Evidence.target must be a RECORDED version")
            self.target = self.target.id
        if not isinstance(self.target, str) or not self.target:
            raise TypeError(
                f"Evidence.target must be a version id or a Version; got {self.target!r}"
            )
        if not isinstance(self.source, str) or not self.source:
            raise TypeError(
                f"Evidence.source must be a non-empty str; got {self.source!r}"
            )

    @property
    def recorded(self) -> bool:
        return self.id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "payload": self.payload,
            "target": self.target,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        return cls(
            source=data["source"],
            payload=data.get("payload"),
            target=data["target"],
            id=data.get("id"),
            created_at=data.get("created_at"),
        )

    def __str__(self) -> str:
        return (
            f"Evidence(id={self.id or '-'}, source={self.source}, "
            f"target={self.target}, payload={self.payload!r})"
        )
