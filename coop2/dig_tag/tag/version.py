"""Version: one exact task version, the information node of the TAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .spec import TaskSpec


@dataclass(eq=False)
class Version:
    """q = (k, delta, sigma): one exact task version.

    `identity` is the persistent task identity k (minted by OPEN, or by an
    injected root); `spec` is delta; `state` is sigma, the reported current
    state; `metadata` is free domain extras. Each version is a fresh token:
    distinct versions may carry identical fields, and a historical version
    is never edited -- EDIT and UPDATE return a new version. A version is
    T's alone: what a tool hands back about it is information, which D
    records as an event."""

    ID_PREFIX = "q"      # the ids minted for versions

    identity: Optional[str] = None
    spec: Optional[TaskSpec] = None
    state: Any = None
    id: Optional[str] = None
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TaskSpec):
            raise TypeError(f"Version.spec must be a TaskSpec; got {self.spec!r}")
        if self.identity is not None and (not isinstance(self.identity, str) or not self.identity):
            raise TypeError(f"Version.identity must be a non-empty str or None; got {self.identity!r}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"Version.metadata must be a dict; got {type(self.metadata).__name__}")

    @property
    def recorded(self) -> bool:
        """Whether the graph has taken this version (it has an id)."""
        return self.id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "identity": self.identity,
            "spec": self.spec.to_dict(),
            "state": self.state,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Version":
        return cls(
            identity=data.get("identity"),
            spec=TaskSpec.from_dict(data.get("spec") or {}),
            state=data.get("state"),
            id=data.get("id"),
            created_at=data.get("created_at"),
            metadata=dict(data.get("metadata") or {}),
        )

    def __str__(self) -> str:
        return (
            f"Version(id={self.id or '-'}, identity={self.identity or '-'}, "
            f"goal={self.spec.goal!r}, state={self.state!r})"
        )
