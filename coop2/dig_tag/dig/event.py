"""DIGEvent: one information event."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

# The two declared origins the substrate writes itself. The third kind of
# origin is an agent id; agents are not enumerated here.
ENVIRONMENT_ORIGIN = "environment"   # feedback returned by an environment call
EXTERNAL_ORIGIN = "external"         # a root event supplied from outside the run


@dataclass(eq=False)
class DIGEvent:
    """One information event: an immutable node with an arbitrary payload.

    Two provenances, two fields. `producer` is STRUCTURAL: the call-return
    pair that inserted the event, written by the record funnel and never by
    the caller (None once recorded means a root event). `origin` is
    DECLARED: who the event is said to come from -- an agent id,
    `ENVIRONMENT_ORIGIN`, `EXTERNAL_ORIGIN`, or whatever the returner
    declares. Neither certifies the payload.

    Compared by identity: the same object is one activation's return and a
    later activation's input, and the graph maps it to one node.
    """

    @dataclass(frozen=True)
    class Producer:
        """The call-return pair that inserted an event."""

        activation_id: str
        call_index: int

        def __post_init__(self) -> None:
            if not isinstance(self.activation_id, str) or not self.activation_id:
                raise TypeError(
                    "DIGEvent.Producer.activation_id must be a non-empty str; "
                    f"got {self.activation_id!r}"
                )
            if not isinstance(self.call_index, int) or self.call_index < 0:
                raise TypeError(
                    "DIGEvent.Producer.call_index must be a non-negative int; "
                    f"got {self.call_index!r}"
                )

        def __str__(self) -> str:
            return f"{self.activation_id}:{self.call_index}"

        @classmethod
        def from_dict(cls, data: Mapping[str, Any]) -> "DIGEvent.Producer":
            return cls(
                activation_id=data["activation_id"],
                call_index=int(data["call_index"]),
            )

    payload: Dict[str, Any] = field(default_factory=dict)
    origin: Optional[str] = None
    id: Optional[str] = None
    producer: Optional[Producer] = None
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise TypeError(
                f"DIGEvent.payload must be a dict; got {type(self.payload).__name__}"
            )
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"DIGEvent.metadata must be a dict; got {type(self.metadata).__name__}"
            )
        if self.origin is not None and not isinstance(self.origin, str):
            raise TypeError(
                f"DIGEvent.origin must be a str or None; got {self.origin!r}"
            )
        if self.producer is not None and not isinstance(self.producer, DIGEvent.Producer):
            raise TypeError(
                "DIGEvent.producer must be a DIGEvent.Producer or None; "
                f"got {self.producer!r}"
            )

    @property
    def recorded(self) -> bool:
        """Whether the record funnel has taken this event (it has an id)."""
        return self.id is not None

    @property
    def is_root(self) -> bool:
        """A recorded event with no producing call-return pair."""
        return self.recorded and self.producer is None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node; references (the producer) are by id."""
        return {
            "id": self.id,
            "payload": self.payload,
            "origin": self.origin,
            "producer": (
                {
                    "activation_id": self.producer.activation_id,
                    "call_index": self.producer.call_index,
                }
                if self.producer is not None else None
            ),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DIGEvent":
        """Rebuild a serialized event (`to_dict` output of this class)."""
        producer = data.get("producer")
        return cls(
            payload=dict(data.get("payload") or {}),
            origin=data.get("origin"),
            id=data.get("id"),
            producer=(
                DIGEvent.Producer.from_dict(producer) if producer is not None else None
            ),
            created_at=data.get("created_at"),
            metadata=dict(data.get("metadata") or {}),
        )

    def __str__(self) -> str:
        producer = str(self.producer) if self.producer is not None else "root"
        return (
            f"DIGEvent(id={self.id or '-'}, origin={self.origin or '-'}, "
            f"producer={producer}, payload={self.payload})"
        )
