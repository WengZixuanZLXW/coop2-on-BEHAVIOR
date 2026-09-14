"""Sequential ids with a prefix."""

from __future__ import annotations

from typing import Container


class Minter:
    """Mints `<prefix><n>` in sequence, skipping an id already taken (a
    record loaded from disk may have spent a counter's value elsewhere).
    The counter is part of the record, so a loaded graph continues where
    it stopped."""

    def __init__(self, prefix: str, counter: int = 0) -> None:
        self.prefix = prefix
        self.counter = counter

    def next(self, taken: Container[str]) -> str:
        self.counter += 1
        while f"{self.prefix}{self.counter}" in taken:
            self.counter += 1
        return f"{self.prefix}{self.counter}"
