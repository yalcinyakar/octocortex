from __future__ import annotations

from collections import deque

from .models import Event


class SparseEventBus:
    """Publishes only events whose salience clears a configurable threshold."""

    def __init__(self, threshold: float = 0.55, history_size: int = 100) -> None:
        self.threshold = threshold
        self._pending: deque[Event] = deque()
        self.history: deque[Event] = deque(maxlen=history_size)
        self.observations = 0
        self.published = 0

    def observe(self, event: Event) -> bool:
        self.observations += 1
        if event.salience < self.threshold:
            return False
        self._pending.append(event)
        self.history.append(event)
        self.published += 1
        return True

    def drain(self) -> list[Event]:
        events = list(self._pending)
        self._pending.clear()
        return events

    @property
    def sparsity(self) -> float:
        if not self.observations:
            return 1.0
        return 1.0 - (self.published / self.observations)

