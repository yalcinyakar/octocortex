from __future__ import annotations

from collections import deque

from .octoir import SemanticPacket


class SparseEventBus:
    """Publishes only events whose salience clears a configurable threshold."""

    def __init__(self, threshold: float = 0.55, history_size: int = 100) -> None:
        self.threshold = threshold
        self._pending: deque[SemanticPacket] = deque()
        self.history: deque[SemanticPacket] = deque(maxlen=history_size)
        self.observations = 0
        self.published = 0

    def observe(self, packet: SemanticPacket) -> bool:
        self.observations += 1
        if packet.salience < self.threshold:
            return False
        self._pending.append(packet)
        self.history.append(packet)
        self.published += 1
        return True

    def drain(self) -> list[SemanticPacket]:
        events = list(self._pending)
        self._pending.clear()
        return events

    @property
    def sparsity(self) -> float:
        if not self.observations:
            return 1.0
        return 1.0 - (self.published / self.observations)
