from __future__ import annotations

from .arbitrator import Arbitrator
from .models import Decision, Event, Proposal


class GlobalWorkspace:
    """Integrates salient broadcasts and resolves competing local policies."""

    def __init__(self) -> None:
        self.arbitrator = Arbitrator()
        self.attention: list[Event] = []
        self.global_energy = 0.0

    def integrate(self, events: list[Event], proposals: list[Proposal]) -> Decision:
        self.attention = sorted(events, key=lambda event: event.salience, reverse=True)[:5]
        self.global_energy += 0.1 + (0.16 * len(events)) + (0.04 * len(proposals))
        return self.arbitrator.choose(proposals)

