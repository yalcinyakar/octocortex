from __future__ import annotations

from .arbitrator import Arbitrator
from .models import Decision, Proposal
from .octoir import SemanticPacket


class GlobalWorkspace:
    """Integrates salient broadcasts and resolves competing local policies."""

    def __init__(self) -> None:
        self.arbitrator = Arbitrator()
        self.attention: list[SemanticPacket] = []
        self.global_energy = 0.0

    def integrate(self, packets: list[SemanticPacket], proposals: list[Proposal]) -> Decision:
        self.attention = sorted(packets, key=lambda packet: packet.salience, reverse=True)[:5]
        self.global_energy += 0.1 + (0.16 * len(packets)) + (0.04 * len(proposals))
        return self.arbitrator.choose(proposals)
