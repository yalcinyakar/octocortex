from __future__ import annotations

from collections import Counter

from octocortex.core.models import ActionCode, Proposal, ReasonCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.learning.adapter import SparseSemanticAdapter


class MemoryArm:
    unit = UnitCode.MEMORY

    def __init__(self, adapter: SparseSemanticAdapter) -> None:
        self.adapter = adapter
        self.bad_cells: Counter[tuple[int, int]] = Counter()
        self.energy = 0.0

    def remember_collision(self, cell: tuple[int, int]) -> None:
        self.bad_cells[cell] += 1

    def recall(self, next_cells: dict[ActionCode, tuple[int, int]], tick: int) -> tuple[list[SemanticPacket], list[Proposal]]:
        self.energy += 0.025
        risky = [(action, cell, self.bad_cells[cell]) for action, cell in next_cells.items() if self.bad_cells[cell]]
        if not risky:
            learned = self.adapter.encode_semantic(self.unit, ConceptCode.MEMORY_QUIET)
            return [SemanticPacket(
                source=self.unit, target=UnitCode.WORKSPACE,
                concept=ConceptCode.MEMORY_QUIET, tick=tick,
                confidence=0.8, salience=0.15, ttl_ms=100,
                latent=learned.latent,
            )], []
        action, cell, count = max(risky, key=lambda item: item[2])
        learned = self.adapter.encode_semantic(
            self.unit, ConceptCode.COLLISION_RECALL,
            (cell[0] / 11.0, cell[1] / 7.0, min(1.0, count / 5.0), float(action) / 4.0, min(0.98, 0.6 + 0.1 * count), 0.55),
        )
        return [SemanticPacket(
            source=self.unit, target=UnitCode.WORKSPACE,
            concept=ConceptCode.COLLISION_RECALL, tick=tick,
            confidence=min(0.95, 0.55 + 0.1 * count),
            salience=min(1.0, 0.55 + 0.12 * count),
            urgency=0.55, risk=min(0.98, 0.6 + 0.1 * count),
            ttl_ms=500,
            state_delta=(float(cell[0]), float(cell[1]), float(count), float(action)),
            latent=learned.latent,
        )], [Proposal(
            arm=self.unit,
            action=ActionCode.WAIT,
            confidence=min(0.95, 0.55 + 0.1 * count),
            risk=min(0.98, 0.6 + 0.1 * count),
            urgency=0.55,
            energy_cost=0.02,
            reason_code=ReasonCode.COLLISION_MEMORY,
            reason_value=float(count),
        )]
