from __future__ import annotations

from octocortex.core.models import ActionCode, Proposal, ReasonCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.learning.adapter import SparseSemanticAdapter
from octocortex.snn.network import DangerNetwork


class VisionArm:
    unit = UnitCode.PERCEPTION

    def __init__(self, adapter: SparseSemanticAdapter) -> None:
        self.adapter = adapter
        self.network = DangerNetwork()
        self.energy = 0.0

    def inspect(self, state: dict, tick: int) -> tuple[list[SemanticPacket], list[Proposal], dict]:
        x, y = state["agent"]
        obstacles = {tuple(cell) for cell in state["obstacles"]}
        adjacent = sum(
            (x + dx, y + dy) in obstacles
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        danger = adjacent / 2.0
        spike, potential = self.network.process(danger)
        self.energy += 0.04
        concept = ConceptCode.DANGER_SPIKE if spike else ConceptCode.VISUAL_SCAN
        learned = self.adapter.encode_semantic(
            self.unit, concept,
            (adjacent / 4.0, danger, potential, float(spike), min(1.0, danger), 0.86 if spike else 0.0),
        )
        packets = [
            SemanticPacket(
                source=self.unit,
                target=UnitCode.WORKSPACE,
                concept=concept,
                tick=tick,
                confidence=min(1.0, 0.55 + danger),
                uncertainty=max(0.0, 0.45 - danger / 2),
                salience=min(1.0, danger),
                urgency=0.86 if spike else 0.0,
                risk=min(1.0, danger),
                ttl_ms=180,
                state_delta=(float(adjacent),),
                latent=learned.latent,
                flags=1 if spike else 0,
            )
        ]
        proposals: list[Proposal] = []
        if spike and adjacent:
            proposals.append(Proposal(
                arm=self.unit,
                action=ActionCode.WAIT,
                confidence=0.86,
                risk=min(1.0, 0.72 + danger / 3),
                urgency=0.86,
                energy_cost=0.05,
                reason_code=ReasonCode.DANGER_SPIKE,
            ))
        return packets, proposals, {"danger": danger, "spike": spike, "potential": potential}
