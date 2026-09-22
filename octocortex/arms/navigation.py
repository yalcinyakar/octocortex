from __future__ import annotations

from octocortex.core.models import ActionCode, Proposal, ReasonCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.learning.adapter import SparseSemanticAdapter


MOVES = {
    ActionCode.UP: (0, -1),
    ActionCode.DOWN: (0, 1),
    ActionCode.LEFT: (-1, 0),
    ActionCode.RIGHT: (1, 0),
}


class NavigationArm:
    unit = UnitCode.PLANNING

    def __init__(self, adapter: SparseSemanticAdapter) -> None:
        self.adapter = adapter
        self.energy = 0.0

    def next_cells(self, agent: tuple[int, int]) -> dict[ActionCode, tuple[int, int]]:
        x, y = agent
        return {action: (x + dx, y + dy) for action, (dx, dy) in MOVES.items()}

    def propose(self, state: dict, tick: int) -> tuple[list[SemanticPacket], list[Proposal], dict[ActionCode, tuple[int, int]]]:
        agent = tuple(state["agent"])
        goal = tuple(state["goal"])
        obstacles = {tuple(cell) for cell in state["obstacles"]}
        width, height = state["size"]
        cells = self.next_cells(agent)
        valid = {
            action: cell for action, cell in cells.items()
            if 0 <= cell[0] < width and 0 <= cell[1] < height and cell not in obstacles
        }
        self.energy += 0.035 + 0.008 * len(valid)
        if not valid:
            learned = self.adapter.encode_semantic(self.unit, ConceptCode.TRAPPED, (1.0, 1.0))
            return [SemanticPacket(
                source=self.unit, target=UnitCode.WORKSPACE,
                concept=ConceptCode.TRAPPED, tick=tick,
                confidence=1.0, salience=1.0, urgency=1.0, risk=1.0,
                ttl_ms=250,
                latent=learned.quantized_latent,
                semantic_code=learned.semantic_code,
                quantization_error=learned.quantization_error,
            )], [Proposal(
                self.unit, ActionCode.WAIT, 1.0, risk=1.0, urgency=1.0,
                reason_code=ReasonCode.NO_VALID_MOVE,
            )], cells

        current_distance = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])
        action, cell = min(valid.items(), key=lambda item: (
            abs(item[1][0] - goal[0]) + abs(item[1][1] - goal[1]), item[0].name
        ))
        new_distance = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        progress = current_distance - new_distance
        proposal = Proposal(
            arm=self.unit,
            action=action,
            confidence=0.78 if progress > 0 else 0.58,
            expected_reward=0.82 if progress > 0 else 0.25,
            risk=0.05,
            urgency=0.22,
            energy_cost=0.08,
            reason_code=ReasonCode.GOAL_PROGRESS,
            reason_value=float(new_distance),
        )
        learned = self.adapter.encode_semantic(
            self.unit, ConceptCode.ROUTE_PROPOSAL,
            (
                float(action) / 4.0, cell[0] / 11.0, cell[1] / 7.0,
                new_distance / 18.0, proposal.confidence,
                proposal.expected_reward,
            ),
        )
        return [SemanticPacket(
            source=self.unit, target=UnitCode.WORKSPACE,
            concept=ConceptCode.ROUTE_PROPOSAL, tick=tick,
            confidence=proposal.confidence, uncertainty=1.0 - proposal.confidence,
            salience=0.42, urgency=proposal.urgency,
            risk=proposal.risk, expected_reward=proposal.expected_reward,
            ttl_ms=250,
            state_delta=(float(action), float(cell[0]), float(cell[1]), float(new_distance)),
            latent=learned.quantized_latent,
            semantic_code=learned.semantic_code,
            quantization_error=learned.quantization_error,
        )], [proposal], cells
