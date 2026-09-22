from __future__ import annotations

from octocortex.core.models import Event, Proposal
from octocortex.snn.network import DangerNetwork


class VisionArm:
    name = "vision"

    def __init__(self) -> None:
        self.network = DangerNetwork()
        self.energy = 0.0

    def inspect(self, state: dict, tick: int) -> tuple[list[Event], list[Proposal], dict]:
        x, y = state["agent"]
        obstacles = {tuple(cell) for cell in state["obstacles"]}
        adjacent = sum(
            (x + dx, y + dy) in obstacles
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        danger = adjacent / 2.0
        spike, potential = self.network.process(danger)
        self.energy += 0.04
        events = [
            Event(self.name, "danger_spike" if spike else "visual_scan", min(1.0, danger), {
                "adjacent_obstacles": adjacent,
                "spike": spike,
                "potential": round(potential, 3),
            }, tick)
        ]
        proposals: list[Proposal] = []
        if spike and adjacent:
            proposals.append(Proposal(
                arm=self.name,
                action="WAIT",
                confidence=0.86,
                risk=min(1.0, 0.72 + danger / 3),
                urgency=0.86,
                energy_cost=0.05,
                reason="local danger neuron spiked",
            ))
        return events, proposals, {"danger": danger, "spike": spike, "potential": potential}

