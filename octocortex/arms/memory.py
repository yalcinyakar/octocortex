from __future__ import annotations

from collections import Counter

from octocortex.core.models import Event, Proposal


class MemoryArm:
    name = "memory"

    def __init__(self) -> None:
        self.bad_cells: Counter[tuple[int, int]] = Counter()
        self.energy = 0.0

    def remember_collision(self, cell: tuple[int, int]) -> None:
        self.bad_cells[cell] += 1

    def recall(self, next_cells: dict[str, tuple[int, int]], tick: int) -> tuple[list[Event], list[Proposal]]:
        self.energy += 0.025
        risky = [(action, cell, self.bad_cells[cell]) for action, cell in next_cells.items() if self.bad_cells[cell]]
        if not risky:
            return [Event(self.name, "memory_quiet", 0.15, {}, tick)], []
        action, cell, count = max(risky, key=lambda item: item[2])
        return [Event(self.name, "collision_recall", min(1.0, 0.55 + 0.12 * count), {
            "cell": list(cell), "count": count, "avoid_action": action,
        }, tick)], [Proposal(
            arm=self.name,
            action="WAIT",
            confidence=min(0.95, 0.55 + 0.1 * count),
            risk=min(0.98, 0.6 + 0.1 * count),
            urgency=0.55,
            energy_cost=0.02,
            reason=f"recalled {count} collision(s) near {cell}",
        )]

