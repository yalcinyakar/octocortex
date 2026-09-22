from __future__ import annotations

from octocortex.core.models import Event, Proposal


MOVES = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


class NavigationArm:
    name = "navigation"

    def __init__(self) -> None:
        self.energy = 0.0

    def next_cells(self, agent: tuple[int, int]) -> dict[str, tuple[int, int]]:
        x, y = agent
        return {action: (x + dx, y + dy) for action, (dx, dy) in MOVES.items()}

    def propose(self, state: dict, tick: int) -> tuple[list[Event], list[Proposal], dict[str, tuple[int, int]]]:
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
            return [Event(self.name, "trapped", 1.0, {}, tick)], [Proposal(
                self.name, "WAIT", 1.0, risk=1.0, urgency=1.0, reason="no valid move",
            )], cells

        current_distance = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])
        action, cell = min(valid.items(), key=lambda item: (
            abs(item[1][0] - goal[0]) + abs(item[1][1] - goal[1]), item[0]
        ))
        new_distance = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        progress = current_distance - new_distance
        proposal = Proposal(
            arm=self.name,
            action=action,
            confidence=0.78 if progress > 0 else 0.58,
            expected_reward=0.82 if progress > 0 else 0.25,
            risk=0.05,
            urgency=0.22,
            energy_cost=0.08,
            reason=f"{action.lower()} reduces goal distance to {new_distance}",
        )
        return [Event(self.name, "route_proposal", 0.42, {
            "action": action, "target": list(cell), "distance": new_distance,
        }, tick)], [proposal], cells

