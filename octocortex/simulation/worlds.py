from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorldSpec:
    size: tuple[int, int]
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    seed: int | None = None

    def state(self, agent: tuple[int, int] | None = None) -> dict:
        return {
            "size": list(self.size),
            "agent": list(agent if agent is not None else self.start),
            "goal": list(self.goal),
            "obstacles": [list(cell) for cell in sorted(self.obstacles)],
        }

    @property
    def fingerprint(self) -> tuple:
        return self.size, self.start, self.goal, tuple(sorted(self.obstacles))


DEFAULT_WORLD = WorldSpec(
    size=(12, 8),
    start=(0, 0),
    goal=(11, 7),
    obstacles=frozenset({
        (2, 0), (2, 1), (2, 2), (4, 2), (5, 2), (6, 2),
        (6, 3), (6, 4), (8, 4), (9, 4), (10, 4), (8, 6),
    }),
)


def generate_world(seed: int, size: tuple[int, int] = (12, 8), density: float = 0.18) -> WorldSpec:
    """Create a reproducible unseen world with at least one monotonic route."""
    rng = random.Random(seed)
    width, height = size
    corners = ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
    corner_index = seed % len(corners)
    start = corners[corner_index]
    goal = corners[3 - corner_index]

    x, y = start
    route = {(x, y), goal}
    while (x, y) != goal:
        choices = []
        if x != goal[0]:
            choices.append((x + (1 if goal[0] > x else -1), y))
        if y != goal[1]:
            choices.append((x, y + (1 if goal[1] > y else -1)))
        x, y = rng.choice(choices)
        route.add((x, y))

    obstacles = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in route and rng.random() < density
    }
    # Keep the benchmark focused on policy transfer rather than maze search:
    # every free cell has at least one free neighbor closer to the goal.
    cells = [(x, y) for y in range(height) for x in range(width)]
    distance = lambda cell: abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
    for cell in sorted(cells, key=distance):
        if cell == goal or cell in obstacles:
            continue
        closer = [
            candidate for candidate in (
                (cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]),
                (cell[0], cell[1] - 1), (cell[0], cell[1] + 1),
            )
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height
            and distance(candidate) < distance(cell)
        ]
        if closer and all(candidate in obstacles for candidate in closer):
            obstacles.remove(rng.choice(closer))
    return WorldSpec(size, start, goal, frozenset(obstacles), seed)


TRAIN_WORLDS = tuple(generate_world(1_000 + seed) for seed in range(32))
TEST_WORLDS = tuple(generate_world(10_000 + seed) for seed in range(100))


def generate_detour_world(seed: int, size: tuple[int, int] = (12, 8), density: float = 0.10) -> WorldSpec:
    """Create a world whose shortest route must initially move away from the goal."""
    rng = random.Random(seed)
    width, height = size
    top_start = seed % 2 == 0
    start = (0, 0 if top_start else height - 1)
    goal = (width - 1, start[1])
    gap_y = height - 1 if top_start else 0
    wall_x = 4 + seed % max(1, width - 8)
    barrier = {(wall_x, y) for y in range(height) if y != gap_y}
    guaranteed_route = (
        {(0, y) for y in range(min(start[1], gap_y), max(start[1], gap_y) + 1)}
        | {(x, gap_y) for x in range(width)}
        | {(width - 1, y) for y in range(min(goal[1], gap_y), max(goal[1], gap_y) + 1)}
    )
    obstacles = set(barrier)
    for y in range(height):
        for x in range(width):
            cell = (x, y)
            if cell not in guaranteed_route and cell not in obstacles and rng.random() < density:
                obstacles.add(cell)
    return WorldSpec(size, start, goal, frozenset(obstacles), seed)


HARD_TRAIN_WORLDS = tuple(generate_detour_world(20_000 + seed) for seed in range(32))
HARD_TEST_WORLDS = tuple(generate_detour_world(30_000 + seed) for seed in range(100))
