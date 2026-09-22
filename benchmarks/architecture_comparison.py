"""Reproducible architecture benchmark for OctoCortex and two baselines."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass

from octocortex.core.models import ActionCode
from octocortex.core.octoir import UnitCode
from octocortex.simulation.environment import MOVES, OctoSimulation


SIZE = (12, 8)
START = (0, 0)
GOAL = (11, 7)
OBSTACLES = {
    (2, 0), (2, 1), (2, 2), (4, 2), (5, 2), (6, 2),
    (6, 3), (6, 4), (8, 4), (9, 4), (10, 4), (8, 6),
}
ACTIONS = tuple(MOVES)
FAILURE_UNITS = (UnitCode.PERCEPTION, UnitCode.MEMORY, UnitCode.PLANNING)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    sensor_noise: float
    failure_probability: float


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    architecture: str
    scenario: str
    seed: int
    success: bool
    steps: int
    collisions: int
    energy_proxy: float
    runtime_us: float
    failed_component: str | None


SCENARIOS = (
    Scenario("clean", 0.0, 0.0),
    Scenario("sensor_noise", 0.15, 0.0),
    Scenario("component_failure", 0.0, 0.35),
    Scenario("combined", 0.15, 0.35),
)


def _distance(cell: tuple[int, int]) -> int:
    return abs(cell[0] - GOAL[0]) + abs(cell[1] - GOAL[1])


def _next_cells(agent: tuple[int, int]) -> dict[ActionCode, tuple[int, int]]:
    return {action: (agent[0] + dx, agent[1] + dy) for action, (dx, dy) in MOVES.items()}


def _perceived_obstacles(agent: tuple[int, int], noise: float, rng: random.Random) -> set[tuple[int, int]]:
    perceived = set(OBSTACLES)
    for cell in _next_cells(agent).values():
        if 0 <= cell[0] < SIZE[0] and 0 <= cell[1] < SIZE[1] and rng.random() < noise:
            perceived.symmetric_difference_update({cell})
    return perceived


def _valid_moves(agent: tuple[int, int], perceived: set[tuple[int, int]]) -> dict[ActionCode, tuple[int, int]]:
    return {
        action: cell for action, cell in _next_cells(agent).items()
        if 0 <= cell[0] < SIZE[0] and 0 <= cell[1] < SIZE[1] and cell not in perceived
    }


def _greedy(valid: dict[ActionCode, tuple[int, int]]) -> ActionCode:
    return min(valid, key=lambda action: (_distance(valid[action]), int(action))) if valid else ActionCode.WAIT


def _failure_for(seed: int, probability: float) -> UnitCode | None:
    rng = random.Random(seed ^ 0x5EED)
    return rng.choice(FAILURE_UNITS) if rng.random() < probability else None


def run_centralized(scenario: Scenario, seed: int, max_steps: int = 80) -> EpisodeResult:
    rng = random.Random(seed)
    failed = _failure_for(seed, scenario.failure_probability)
    agent, collisions, energy = START, 0, 0.0
    started = time.perf_counter_ns()
    for step in range(1, max_steps + 1):
        energy += 0.62
        if failed is not None:
            action = ActionCode.WAIT
        else:
            action = _greedy(_valid_moves(agent, _perceived_obstacles(agent, scenario.sensor_noise, rng)))
        if action in MOVES:
            target = _next_cells(agent)[action]
            if target in OBSTACLES or not (0 <= target[0] < SIZE[0] and 0 <= target[1] < SIZE[1]):
                collisions += 1
            else:
                agent = target
        if agent == GOAL:
            break
    runtime = (time.perf_counter_ns() - started) / 1_000
    return EpisodeResult("centralized", scenario.name, seed, agent == GOAL, step, collisions, round(energy, 4), runtime, failed.name.lower() if failed else None)


def run_distributed(scenario: Scenario, seed: int, max_steps: int = 80) -> EpisodeResult:
    rng = random.Random(seed)
    failed = _failure_for(seed, scenario.failure_probability)
    agent, collisions, energy = START, 0, 0.0
    collision_memory: Counter[tuple[int, int]] = Counter()
    started = time.perf_counter_ns()
    for step in range(1, max_steps + 1):
        perceived = _perceived_obstacles(agent, scenario.sensor_noise, rng)
        valid = _valid_moves(agent, perceived)
        votes: list[ActionCode] = []
        if failed != UnitCode.PLANNING:
            votes.append(_greedy(valid)); energy += 0.28
        if failed != UnitCode.PERCEPTION:
            safest = min(valid, key=lambda action: (
                sum(neighbor in perceived for neighbor in _next_cells(valid[action]).values()),
                _distance(valid[action]), int(action),
            )) if valid else ActionCode.WAIT
            votes.append(safest); energy += 0.24
        if failed != UnitCode.MEMORY:
            remembered = {action: cell for action, cell in valid.items() if not collision_memory[cell]}
            votes.append(_greedy(remembered or valid)); energy += 0.18
        counts = Counter(votes)
        action = min(counts, key=lambda candidate: (-counts[candidate], _distance(_next_cells(agent).get(candidate, agent)), int(candidate))) if counts else ActionCode.WAIT
        if action in MOVES:
            target = _next_cells(agent)[action]
            if target in OBSTACLES or not (0 <= target[0] < SIZE[0] and 0 <= target[1] < SIZE[1]):
                collisions += 1; collision_memory[target] += 1
            else:
                agent = target
        if agent == GOAL:
            break
    runtime = (time.perf_counter_ns() - started) / 1_000
    return EpisodeResult("distributed", scenario.name, seed, agent == GOAL, step, collisions, round(energy, 4), runtime, failed.name.lower() if failed else None)


def run_octocortex(scenario: Scenario, seed: int, max_steps: int = 80) -> EpisodeResult:
    failed = _failure_for(seed, scenario.failure_probability)
    disabled = frozenset({failed}) if failed else frozenset()
    simulation = OctoSimulation(seed=seed, sensor_noise=scenario.sensor_noise, disabled_units=disabled)
    started = time.perf_counter_ns()
    state = simulation.snapshot()
    for step in range(1, max_steps + 1):
        state = simulation.step()
        if state["done"]:
            break
    runtime = (time.perf_counter_ns() - started) / 1_000
    energy = state["metrics"]["local_energy"] + state["metrics"]["global_energy"]
    return EpisodeResult("octocortex", scenario.name, seed, state["done"], step, state["collisions"], round(energy, 4), runtime, failed.name.lower() if failed else None)


def run_benchmark(trials: int = 50, max_steps: int = 80) -> dict:
    runners = (run_centralized, run_distributed, run_octocortex)
    episodes = [
        runner(scenario, seed, max_steps)
        for scenario in SCENARIOS
        for seed in range(trials)
        for runner in runners
    ]
    summary = []
    for scenario in SCENARIOS:
        for runner in runners:
            name = runner.__name__.removeprefix("run_")
            rows = [row for row in episodes if row.scenario == scenario.name and row.architecture == name]
            successes = [row for row in rows if row.success]
            summary.append({
                "scenario": scenario.name,
                "architecture": name,
                "success_rate": round(len(successes) / len(rows), 3),
                "mean_steps": round(statistics.fmean(row.steps for row in successes), 2) if successes else None,
                "mean_collisions": round(statistics.fmean(row.collisions for row in rows), 2),
                "mean_energy_proxy": round(statistics.fmean(row.energy_proxy for row in rows), 3),
                "median_runtime_us": round(statistics.median(row.runtime_us for row in rows), 2),
            })
    return {"trials_per_case": trials, "max_steps": max_steps, "summary": summary, "episodes": [asdict(row) for row in episodes]}


def print_table(report: dict) -> None:
    print("scenario           architecture  success  steps  collisions  energy  runtime_us")
    for row in report["summary"]:
        steps = "—" if row["mean_steps"] is None else f'{row["mean_steps"]:.2f}'
        print(f'{row["scenario"]:<18} {row["architecture"]:<13} {row["success_rate"]:>7.1%} {steps:>6} {row["mean_collisions"]:>11.2f} {row["mean_energy_proxy"]:>7.3f} {row["median_runtime_us"]:>11.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(args.trials, args.max_steps)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
