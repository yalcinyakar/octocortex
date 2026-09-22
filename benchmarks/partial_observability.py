"""Benchmark recurrent world-model value under local-only sensing."""

from __future__ import annotations

import argparse
import json
import statistics

from octocortex.simulation.environment import OctoSimulation
from octocortex.simulation.worlds import HARD_TEST_WORLDS


def run_condition(name: str, enable_prediction: bool, worlds: int, max_steps: int) -> dict:
    outcomes = []
    for world in HARD_TEST_WORLDS[:worlds]:
        simulation = OctoSimulation(
            seed=world.seed or 0,
            world=world,
            partial_observability=True,
            enable_prediction=enable_prediction,
        )
        for step in range(1, max_steps + 1):
            state = simulation.step()
            if state["done"]:
                break
        outcomes.append((state["done"], step, state["collisions"], state["metrics"]["world_model_coverage"]))
    successes = [outcome for outcome in outcomes if outcome[0]]
    return {
        "condition": name,
        "worlds": len(outcomes),
        "success_rate": round(len(successes) / len(outcomes), 3),
        "mean_steps": round(statistics.fmean(row[1] for row in successes), 2) if successes else None,
        "mean_collisions": round(statistics.fmean(row[2] for row in outcomes), 3),
        "mean_coverage": round(statistics.fmean(row[3] for row in outcomes), 3),
    }


def run_benchmark(worlds: int = 100, max_steps: int = 120) -> dict:
    return {
        "sensor_radius": 1,
        "full_map_available": False,
        "results": [
            run_condition("local_only", False, worlds, max_steps),
            run_condition("octocortex_world_model", True, worlds, max_steps),
        ],
    }


def print_table(report: dict) -> None:
    print("condition                worlds  success  steps  collisions  coverage")
    for row in report["results"]:
        print(
            f'{row["condition"]:<24} {row["worlds"]:>6} {row["success_rate"]:>7.1%} '
            f'{row["mean_steps"] if row["mean_steps"] is not None else "—":>6} '
            f'{row["mean_collisions"]:>11.3f} {row["mean_coverage"]:>9.1%}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(args.worlds, args.max_steps)
    print(json.dumps(report, indent=2) if args.json else "") if args.json else print_table(report)
