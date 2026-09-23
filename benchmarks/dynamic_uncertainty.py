"""Dynamic-world benchmark for stale-belief adaptation and uncertainty."""

from __future__ import annotations

import argparse
import json
import statistics

from octocortex.simulation.environment import OctoSimulation
from octocortex.simulation.worlds import DYNAMIC_TEST_WORLDS


def run_condition(name: str, stale_after: int | None, worlds: int, max_steps: int) -> dict:
    outcomes = []
    for world in DYNAMIC_TEST_WORLDS[:worlds]:
        simulation = OctoSimulation(
            seed=world.seed or 0,
            world=world,
            partial_observability=True,
            prediction_stale_after=stale_after,
            sensor_noise=0.02,
        )
        peak_volatility = 0.0
        for step in range(1, max_steps + 1):
            state = simulation.step()
            peak_volatility = max(peak_volatility, state["metrics"]["world_model_volatility"])
            if state["done"]:
                break
        outcomes.append((
            state["done"], step, state["collisions"],
            state["metrics"]["world_model_expired"], peak_volatility,
        ))
    successes = [row for row in outcomes if row[0]]
    return {
        "condition": name,
        "worlds": len(outcomes),
        "success_rate": round(len(successes) / len(outcomes), 3),
        "mean_steps": round(statistics.fmean(row[1] for row in successes), 2) if successes else None,
        "mean_collisions": round(statistics.fmean(row[2] for row in outcomes), 3),
        "mean_expired_cells": round(statistics.fmean(row[3] for row in outcomes), 2),
        "mean_peak_volatility": round(statistics.fmean(row[4] for row in outcomes), 3),
    }


def run_benchmark(worlds: int = 100, max_steps: int = 100) -> dict:
    return {"results": [
        run_condition("persistent_static_belief", None, worlds, max_steps),
        run_condition("adaptive_dynamic_belief", 2, worlds, max_steps),
    ]}


def print_table(report: dict) -> None:
    print("condition                   success  steps  collisions  expired  volatility")
    for row in report["results"]:
        steps = "—" if row["mean_steps"] is None else f'{row["mean_steps"]:.2f}'
        print(
            f'{row["condition"]:<27} {row["success_rate"]:>7.1%} {steps:>6} '
            f'{row["mean_collisions"]:>11.3f} {row["mean_expired_cells"]:>8.2f} '
            f'{row["mean_peak_volatility"]:>11.3f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(args.worlds, args.max_steps)
    print(json.dumps(report, indent=2) if args.json else "") if args.json else print_table(report)
