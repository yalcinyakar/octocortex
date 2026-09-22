"""Train/test transfer benchmark for distilled OctoCortex backup policies."""

from __future__ import annotations

import argparse
import json
import statistics

from octocortex.core.octoir import UnitCode
from octocortex.core.capabilities import CapabilityCode
from octocortex.learning.backup_planner import DistilledBackupPlanner
from octocortex.simulation.environment import OctoSimulation
from octocortex.simulation.worlds import TEST_WORLDS, TRAIN_WORLDS, WorldSpec


def evaluate_worlds(owner: UnitCode, worlds: tuple[WorldSpec, ...], max_steps: int = 80) -> dict:
    outcomes = []
    for world in worlds:
        simulation = OctoSimulation(
            seed=world.seed or 0,
            world=world,
            disabled_units=frozenset({UnitCode.PLANNING}),
        )
        planner = simulation.backup_planners[owner]
        simulation.capabilities.routes[CapabilityCode.PLAN_ROUTE] = (
            UnitCode.PLANNING, owner,
        )
        correct = total = 0
        for step in range(1, max_steps + 1):
            state = simulation._perceived_state()
            correct += planner.choose(state) == planner._teacher(state)
            total += 1
            snapshot = simulation.step()
            if snapshot["done"]:
                break
        outcomes.append((snapshot["done"], step, snapshot["collisions"], correct / total))
    successes = [result for result in outcomes if result[0]]
    return {
        "owner": owner.name.lower(),
        "worlds": len(worlds),
        "success_rate": round(len(successes) / len(outcomes), 3),
        "mean_steps": round(statistics.fmean(result[1] for result in successes), 2) if successes else None,
        "mean_collisions": round(statistics.fmean(result[2] for result in outcomes), 3),
        "action_agreement": round(statistics.fmean(result[3] for result in outcomes), 3),
    }


def run_generalization(train_count: int = 32, test_count: int = 100, max_steps: int = 80) -> dict:
    train_worlds = TRAIN_WORLDS[:train_count]
    test_worlds = TEST_WORLDS[:test_count]
    if {world.fingerprint for world in train_worlds} & {world.fingerprint for world in test_worlds}:
        raise RuntimeError("training and test worlds overlap")
    return {
        "train_worlds": len(train_worlds),
        "test_worlds": len(test_worlds),
        "split_overlap": 0,
        "results": [
            {"split": split, **evaluate_worlds(owner, worlds, max_steps)}
            for split, worlds in (("train", train_worlds), ("unseen", test_worlds))
            for owner in (UnitCode.MEMORY, UnitCode.PERCEPTION)
        ],
    }


def print_table(report: dict) -> None:
    print("split   owner       worlds  success  steps  collisions  action_agreement")
    for row in report["results"]:
        print(
            f'{row["split"]:<7} {row["owner"]:<11} {row["worlds"]:>6} '
            f'{row["success_rate"]:>7.1%} {row["mean_steps"]:>6.2f} '
            f'{row["mean_collisions"]:>11.3f} {row["action_agreement"]:>17.1%}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-worlds", type=int, default=32)
    parser.add_argument("--test-worlds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_generalization(args.train_worlds, args.test_worlds, args.max_steps)
    print(json.dumps(result, indent=2) if args.json else "") if args.json else print_table(result)
