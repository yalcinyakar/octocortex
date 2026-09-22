from __future__ import annotations

import math
import random
from typing import ClassVar

from octocortex.core.models import ActionCode, Proposal, ReasonCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.learning.adapter import SparseSemanticAdapter


MOVES = {
    ActionCode.UP: (0, -1),
    ActionCode.DOWN: (0, 1),
    ActionCode.LEFT: (-1, 0),
    ActionCode.RIGHT: (1, 0),
}


class DistilledBackupPlanner:
    """Independent linear policy distilled from route-planning examples."""

    _distillation_cache: ClassVar[dict[tuple, tuple[list[float], int, float]]] = {}

    def __init__(self, owner: UnitCode, seed: int, learning_rate: float = 0.12) -> None:
        self.owner = owner
        self.learning_rate = learning_rate
        rng = random.Random(seed)
        self.weights = [rng.uniform(-0.02, 0.02) for _ in range(5)]
        self.training_examples = 0
        self.training_accuracy = 0.0
        self.energy = 0.0

    @staticmethod
    def _cells(agent: tuple[int, int]) -> dict[ActionCode, tuple[int, int]]:
        return {action: (agent[0] + dx, agent[1] + dy) for action, (dx, dy) in MOVES.items()}

    @staticmethod
    def _distance(cell: tuple[int, int], goal: tuple[int, int]) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    def _features(self, state: dict, action: ActionCode) -> tuple[float, ...]:
        agent, goal = tuple(state["agent"]), tuple(state["goal"])
        width, height = state["size"]
        obstacles = {tuple(cell) for cell in state["obstacles"]}
        cell = self._cells(agent)[action]
        valid = 0 <= cell[0] < width and 0 <= cell[1] < height and cell not in obstacles
        progress = (self._distance(agent, goal) - self._distance(cell, goal)) / max(1, width + height - 2)
        clearance = sum(
            (cell[0] + dx, cell[1] + dy) not in obstacles
            and 0 <= cell[0] + dx < width and 0 <= cell[1] + dy < height
            for dx, dy in MOVES.values()
        ) / 4.0
        dx, dy = MOVES[action]
        alignment = (
            dx * (1 if goal[0] > agent[0] else -1 if goal[0] < agent[0] else 0)
            + dy * (1 if goal[1] > agent[1] else -1 if goal[1] < agent[1] else 0)
        ) / 2.0
        return (1.0, 1.0 if valid else -1.0, progress, clearance, alignment)

    def _score(self, features: tuple[float, ...]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features))

    def choose(self, state: dict) -> ActionCode:
        return max(MOVES, key=lambda action: (self._score(self._features(state, action)), -int(action)))

    def _teacher(self, state: dict) -> ActionCode:
        agent, goal = tuple(state["agent"]), tuple(state["goal"])
        width, height = state["size"]
        obstacles = {tuple(cell) for cell in state["obstacles"]}
        valid = {
            action: cell for action, cell in self._cells(agent).items()
            if 0 <= cell[0] < width and 0 <= cell[1] < height and cell not in obstacles
        }
        return min(valid, key=lambda action: (self._distance(valid[action], goal), int(action))) if valid else ActionCode.WAIT

    def distill(self, world: dict, epochs: int = 24) -> None:
        width, height = world["size"]
        obstacles = {tuple(cell) for cell in world["obstacles"]}
        cache_key = (
            int(self.owner), width, height, tuple(world["goal"]),
            tuple(sorted(obstacles)), epochs,
        )
        if cache_key in self._distillation_cache:
            weights, self.training_examples, self.training_accuracy = self._distillation_cache[cache_key]
            self.weights = list(weights)
            return
        examples = [
            {**world, "agent": [x, y]}
            for y in range(height) for x in range(width)
            if (x, y) not in obstacles and (x, y) != tuple(world["goal"])
        ]
        rng = random.Random(1000 + int(self.owner))
        for _ in range(epochs):
            rng.shuffle(examples)
            for state in examples:
                teacher = self._teacher(state)
                predicted = self.choose(state)
                if teacher is ActionCode.WAIT or predicted == teacher:
                    continue
                teacher_features = self._features(state, teacher)
                predicted_features = self._features(state, predicted)
                for index in range(len(self.weights)):
                    self.weights[index] += self.learning_rate * (
                        teacher_features[index] - predicted_features[index]
                    )
        self.training_examples = len(examples) * epochs
        self.training_accuracy = sum(self.choose(state) == self._teacher(state) for state in examples) / len(examples)
        self._distillation_cache[cache_key] = (
            list(self.weights), self.training_examples, self.training_accuracy,
        )

    def propose(
        self, state: dict, tick: int, adapter: SparseSemanticAdapter,
    ) -> tuple[list[SemanticPacket], list[Proposal], dict[ActionCode, tuple[int, int]]]:
        cells = self._cells(tuple(state["agent"]))
        action = self.choose(state)
        ranked = sorted((self._score(self._features(state, candidate)), candidate) for candidate in MOVES)
        margin = ranked[-1][0] - ranked[-2][0]
        confidence = min(0.92, 0.62 + 0.2 * math.tanh(max(0.0, margin)))
        cell = cells[action]
        distance = self._distance(cell, tuple(state["goal"]))
        self.energy += 0.022
        learned = adapter.encode_semantic(
            self.owner, ConceptCode.ROUTE_PROPOSAL,
            (float(action) / 4.0, cell[0] / 11.0, cell[1] / 7.0, distance / 18.0, confidence, 0.76),
        )
        packet = SemanticPacket(
            source=self.owner, target=UnitCode.WORKSPACE,
            concept=ConceptCode.ROUTE_PROPOSAL, tick=tick,
            confidence=confidence, uncertainty=1.0 - confidence,
            salience=0.46, urgency=0.3, risk=0.08,
            expected_reward=0.76, ttl_ms=250,
            state_delta=(float(action), float(cell[0]), float(cell[1]), float(distance)),
            latent=learned.quantized_latent,
            semantic_code=learned.semantic_code,
            quantization_error=learned.quantization_error,
            flags=2,
        )
        proposal = Proposal(
            arm=self.owner, action=action, confidence=confidence,
            expected_reward=0.76, risk=0.08, urgency=0.3,
            energy_cost=0.04, reason_code=ReasonCode.GOAL_PROGRESS,
            reason_value=float(distance),
        )
        return [packet], [proposal], cells
