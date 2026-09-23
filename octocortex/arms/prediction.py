from __future__ import annotations

from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.learning.adapter import SparseSemanticAdapter


class PredictionArm:
    """Recurrent occupancy belief built only from local observations."""

    unit = UnitCode.PREDICTION

    def __init__(self, adapter: SparseSemanticAdapter, stale_after: int | None = None) -> None:
        self.adapter = adapter
        self.known_obstacles: set[tuple[int, int]] = set()
        self.known_free: set[tuple[int, int]] = set()
        self.size = (0, 0)
        self.energy = 0.0
        self.stale_after = stale_after
        self.last_seen: dict[tuple[int, int], int] = {}
        self.last_value: dict[tuple[int, int], bool] = {}
        self.volatility = 0.0
        self.expired_cells = 0

    @property
    def observed_cells(self) -> int:
        return len(self.known_obstacles | self.known_free)

    @property
    def coverage(self) -> float:
        total = self.size[0] * self.size[1]
        return self.observed_cells / total if total else 0.0

    def observe(self, observation: dict, tick: int) -> list[SemanticPacket]:
        self.size = tuple(observation["size"])
        expired = 0
        if self.stale_after is not None:
            adaptive_horizon = max(2, round(self.stale_after * (1.0 - 0.7 * self.volatility)))
            for cell, seen_tick in tuple(self.last_seen.items()):
                if tick - seen_tick <= adaptive_horizon:
                    continue
                self.known_obstacles.discard(cell)
                self.known_free.discard(cell)
                del self.last_seen[cell]
                expired += 1
        before = self.observed_cells
        changes = 0
        for x, y, occupied in observation["visible"]:
            cell = (int(x), int(y))
            if cell in self.last_value and self.last_value[cell] != bool(occupied):
                changes += 1
            self.last_value[cell] = bool(occupied)
            self.last_seen[cell] = tick
            if occupied:
                self.known_obstacles.add(cell)
                self.known_free.discard(cell)
            else:
                self.known_free.add(cell)
                self.known_obstacles.discard(cell)
        new_cells = self.observed_cells - before
        change_rate = changes / max(1, len(observation["visible"]))
        self.volatility = 0.82 * self.volatility + 0.18 * change_rate
        self.expired_cells += expired
        uncertainty = min(1.0, 1.0 - self.coverage + self.volatility)
        confidence = max(0.0, self.coverage * (1.0 - self.volatility))
        self.energy += 0.018 + 0.002 * len(observation["visible"])
        learned = self.adapter.encode_semantic(
            self.unit,
            ConceptCode.WORLD_MODEL_UPDATE,
            (self.coverage, uncertainty, new_cells / 9.0, self.volatility, expired / 9.0),
        )
        return [SemanticPacket(
            source=self.unit,
            target=UnitCode.WORKSPACE,
            concept=ConceptCode.WORLD_MODEL_UPDATE,
            tick=tick,
            confidence=confidence,
            uncertainty=uncertainty,
            salience=min(0.8, 0.2 + new_cells / 9.0),
            ttl_ms=300,
            state_delta=(
                float(new_cells), float(self.observed_cells), float(len(self.known_obstacles)),
                float(expired), float(self.volatility),
            ),
            latent=learned.quantized_latent,
            semantic_code=learned.semantic_code,
            quantization_error=learned.quantization_error,
            flags=4 if self.volatility >= 0.015 else 0,
        )]

    def belief_state(self, agent: tuple[int, int], goal: tuple[int, int]) -> dict:
        total = self.size[0] * self.size[1]
        return {
            "size": list(self.size),
            "agent": list(agent),
            "goal": list(goal),
            "obstacles": [list(cell) for cell in sorted(self.known_obstacles)],
            "observed_cells": self.observed_cells,
            "unknown_cells": total - self.observed_cells,
            "model_uncertainty": 1.0 - self.coverage,
            "model_volatility": self.volatility,
        }
