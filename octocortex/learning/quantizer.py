from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class QuantizedResult:
    code: int
    vector: tuple[float, ...]
    error: float


class OnlineVectorQuantizer:
    """Small online codebook that turns a latent vector into a stable code."""

    def __init__(self, dimension: int = 4, codebook_size: int = 8, learning_rate: float = 0.08, seed: int = 29) -> None:
        if dimension < 1 or codebook_size < 1:
            raise ValueError("dimension and codebook_size must be positive")
        self.dimension = dimension
        self.codebook_size = codebook_size
        self.learning_rate = learning_rate
        rng = random.Random(seed)
        self.codebook = [
            [rng.uniform(-0.5, 0.5) for _ in range(dimension)]
            for _ in range(codebook_size)
        ]
        self.usage = [0] * codebook_size
        self.steps = 0
        self.last_error = 0.0

    def quantize(self, values: Iterable[float], learn: bool = True) -> QuantizedResult:
        vector = tuple(float(value) for value in values)
        if len(vector) != self.dimension:
            raise ValueError(f"expected {self.dimension} latent values, got {len(vector)}")
        distances = [
            sum((value - centroid[index]) ** 2 for index, value in enumerate(vector))
            for centroid in self.codebook
        ]
        code = min(range(self.codebook_size), key=lambda index: (distances[index], index))
        error = math.sqrt(distances[code] / self.dimension)
        if learn:
            centroid = self.codebook[code]
            for index, value in enumerate(vector):
                centroid[index] += self.learning_rate * (value - centroid[index])
            self.usage[code] += 1
            self.steps += 1
            self.last_error = error
        # Zero is reserved for legacy/unquantized packets on the wire.
        return QuantizedResult(code + 1, tuple(self.codebook[code]), error)

    @property
    def codes_used(self) -> int:
        return sum(count > 0 for count in self.usage)

    def checkpoint(self) -> dict:
        return {
            "dimension": self.dimension,
            "codebook_size": self.codebook_size,
            "learning_rate": self.learning_rate,
            "codebook": self.codebook,
            "usage": self.usage,
            "steps": self.steps,
        }
