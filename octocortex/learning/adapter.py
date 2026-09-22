from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class AdapterResult:
    latent: tuple[float, ...]
    reconstruction: tuple[float, ...]
    loss: float


class SparseSemanticAdapter:
    """Tiny online autoencoder that learns the shared OctoIR latent space.

    The adapter intentionally has no external ML dependency. It is a reference
    implementation of the learning contract that can later be replaced by a
    PyTorch/JAX module without changing OctoIR packets.
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 4,
        active_latents: int = 2,
        learning_rate: float = 0.04,
        seed: int = 17,
    ) -> None:
        if not 0 < active_latents <= latent_dim:
            raise ValueError("active_latents must be between 1 and latent_dim")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.active_latents = active_latents
        self.learning_rate = learning_rate
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(input_dim)
        self.encoder = [
            [rng.uniform(-scale, scale) for _ in range(input_dim)]
            for _ in range(latent_dim)
        ]
        self.decoder = [
            [rng.uniform(-scale, scale) for _ in range(latent_dim)]
            for _ in range(input_dim)
        ]
        self.encoder_bias = [0.0] * latent_dim
        self.decoder_bias = [0.0] * input_dim
        self.steps = 0
        self.total_loss = 0.0
        self.last_loss = 0.0

    def semantic_vector(self, source: int, concept: int, features: Iterable[float] = ()) -> tuple[float, ...]:
        feature_values = [max(-1.0, min(1.0, float(value))) for value in features]
        values = [concept / 8.0, source / 7.0, *feature_values[: self.input_dim - 2]]
        values.extend([0.0] * (self.input_dim - len(values)))
        return tuple(values)

    def _forward(self, values: tuple[float, ...]) -> tuple[list[float], list[float], list[float]]:
        hidden = [
            math.tanh(sum(weight * value for weight, value in zip(row, values)) + bias)
            for row, bias in zip(self.encoder, self.encoder_bias)
        ]
        active = sorted(range(self.latent_dim), key=lambda index: abs(hidden[index]), reverse=True)[: self.active_latents]
        latent = [hidden[index] if index in active else 0.0 for index in range(self.latent_dim)]
        reconstruction = [
            sum(weight * value for weight, value in zip(row, latent)) + bias
            for row, bias in zip(self.decoder, self.decoder_bias)
        ]
        return hidden, latent, reconstruction

    def encode_and_learn(self, values: Iterable[float]) -> AdapterResult:
        vector = tuple(max(-1.0, min(1.0, float(value))) for value in values)
        if len(vector) != self.input_dim:
            raise ValueError(f"expected {self.input_dim} adapter inputs, got {len(vector)}")
        hidden, latent, reconstruction = self._forward(vector)
        errors = [reconstructed - expected for reconstructed, expected in zip(reconstruction, vector)]
        loss = sum(error * error for error in errors) / self.input_dim

        # Calculate the encoder signal before updating decoder weights.
        hidden_gradient = [
            sum(self.decoder[out_index][latent_index] * errors[out_index] for out_index in range(self.input_dim))
            * (1.0 - hidden[latent_index] ** 2)
            if latent[latent_index] else 0.0
            for latent_index in range(self.latent_dim)
        ]
        rate = self.learning_rate
        for out_index in range(self.input_dim):
            for latent_index in range(self.latent_dim):
                self.decoder[out_index][latent_index] -= rate * errors[out_index] * latent[latent_index]
            self.decoder_bias[out_index] -= rate * errors[out_index]
        for latent_index in range(self.latent_dim):
            for input_index in range(self.input_dim):
                self.encoder[latent_index][input_index] -= rate * hidden_gradient[latent_index] * vector[input_index]
            self.encoder_bias[latent_index] -= rate * hidden_gradient[latent_index]

        self.steps += 1
        self.last_loss = loss
        self.total_loss += loss
        return AdapterResult(tuple(latent), tuple(reconstruction), loss)

    def encode_semantic(self, source: int, concept: int, features: Iterable[float] = ()) -> AdapterResult:
        return self.encode_and_learn(self.semantic_vector(source, concept, features))

    @property
    def mean_loss(self) -> float:
        return self.total_loss / self.steps if self.steps else 0.0

    def checkpoint(self) -> dict:
        return {
            "version": 1,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "active_latents": self.active_latents,
            "learning_rate": self.learning_rate,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "encoder_bias": self.encoder_bias,
            "decoder_bias": self.decoder_bias,
            "steps": self.steps,
            "total_loss": self.total_loss,
        }
