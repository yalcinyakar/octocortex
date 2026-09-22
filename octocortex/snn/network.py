from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LIFNeuron:
    threshold: float = 1.0
    leak: float = 0.72
    potential: float = 0.0
    spikes: int = 0

    def step(self, current: float) -> bool:
        self.potential = self.potential * self.leak + current
        if self.potential < self.threshold:
            return False
        self.potential = 0.0
        self.spikes += 1
        return True


class DangerNetwork:
    """A minimal local SNN that spikes when nearby obstacle pressure accumulates."""

    def __init__(self) -> None:
        self.neuron = LIFNeuron(threshold=1.0, leak=0.6)
        self.energy = 0.0

    def process(self, danger_signal: float) -> tuple[bool, float]:
        self.energy += 0.015 + (0.035 * danger_signal)
        spike = self.neuron.step(danger_signal)
        return spike, self.neuron.potential

