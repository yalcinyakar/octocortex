from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Event:
    source: str
    kind: str
    salience: float
    payload: dict[str, Any] = field(default_factory=dict)
    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Proposal:
    arm: str
    action: str
    confidence: float
    expected_reward: float = 0.0
    risk: float = 0.0
    urgency: float = 0.0
    energy_cost: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Decision:
    action: str
    winner: str
    score: float
    reason: str
    proposals: list[Proposal]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proposals"] = [proposal.to_dict() for proposal in self.proposals]
        return data

