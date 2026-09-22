from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

from .octoir import UnitCode


class ActionCode(IntEnum):
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class ReasonCode(IntEnum):
    NONE = 0
    GOAL_PROGRESS = 1
    DANGER_SPIKE = 2
    COLLISION_MEMORY = 3
    NO_VALID_MOVE = 4


def reason_text(code: ReasonCode, value: float = 0.0) -> str:
    """Human-facing decoder kept outside the internal decision protocol."""
    if code is ReasonCode.GOAL_PROGRESS:
        return f"move reduces goal distance to {int(value)}"
    if code is ReasonCode.DANGER_SPIKE:
        return "local danger neuron spiked"
    if code is ReasonCode.COLLISION_MEMORY:
        return f"recalled {int(value)} collision(s)"
    if code is ReasonCode.NO_VALID_MOVE:
        return "no valid move"
    return "no semantic reason code"


@dataclass(slots=True)
class Proposal:
    arm: UnitCode
    action: ActionCode
    confidence: float
    expected_reward: float = 0.0
    risk: float = 0.0
    urgency: float = 0.0
    energy_cost: float = 0.0
    reason_code: ReasonCode = ReasonCode.NONE
    reason_value: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["arm"] = self.arm.name.lower()
        data["action"] = self.action.name
        data["reason"] = reason_text(self.reason_code, self.reason_value)
        data["reason_code"] = int(self.reason_code)
        return data


@dataclass(slots=True)
class Decision:
    action: ActionCode
    winner: UnitCode
    score: float
    reason_code: ReasonCode
    reason_value: float
    safety_veto: bool
    proposals: list[Proposal]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["action"] = self.action.name
        data["winner"] = self.winner.name.lower()
        prefix = "Safety veto" if self.safety_veto else "Highest arbitration score"
        data["reason"] = f"{prefix}: {reason_text(self.reason_code, self.reason_value)}"
        data["reason_code"] = int(self.reason_code)
        data["proposals"] = [proposal.to_dict() for proposal in self.proposals]
        return data
