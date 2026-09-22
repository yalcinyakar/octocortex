from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from octocortex.core.octoir import UnitCode


class CapabilityCode(IntEnum):
    PLAN_ROUTE = 1


@dataclass(frozen=True, slots=True)
class CapabilityLease:
    capability: CapabilityCode
    owner: UnitCode
    primary: UnitCode

    @property
    def delegated(self) -> bool:
        return self.owner != self.primary


class CapabilityRouter:
    """Assigns a capability to the first healthy unit in its fallback chain."""

    def __init__(self) -> None:
        self.routes = {
            CapabilityCode.PLAN_ROUTE: (
                UnitCode.PLANNING,
                UnitCode.MEMORY,
                UnitCode.PERCEPTION,
            ),
        }
        self.assignments: dict[CapabilityCode, UnitCode] = {}
        self.handoffs = 0

    def resolve(self, capability: CapabilityCode, available: set[UnitCode]) -> CapabilityLease | None:
        chain = self.routes[capability]
        owner = next((unit for unit in chain if unit in available), None)
        if owner is None:
            return None
        previous = self.assignments.get(capability, chain[0])
        if owner != previous:
            self.handoffs += 1
        self.assignments[capability] = owner
        return CapabilityLease(capability, owner, chain[0])
