from __future__ import annotations

from .models import ActionCode, Decision, Proposal, ReasonCode
from .octoir import UnitCode


class Arbitrator:
    """Scores local proposals while giving safety vetoes explicit precedence."""

    def score(self, proposal: Proposal) -> float:
        return (
            0.34 * proposal.confidence
            + 0.24 * proposal.expected_reward
            + 0.28 * proposal.risk
            + 0.22 * proposal.urgency
            - 0.08 * proposal.energy_cost
        )

    def choose(self, proposals: list[Proposal]) -> Decision:
        if not proposals:
            return Decision(
                ActionCode.WAIT, UnitCode.WORKSPACE, 0.0,
                ReasonCode.NONE, 0.0, False, [],
            )

        vetoes = [p for p in proposals if p.risk >= 0.9 and p.urgency >= 0.75]
        candidates = vetoes or proposals
        winner = max(candidates, key=self.score)
        return Decision(
            action=winner.action,
            winner=winner.arm,
            score=round(self.score(winner), 4),
            reason_code=winner.reason_code,
            reason_value=winner.reason_value,
            safety_veto=bool(vetoes),
            proposals=proposals,
        )
