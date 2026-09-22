from __future__ import annotations

from .models import Decision, Proposal


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
            return Decision("WAIT", "workspace", 0.0, "No arm proposed an action", [])

        vetoes = [p for p in proposals if p.risk >= 0.9 and p.urgency >= 0.75]
        candidates = vetoes or proposals
        winner = max(candidates, key=self.score)
        prefix = "Safety veto" if vetoes else "Highest arbitration score"
        return Decision(
            action=winner.action,
            winner=winner.arm,
            score=round(self.score(winner), 4),
            reason=f"{prefix}: {winner.reason}",
            proposals=proposals,
        )

