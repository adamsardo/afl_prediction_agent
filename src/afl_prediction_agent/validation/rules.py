from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from afl_prediction_agent.contracts import FinalDecisionResponse, MatchDossierContract


@dataclass(slots=True)
class ValidationOutcome:
    status: str
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


def validate_final_response(
    dossier: MatchDossierContract,
    response: FinalDecisionResponse,
) -> ValidationOutcome:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if response.predicted_winner_team_id not in {
        dossier.match.home_team.team_id,
        dossier.match.away_team.team_id,
    }:
        errors.append({"message": "predicted_winner_team_id must match one of the two teams"})

    market_home = dossier.market.home_implied_probability
    if market_home is not None:
        if abs(response.home_win_probability - market_home) > 0.18:
            warnings.append(
                {
                    "message": "final probability strongly disagrees with market",
                    "market_home_probability": market_home,
                    "agent_home_probability": response.home_win_probability,
                }
            )

    if abs(response.home_win_probability - dossier.baseline.home_win_probability) > 0.2:
        warnings.append(
            {
                "message": "final probability strongly disagrees with baseline",
                "baseline_home_probability": dossier.baseline.home_win_probability,
                "agent_home_probability": response.home_win_probability,
            }
        )

    if response.confidence_score > 85:
        warnings.append({"message": "confidence above 85 should be manually reviewable"})

    if abs(response.predicted_margin) > 60:
        warnings.append({"message": "predicted margin is very large for v1"})

    status = "passed" if not errors else "failed"
    return ValidationOutcome(status=status, errors=errors, warnings=warnings)
