from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from afl_prediction_agent.contracts import FinalDecisionResponse, ModelSettings


def test_final_decision_contract_rejects_invalid_probability_sum() -> None:
    with pytest.raises(ValidationError):
        FinalDecisionResponse(
            predicted_winner_team_id=uuid.uuid4(),
            home_win_probability=0.7,
            away_win_probability=0.4,
            predicted_margin=10,
            confidence_score=70,
            top_drivers=[
                {
                    "label": "recent form",
                    "leans_to": "home",
                    "strength": 0.7,
                    "evidence": "home side stronger",
                    "source_component": "form_analyst_v1",
                }
            ],
            uncertainty_note="none",
            rationale_summary="invalid probabilities",
        )


def test_model_settings_reject_temperature_for_gpt54_reasoning_effort() -> None:
    with pytest.raises(ValidationError):
        ModelSettings(
            provider="codex_app_server",
            model="gpt-5.4",
            temperature=0.2,
            reasoning_effort="xhigh",
        )
