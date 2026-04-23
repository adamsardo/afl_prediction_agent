from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge

from afl_prediction_agent.core.settings import get_settings


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clip_probability(value: float) -> float:
    return max(0.01, min(0.99, value))


@dataclass(slots=True)
class BaselinePredictionResult:
    home_win_probability: float
    away_win_probability: float
    predicted_margin: float
    confidence_reference: float
    top_drivers: list[dict[str, Any]]


class BaselineModelRegistry:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.model_artifact_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, model_type: str, model_version: str) -> Path:
        return self.settings.model_artifact_dir / f"{model_type}_{model_version}.joblib"

    def load(self, model_type: str, model_version: str) -> dict[str, Any] | None:
        path = self.artifact_path(model_type, model_version)
        if not path.exists():
            return None
        return joblib.load(path)

    def save(self, model_type: str, model_version: str, payload: dict[str, Any]) -> Path:
        path = self.artifact_path(model_type, model_version)
        joblib.dump(payload, path)
        return path

    def train_winner_model(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        labels: list[int],
        model_version: str,
    ) -> Path:
        frame = self._numeric_frame(feature_rows)
        model = LogisticRegression(max_iter=500, random_state=7)
        model.fit(frame.to_numpy(), labels)
        return self.save(
            "winner",
            model_version,
            {"model": model, "feature_names": frame.columns},
        )

    def train_margin_model(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        labels: list[float],
        model_version: str,
    ) -> Path:
        frame = self._numeric_frame(feature_rows)
        model = Ridge(alpha=1.0, random_state=7)
        model.fit(frame.to_numpy(), labels)
        return self.save(
            "margin",
            model_version,
            {"model": model, "feature_names": frame.columns},
        )

    def _numeric_frame(self, feature_rows: list[dict[str, Any]]) -> pl.DataFrame:
        frame = pl.DataFrame(feature_rows)
        numeric_columns = [
            name
            for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            if dtype.is_numeric()
        ]
        return frame.select(numeric_columns).fill_null(0.0)


class DeterministicBaselineService:
    def __init__(
        self,
        *,
        winner_model_version: str,
        margin_model_version: str,
    ) -> None:
        self.registry = BaselineModelRegistry()
        self.winner_model_version = winner_model_version
        self.margin_model_version = margin_model_version

    def predict(self, features: dict[str, Any]) -> BaselinePredictionResult:
        numeric_features = self._numeric_features(features)
        winner_payload = self.registry.load("winner", self.winner_model_version)
        margin_payload = self.registry.load("margin", self.margin_model_version)

        if winner_payload is not None:
            home_probability = self._predict_with_artifact(winner_payload, numeric_features)
        else:
            home_probability = self._heuristic_home_probability(features)

        if margin_payload is not None:
            predicted_margin = self._predict_margin_with_artifact(margin_payload, numeric_features)
        else:
            predicted_margin = self._heuristic_margin(features, home_probability)

        away_probability = round(1.0 - home_probability, 6)
        home_probability = round(home_probability, 6)
        predicted_margin = round(predicted_margin, 2)
        top_drivers = self._top_drivers(features)
        confidence_reference = round(max(home_probability, away_probability), 4)
        return BaselinePredictionResult(
            home_win_probability=home_probability,
            away_win_probability=away_probability,
            predicted_margin=predicted_margin,
            confidence_reference=confidence_reference,
            top_drivers=top_drivers,
        )

    def _predict_with_artifact(self, payload: dict[str, Any], features: dict[str, float]) -> float:
        feature_names: list[str] = payload["feature_names"]
        vector = [[features.get(name, 0.0) for name in feature_names]]
        model: LogisticRegression = payload["model"]
        probability = float(model.predict_proba(vector)[0][1])
        return round(_clip_probability(probability), 6)

    def _predict_margin_with_artifact(self, payload: dict[str, Any], features: dict[str, float]) -> float:
        feature_names: list[str] = payload["feature_names"]
        vector = [[features.get(name, 0.0) for name in feature_names]]
        model: Ridge = payload["model"]
        return float(model.predict(vector)[0])

    def _numeric_features(self, features: dict[str, Any]) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in features.items():
            if isinstance(value, bool):
                output[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                output[key] = float(value)
        return output

    def _heuristic_home_probability(self, features: dict[str, Any]) -> float:
        market_home = features.get("market_home_implied_probability")
        market_anchor = math.log((market_home or 0.55) / (1.0 - (market_home or 0.55)))
        form_edge = (
            (features.get("home_recent_win_rate", 0.5) - features.get("away_recent_win_rate", 0.5)) * 1.4
            + (features.get("home_recent_avg_margin", 0.0) - features.get("away_recent_avg_margin", 0.0)) / 30.0
        )
        lineup_edge = (
            (features.get("home_lineup_strength", 0.0) - features.get("away_lineup_strength", 0.0)) / 45.0
        )
        injury_edge = (
            (features.get("away_injury_count", 0.0) - features.get("home_injury_count", 0.0)) * 0.06
        )
        rest_edge = (
            (features.get("home_rest_days", 7.0) - features.get("away_rest_days", 7.0)) / 14.0
        )
        venue_edge = 0.18 if features.get("home_ground_edge") else 0.0
        squiggle_edge = 0.0
        squiggle_home = features.get("squiggle_home_probability")
        if isinstance(squiggle_home, (int, float)):
            squiggle_edge = (float(squiggle_home) - 0.5) * 0.8
        logit = market_anchor + form_edge + lineup_edge + injury_edge + rest_edge + venue_edge + squiggle_edge
        return round(_clip_probability(_sigmoid(logit)), 6)

    def _heuristic_margin(self, features: dict[str, Any], home_probability: float) -> float:
        probability_margin = (home_probability - 0.5) * 45.0
        recent_margin_edge = (
            (features.get("home_recent_avg_margin", 0.0) - features.get("away_recent_avg_margin", 0.0)) * 0.35
        )
        lineup_edge = (
            (features.get("home_lineup_strength", 0.0) - features.get("away_lineup_strength", 0.0)) * 0.12
        )
        venue_edge = 5.0 if features.get("home_ground_edge") else 0.0
        return probability_margin + recent_margin_edge + lineup_edge + venue_edge

    def _top_drivers(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[tuple[str, float, str, str]] = [
            (
                "recent form",
                abs(features.get("home_recent_win_rate", 0.5) - features.get("away_recent_win_rate", 0.5)),
                "home" if features.get("home_recent_win_rate", 0.5) >= features.get("away_recent_win_rate", 0.5) else "away",
                "rolling recent win rate differential",
            ),
            (
                "lineup strength",
                min(abs(features.get("home_lineup_strength", 0.0) - features.get("away_lineup_strength", 0.0)) / 25.0, 1.0),
                "home" if features.get("home_lineup_strength", 0.0) >= features.get("away_lineup_strength", 0.0) else "away",
                "aggregate player form rating differential",
            ),
            (
                "market consensus",
                abs((features.get("market_home_implied_probability") or 0.5) - 0.5) * 2.0,
                "home"
                if (features.get("market_home_implied_probability") or 0.5)
                >= (features.get("market_away_implied_probability") or 0.5)
                else "away",
                "median bookmaker implied probability at lock time",
            ),
            (
                "venue context",
                0.55 if features.get("home_ground_edge") else 0.2,
                "home" if features.get("home_ground_edge") else "neutral",
                "home ground and travel context",
            ),
        ]
        top = sorted(candidates, key=lambda item: item[1], reverse=True)[:3]
        return [
            {
                "label": label,
                "leans_to": leans_to,
                "strength": round(min(strength, 1.0), 3),
                "evidence": evidence,
                "source_component": "baseline_model",
            }
            for label, strength, leans_to, evidence in top
        ]
