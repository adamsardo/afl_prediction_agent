from __future__ import annotations

import hashlib
import json
from typing import Any

from afl_prediction_agent.contracts import (
    BaselineSummary,
    BenchmarksSection,
    MatchDossierContract,
    MatchSummary,
    MarketSection,
    SelectionSection,
    SourceRefs,
    TeamSummary,
    VenueSummary,
    VenueWeatherSection,
    FormSection,
)
from afl_prediction_agent.features.builder import FeatureBuildResult
from afl_prediction_agent.models.baseline import BaselinePredictionResult
from afl_prediction_agent.storage.context import LoadedMatchContext


class DossierBuilder:
    dossier_version = "dossier_v1"

    def build(
        self,
        *,
        context: LoadedMatchContext,
        feature_result: FeatureBuildResult,
        baseline_result: BaselinePredictionResult,
        feature_set_id,
        baseline_prediction_id,
        benchmark_prediction_id=None,
    ) -> tuple[MatchDossierContract, str]:
        dossier = MatchDossierContract(
            match=MatchSummary(
                match_id=context.match.id,
                season_year=context.season.season_year,
                round_number=context.round.round_number,
                scheduled_at=context.match.scheduled_at,
                venue=VenueSummary(name=context.venue.name, city=context.venue.city),
                home_team=TeamSummary(team_id=context.home_team.id, name=context.home_team.name),
                away_team=TeamSummary(team_id=context.away_team.id, name=context.away_team.name),
            ),
            baseline=BaselineSummary(
                home_win_probability=baseline_result.home_win_probability,
                away_win_probability=baseline_result.away_win_probability,
                predicted_margin=baseline_result.predicted_margin,
                top_drivers=baseline_result.top_drivers,
            ),
            form=FormSection.model_validate(feature_result.form_section),
            selection=SelectionSection.model_validate(feature_result.selection_section),
            venue_weather=VenueWeatherSection.model_validate(feature_result.venue_weather_section),
            market=MarketSection.model_validate(feature_result.market_section),
            benchmarks=BenchmarksSection.model_validate(feature_result.benchmarks_section),
            uncertainties=feature_result.uncertainties,
            source_refs=SourceRefs(
                lineup_snapshot_ids=[
                    context.home_lineup.id if context.home_lineup else None,
                    context.away_lineup.id if context.away_lineup else None,
                ],
                injury_snapshot_id=context.injury_snapshot.id if context.injury_snapshot else None,
                weather_snapshot_id=context.weather_snapshot.id if context.weather_snapshot else None,
                odds_snapshot_id=context.odds_snapshot.id if context.odds_snapshot else None,
                benchmark_prediction_id=benchmark_prediction_id,
                feature_set_id=feature_set_id,
                baseline_prediction_id=baseline_prediction_id,
            ),
        )
        input_hash = hashlib.sha256(
            json.dumps(dossier.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return dossier, input_hash
