from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VenueSummary(StrictModel):
    name: str
    city: str | None = None


class TeamSummary(StrictModel):
    team_id: uuid.UUID
    name: str


class MatchSummary(StrictModel):
    match_id: uuid.UUID
    season_year: int
    round_number: int
    scheduled_at: datetime
    venue: VenueSummary
    home_team: TeamSummary
    away_team: TeamSummary


class BaselineDriver(StrictModel):
    label: str
    leans_to: Literal["home", "away", "neutral"]
    strength: float = Field(ge=0.0, le=1.0)
    evidence: str
    source_component: str


class BaselineSummary(StrictModel):
    home_win_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_margin: float
    top_drivers: list[BaselineDriver] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_probability_pair(self) -> "BaselineSummary":
        total = self.home_win_probability + self.away_win_probability
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("baseline probabilities must sum to 1")
        return self


class FormSection(StrictModel):
    home_recent_form: dict[str, Any]
    away_recent_form: dict[str, Any]
    team_stat_edges: list[dict[str, Any]]


class SelectionSection(StrictModel):
    home_named_changes: int = Field(ge=0)
    away_named_changes: int = Field(ge=0)
    home_continuity_score: float | None = None
    away_continuity_score: float | None = None
    home_lineup_strength: float
    away_lineup_strength: float
    home_selected_experience: float | None = None
    away_selected_experience: float | None = None
    home_missing_player_penalty: float | None = None
    away_missing_player_penalty: float | None = None
    key_absences: list[dict[str, Any]]


class VenueWeatherSection(StrictModel):
    home_ground_edge: bool
    travel_context: dict[str, Any]
    forecast: dict[str, Any]


class MarketSection(StrictModel):
    home_implied_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    away_implied_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    bookmaker_count: int = Field(ge=0)
    market_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_disagreement: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probability_pair(self) -> "MarketSection":
        if self.home_implied_probability is None and self.away_implied_probability is None:
            return self
        if self.home_implied_probability is None or self.away_implied_probability is None:
            raise ValueError("market probabilities must be provided together")
        total = self.home_implied_probability + self.away_implied_probability
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("market probabilities must sum to 1")
        return self


class BenchmarksSection(StrictModel):
    squiggle: dict[str, Any] = Field(default_factory=dict)


class SourceRefs(StrictModel):
    lineup_snapshot_ids: list[uuid.UUID] = Field(default_factory=list, min_length=2)
    injury_snapshot_id: uuid.UUID | None = None
    weather_snapshot_id: uuid.UUID | None = None
    odds_snapshot_id: uuid.UUID | None = None
    benchmark_prediction_id: uuid.UUID | None = None
    feature_set_id: uuid.UUID | None = None
    baseline_prediction_id: uuid.UUID | None = None


class MatchDossierContract(StrictModel):
    match: MatchSummary
    baseline: BaselineSummary
    form: FormSection
    selection: SelectionSection
    venue_weather: VenueWeatherSection
    market: MarketSection
    benchmarks: BenchmarksSection
    uncertainties: list[str] = Field(default_factory=list)
    source_refs: SourceRefs


class AnalystSignal(StrictModel):
    label: str
    leans_to: Literal["home", "away", "neutral"]
    strength: float = Field(ge=0.0, le=1.0)
    evidence: str


class AnalystResponse(StrictModel):
    summary: str
    signals: list[AnalystSignal] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class CasePoint(StrictModel):
    label: str
    strength: float = Field(ge=0.0, le=1.0)
    evidence: str


class CaseAgentResponse(StrictModel):
    side: Literal["home", "away"]
    case_summary: str
    strongest_points: list[CasePoint] = Field(min_length=1)
    weak_points: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)


class FinalDecisionResponse(StrictModel):
    predicted_winner_team_id: uuid.UUID
    home_win_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_margin: float
    confidence_score: float = Field(ge=0.0, le=100.0)
    top_drivers: list[BaselineDriver] = Field(min_length=1)
    uncertainty_note: str
    rationale_summary: str

    @model_validator(mode="after")
    def validate_consistency(self) -> "FinalDecisionResponse":
        total = self.home_win_probability + self.away_win_probability
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1")
        if self.predicted_margin == 0:
            raise ValueError("predicted_margin cannot be zero")
        home_is_winner = self.home_win_probability > self.away_win_probability
        margin_is_home = self.predicted_margin > 0
        if home_is_winner != margin_is_home:
            raise ValueError("predicted_margin sign must align with winner")
        return self


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class ModelSettings(StrictModel):
    provider: str
    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def validate_provider_constraints(self) -> "ModelSettings":
        if (
            self.provider == "codex_app_server"
            and self.model == "gpt-5.4"
            and self.reasoning_effort not in {None, "none"}
            and self.temperature is not None
        ):
            raise ValueError(
                "temperature must be null for codex_app_server + gpt-5.4 when reasoning_effort is set"
            )
        return self


class RunConfigFile(StrictModel):
    config_name: str
    feature_version: str
    winner_model_version: str
    margin_model_version: str
    prompt_set_version: str
    analyst_model: ModelSettings
    case_model: ModelSettings
    final_model: ModelSettings


class RunRoundRequest(StrictModel):
    config_name: str
    lock_timestamp: datetime | None = None
    notes: str | None = None
    fetch_sources: bool = False


class RoundRunSummaryResponse(StrictModel):
    run_id: uuid.UUID
    round_id: uuid.UUID
    status: str
    created_at: datetime
    completed_at: datetime | None = None


class RunDetailResponse(StrictModel):
    run_id: uuid.UUID
    round_id: uuid.UUID
    season_id: uuid.UUID
    status: str
    lock_timestamp: datetime
    match_count: int
    eligible_match_count: int = 0
    skipped_match_count: int = 0
    processed_match_count: int = 0
    baseline_only_match_count: int = 0
    verdict_count: int
    created_at: datetime
    completed_at: datetime | None = None


class MatchRunDetailResponse(StrictModel):
    run_id: uuid.UUID
    match_id: uuid.UUID
    season_year: int | None = None
    round_number: int | None = None
    home_team_name: str | None = None
    away_team_name: str | None = None
    venue_name: str | None = None
    prediction_lock_timestamp: datetime | None = None
    match_status: Literal["completed", "baseline_only", "skipped", "failed", "pending"]
    skip_reason: str | None = None
    analyst_summaries: dict[str, str] = Field(default_factory=dict)
    case_summaries: dict[str, str] = Field(default_factory=dict)
    bookmaker_snapshot: dict[str, Any] | None = None
    squiggle_snapshot: dict[str, Any] | None = None
    final_decision_model_version: str | None = None
    prompt_version_set: str | None = None
    feature_version: str | None = None
    dossier: dict[str, Any] | None = None
    baseline_prediction: dict[str, Any] | None = None
    final_verdict: dict[str, Any] | None = None
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationSummaryResponse(StrictModel):
    season_id: uuid.UUID
    run_config_id: uuid.UUID
    summary_type: str
    summary: dict[str, Any]
