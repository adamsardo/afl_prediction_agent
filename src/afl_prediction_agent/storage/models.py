from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from afl_prediction_agent.core.db.base import (
    Base,
    CreatedAtMixin,
    UUIDPrimaryKeyMixin,
    UpdatedAtMixin,
)
from afl_prediction_agent.core.db.types import JSON_VARIANT


JSONField = JSON_VARIANT


class Competition(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "competitions"

    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)


class Season(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "season_year"),)

    competition_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("competitions.id"), nullable=False
    )
    season_year: Mapped[int] = mapped_column(Integer, nullable=False)


class Round(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "rounds"
    __table_args__ = (UniqueConstraint("season_id", "round_number", "is_finals"),)

    season_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("seasons.id"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    round_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_finals: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Team(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "teams"

    team_code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    short_name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    state_code: Mapped[str | None] = mapped_column(String(10))


class Venue(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "venues"

    venue_code: Mapped[str | None] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str | None] = mapped_column(String(255))
    state_code: Mapped[str | None] = mapped_column(String(10))
    timezone: Mapped[str] = mapped_column(
        String(100), nullable=False, default="Australia/Melbourne"
    )
    bom_location_code: Mapped[str | None] = mapped_column(String(100))
    bom_station_id: Mapped[str | None] = mapped_column(String(100))
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))


class Player(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "players"

    player_code: Mapped[str | None] = mapped_column(String(50), unique=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))
    current_team_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("teams.id")
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Match(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "matches"
    __table_args__ = (
        Index("ix_matches_round_id", "round_id"),
        Index("ix_matches_scheduled_at", "scheduled_at"),
        Index(
            "ix_matches_home_away_scheduled",
            "home_team_id",
            "away_team_id",
            "scheduled_at",
        ),
    )

    season_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("seasons.id"), nullable=False)
    round_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("rounds.id"), nullable=False)
    match_code: Mapped[str | None] = mapped_column(String(100), unique=True)
    home_team_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("teams.id"), nullable=False)
    venue_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("venues.id"), nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    winning_team_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("teams.id"))
    actual_margin: Mapped[int | None] = mapped_column(Integer)


class SourceFetchLog(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "source_fetch_logs"
    __table_args__ = (
        Index("ix_source_fetch_logs_source_name_requested_at", "source_name", "requested_at"),
        Index("ix_source_fetch_logs_entity_type_entity_key", "entity_type", "entity_key"),
    )

    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_key: Mapped[str | None] = mapped_column(String(255))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    request_meta: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)
    response_meta: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)
    raw_payload: Mapped[dict | None] = mapped_column(JSONField)
    error_message: Mapped[str | None] = mapped_column(Text)


class ExternalIdMapping(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "external_id_mappings"
    __table_args__ = (
        UniqueConstraint("source_name", "entity_type", "external_id"),
    )

    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    mapping_status: Mapped[str] = mapped_column(String(50), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


class RunConfig(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "run_configs"

    config_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    winner_model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    margin_model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_set_version: Mapped[str] = mapped_column(String(100), nullable=False)
    final_model_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    final_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    default_temperature: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    default_reasoning_effort: Mapped[str | None] = mapped_column(String(20))
    config: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class PromptTemplate(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint("prompt_set_version", "step_name"),
    )

    prompt_set_version: Mapped[str] = mapped_column(String(100), nullable=False)
    step_name: Mapped[str] = mapped_column(String(100), nullable=False)
    template_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class RoundRun(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "round_runs"
    __table_args__ = (
        Index("ix_round_runs_round_id_created_at", "round_id", "created_at"),
        Index("ix_round_runs_run_config_id", "run_config_id"),
    )

    season_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("seasons.id"), nullable=False)
    round_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("rounds.id"), nullable=False)
    run_config_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("run_configs.id"), nullable=False
    )
    lock_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LineupSnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "lineup_snapshots"
    __table_args__ = (
        Index(
            "ix_lineup_snapshots_match_team_fetched_at",
            "match_id",
            "team_id",
            "fetched_at",
        ),
    )

    round_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("round_runs.id"))
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("teams.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class LineupSnapshotPlayer(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "lineup_snapshot_players"
    __table_args__ = (
        Index("ix_lineup_snapshot_players_lineup_snapshot_id", "lineup_snapshot_id"),
        Index("ix_lineup_snapshot_players_player_id", "player_id"),
    )

    lineup_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("lineup_snapshots.id"), nullable=False
    )
    player_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("players.id"))
    source_player_name: Mapped[str] = mapped_column(String(255), nullable=False)
    slot_label: Mapped[str | None] = mapped_column(String(100))
    named_role: Mapped[str | None] = mapped_column(String(100))
    is_selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_interchange: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_emergency: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_sub: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mapping_status: Mapped[str] = mapped_column(String(50), nullable=False)


class InjurySnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "injury_snapshots"

    round_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("round_runs.id"))
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class InjurySnapshotEntry(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "injury_snapshot_entries"
    __table_args__ = (
        Index("ix_injury_snapshot_entries_snapshot_id", "injury_snapshot_id"),
        Index("ix_injury_snapshot_entries_team_id", "team_id"),
        Index("ix_injury_snapshot_entries_player_id", "player_id"),
    )

    injury_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("injury_snapshots.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("teams.id"))
    player_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("players.id"))
    source_player_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status_label: Mapped[str] = mapped_column(String(100), nullable=False)
    injury_note: Mapped[str | None] = mapped_column(Text)
    estimated_return_text: Mapped[str | None] = mapped_column(String(255))
    uncertainty_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mapping_status: Mapped[str] = mapped_column(String(50), nullable=False)


class WeatherSnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "weather_snapshots"
    __table_args__ = (
        Index("ix_weather_snapshots_match_fetched_at", "match_id", "fetched_at"),
    )

    round_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("round_runs.id"))
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    venue_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("venues.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    temperature_c: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    rain_probability_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    rainfall_mm: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    wind_kmh: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    weather_text: Mapped[str | None] = mapped_column(String(255))
    severe_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    payload: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class OddsSnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "odds_snapshots"
    __table_args__ = (
        Index("ix_odds_snapshots_match_fetched_at", "match_id", "fetched_at"),
    )

    round_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("round_runs.id"))
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_median_price: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    away_median_price: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    home_implied_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    away_implied_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    bookmaker_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class OddsSnapshotBook(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "odds_snapshot_books"

    odds_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("odds_snapshots.id"), nullable=False
    )
    bookmaker_key: Mapped[str] = mapped_column(String(100), nullable=False)
    market_key: Mapped[str] = mapped_column(String(100), nullable=False)
    home_price: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    away_price: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    overround_pct: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))


class BenchmarkPrediction(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "benchmark_predictions"

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_winner_team_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("teams.id")
    )
    home_win_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    away_win_probability: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    predicted_margin: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    payload: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class TeamMatchStat(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "team_match_stats"
    __table_args__ = (
        UniqueConstraint("match_id", "team_id", "source_name"),
    )

    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("teams.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    stats: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class PlayerMatchStat(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "player_match_stats"
    __table_args__ = (
        UniqueConstraint("match_id", "player_id", "source_name"),
    )

    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("teams.id"), nullable=False)
    player_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("players.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    stats: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class FeatureSet(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "feature_sets"
    __table_args__ = (
        Index("ix_feature_sets_match_feature_version", "match_id", "feature_version"),
        Index("ix_feature_sets_round_run_id", "round_run_id"),
    )

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    features: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class BaselineModelRun(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "baseline_model_runs"

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    training_window: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class BaselinePrediction(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "baseline_predictions"
    __table_args__ = (
        Index("ix_baseline_predictions_round_match", "round_run_id", "match_id"),
    )

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    winner_model_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("baseline_model_runs.id")
    )
    margin_model_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("baseline_model_runs.id")
    )
    predicted_winner_team_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("teams.id")
    )
    home_win_probability: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    away_win_probability: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    predicted_margin: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    confidence_reference: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    top_drivers: Mapped[list[dict]] = mapped_column(JSONField, nullable=False, default=list)


class MatchDossier(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "match_dossiers"
    __table_args__ = (
        UniqueConstraint("round_run_id", "match_id"),
    )

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    dossier_version: Mapped[str] = mapped_column(String(100), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dossier: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class AgentStep(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "agent_steps"
    __table_args__ = (
        Index("ix_agent_steps_round_run_match", "round_run_id", "match_id"),
        Index("ix_agent_steps_step_name", "step_name"),
        Index("ix_agent_steps_status", "status"),
    )

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    step_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("prompt_templates.id")
    )
    rendered_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    temperature: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    reasoning_effort: Mapped[str | None] = mapped_column(String(20))
    input_json: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)
    output_json: Mapped[dict | None] = mapped_column(JSONField)
    provider_meta: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    tokens_input: Mapped[int | None] = mapped_column(Integer)
    tokens_output: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)


class FinalAgentVerdict(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "final_agent_verdicts"
    __table_args__ = (
        Index("ix_final_agent_verdicts_round_match", "round_run_id", "match_id"),
    )

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    final_agent_step_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("agent_steps.id"), nullable=False
    )
    predicted_winner_team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("teams.id"), nullable=False
    )
    home_win_probability: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    away_win_probability: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    predicted_margin: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    top_drivers: Mapped[list[dict]] = mapped_column(JSONField, nullable=False, default=list)
    uncertainty_note: Mapped[str] = mapped_column(Text, nullable=False)
    rationale_summary: Mapped[str] = mapped_column(Text, nullable=False)
    validation_status: Mapped[str] = mapped_column(String(50), nullable=False)
    correction_pass_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ValidationLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "validation_logs"

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    component_name: Mapped[str] = mapped_column(String(100), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(50), nullable=False)
    errors: Mapped[list[dict] | None] = mapped_column(JSONField)


class AuditEvent(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "audit_events"

    round_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("round_runs.id"))
    match_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("matches.id"))
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)


class MatchEvaluation(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "match_evaluations"

    round_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("round_runs.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("matches.id"), nullable=False)
    actual_winner_team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("teams.id"), nullable=False
    )
    actual_margin: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    agent_winner_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    baseline_winner_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    agent_margin_error: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    baseline_margin_error: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    agent_brier: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    baseline_brier: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    agent_log_loss: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    baseline_log_loss: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))


class SeasonEvaluationSummary(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "season_evaluation_summaries"

    season_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("seasons.id"), nullable=False)
    run_config_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("run_configs.id"), nullable=False
    )
    summary_type: Mapped[str] = mapped_column(String(100), nullable=False)
    summary: Mapped[dict] = mapped_column(JSONField, nullable=False, default=dict)
