from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class FetchEnvelope:
    source_name: str
    fetched_at: datetime
    request_meta: dict[str, Any] = field(default_factory=dict)
    response_meta: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedTeam:
    name: str
    short_name: str | None = None
    team_code: str | None = None
    external_id: str | None = None
    state_code: str | None = None


@dataclass(slots=True)
class NormalizedVenue:
    name: str
    city: str | None = None
    state_code: str | None = None
    timezone_name: str = "Australia/Melbourne"
    external_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass(slots=True)
class NormalizedFixtureMatch:
    season_year: int
    round_number: int
    round_name: str
    scheduled_at: datetime
    home_team: NormalizedTeam
    away_team: NormalizedTeam
    venue: NormalizedVenue
    is_finals: bool = False
    match_code: str | None = None
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None


@dataclass(slots=True)
class NormalizedLineupPlayer:
    source_player_name: str
    source_player_id: str | None = None
    slot_label: str | None = None
    named_role: str | None = None
    is_selected: bool = True
    is_interchange: bool = False
    is_emergency: bool = False
    is_sub: bool = False


@dataclass(slots=True)
class NormalizedLineupSnapshot:
    match_code: str | None
    home_or_away: str
    team: NormalizedTeam
    players: list[NormalizedLineupPlayer]
    published_at: datetime | None = None


@dataclass(slots=True)
class NormalizedInjuryEntry:
    team_name: str
    source_player_name: str
    status_label: str
    injury_note: str | None = None
    estimated_return_text: str | None = None
    source_player_id: str | None = None
    uncertainty_flag: bool = False


@dataclass(slots=True)
class NormalizedInjurySnapshot:
    entries: list[NormalizedInjuryEntry]
    published_at: datetime | None = None


@dataclass(slots=True)
class NormalizedWeatherSnapshot:
    venue_name: str
    forecast_for: datetime
    temperature_c: float | None = None
    rain_probability_pct: float | None = None
    rainfall_mm: float | None = None
    wind_kmh: float | None = None
    weather_text: str | None = None
    severe_flag: bool = False


@dataclass(slots=True)
class NormalizedOddsBook:
    bookmaker_key: str
    market_key: str
    home_price: float
    away_price: float


@dataclass(slots=True)
class NormalizedOddsSnapshot:
    home_team_name: str
    away_team_name: str
    commence_time: datetime | None
    books: list[NormalizedOddsBook]


@dataclass(slots=True)
class NormalizedBenchmarkPrediction:
    home_team_name: str
    away_team_name: str
    season_year: int
    round_number: int
    source_name: str
    predicted_winner_name: str | None = None
    home_win_probability: float | None = None
    away_win_probability: float | None = None
    predicted_margin: float | None = None
    match_code: str | None = None


@dataclass(slots=True)
class NormalizedTeamMatchStats:
    match_code: str | None
    home_team_name: str
    away_team_name: str
    team_name: str
    stats: dict[str, Any]
    source_match_id: str | None = None


@dataclass(slots=True)
class NormalizedPlayerMatchStats:
    match_code: str | None
    team_name: str
    player_name: str
    stats: dict[str, Any]
    source_player_id: str | None = None
    source_match_id: str | None = None

