from __future__ import annotations

import uuid
from datetime import datetime, timezone

from afl_prediction_agent.ingestion.services import FixtureIngestionService
from afl_prediction_agent.sources.common.models import (
    FetchEnvelope,
    NormalizedBenchmarkPrediction,
    NormalizedInjuryEntry,
    NormalizedInjurySnapshot,
    NormalizedLineupPlayer,
    NormalizedLineupSnapshot,
    NormalizedOddsBook,
    NormalizedOddsSnapshot,
    NormalizedTeam,
    NormalizedWeatherSnapshot,
)
from afl_prediction_agent.sources.service import RoundSourceSyncService
from afl_prediction_agent.storage.models import BenchmarkPrediction, InjurySnapshot, OddsSnapshot, RoundRun, RunConfig, WeatherSnapshot


class FakeOfficialConnector:
    source_name = "afl_official"

    def fetch_lineups(self, *, season_year: int, round_number: int):
        fetched_at = datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc)
        return (
            FetchEnvelope(source_name=self.source_name, fetched_at=fetched_at),
            [
                NormalizedLineupSnapshot(
                    match_code="2026R7CARvGEE",
                    home_or_away="home",
                    team=NormalizedTeam(name="Carlton", team_code="CAR", external_id="CAR"),
                    players=[
                        NormalizedLineupPlayer("Patrick Cripps", source_player_id="p1"),
                        NormalizedLineupPlayer("Sam Walsh", source_player_id="p2"),
                    ],
                ),
                NormalizedLineupSnapshot(
                    match_code="2026R7CARvGEE",
                    home_or_away="away",
                    team=NormalizedTeam(name="Geelong", team_code="GEE", external_id="GEE"),
                    players=[
                        NormalizedLineupPlayer("Jeremy Cameron", source_player_id="p3"),
                        NormalizedLineupPlayer("Tom Stewart", source_player_id="p4"),
                    ],
                ),
            ],
        )


class FakeInjuryConnector:
    def __init__(self, source_name: str, status_label: str) -> None:
        self.source_name = source_name
        self.status_label = status_label

    def fetch_injuries(self):
        return (
            FetchEnvelope(source_name=self.source_name, fetched_at=datetime(2026, 4, 23, 9, 5, tzinfo=timezone.utc)),
            NormalizedInjurySnapshot(
                entries=[
                    NormalizedInjuryEntry(
                        team_name="Geelong",
                        source_player_name="Jeremy Cameron",
                        status_label=self.status_label,
                        injury_note="Hamstring",
                        estimated_return_text=self.status_label,
                        uncertainty_flag=True,
                    )
                ]
            ),
        )


class FakeBomConnector:
    source_name = "bom"

    def fetch_weather_for_venue(self, *, venue_name: str, scheduled_at: datetime):
        return (
            FetchEnvelope(source_name=self.source_name, fetched_at=datetime(2026, 4, 23, 9, 10, tzinfo=timezone.utc)),
            NormalizedWeatherSnapshot(
                venue_name=venue_name,
                forecast_for=scheduled_at,
                temperature_c=16.0,
                rain_probability_pct=35.0,
                rainfall_mm=None,
                wind_kmh=18.0,
                weather_text="Showers easing",
            ),
        )

    def mapping_for_venue(self, venue_name: str):
        return type(
            "Mapping",
            (),
            {
                "station_id": "95936",
                "forecast_location_name": "Melbourne",
                "forecast_district_name": None,
            },
        )()


class FakeOddsConnector:
    source_name = "the_odds_api"

    def fetch_head_to_head(self, *, as_of=None):
        return (
            FetchEnvelope(source_name=self.source_name, fetched_at=datetime(2026, 4, 23, 9, 15, tzinfo=timezone.utc)),
            [
                NormalizedOddsSnapshot(
                    home_team_name="Carlton",
                    away_team_name="Geelong",
                    commence_time=datetime(2026, 4, 24, 9, 50, tzinfo=timezone.utc),
                    books=[
                        NormalizedOddsBook("tab", "h2h", 1.72, 2.10),
                        NormalizedOddsBook("sportsbet", "h2h", 1.75, 2.05),
                    ],
                )
            ],
        )


class FakeSquiggleConnector:
    source_name = "squiggle"

    def fetch_predictions(self, *, season_year: int, round_number: int):
        return (
            FetchEnvelope(source_name=self.source_name, fetched_at=datetime(2026, 4, 23, 9, 20, tzinfo=timezone.utc)),
            [
                NormalizedBenchmarkPrediction(
                    home_team_name="Carlton",
                    away_team_name="Geelong",
                    season_year=season_year,
                    round_number=round_number,
                    source_name="aggregate",
                    predicted_winner_name="Carlton",
                    home_win_probability=0.58,
                    away_win_probability=0.42,
                    predicted_margin=9.0,
                    match_code="2026R7CARvGEE",
                )
            ],
        )


def test_source_sync_snapshot_round_persists_snapshots_and_benchmarks(session) -> None:
    fixture_service = FixtureIngestionService(session)
    competition = fixture_service.get_or_create_competition("AFL", "Australian Football League")
    season = fixture_service.get_or_create_season(competition.id, 2026)
    target_round = fixture_service.get_or_create_round(
        season_id=season.id,
        round_number=7,
        round_name="Round 7",
    )
    blues = fixture_service.upsert_team(
        team_code="CAR",
        name="Carlton",
        short_name="Blues",
        state_code="VIC",
    )
    cats = fixture_service.upsert_team(
        team_code="GEE",
        name="Geelong",
        short_name="Cats",
        state_code="VIC",
    )
    for player_name, team_id in [
        ("Patrick Cripps", blues.id),
        ("Sam Walsh", blues.id),
        ("Jeremy Cameron", cats.id),
        ("Tom Stewart", cats.id),
    ]:
        fixture_service.upsert_player(full_name=player_name, current_team_id=team_id)
    venue = fixture_service.upsert_venue(
        name="MCG",
        venue_code="MCG",
        city="Melbourne",
        state_code="VIC",
    )
    fixture_service.upsert_match(
        season_id=season.id,
        round_id=target_round.id,
        home_team_id=blues.id,
        away_team_id=cats.id,
        venue_id=venue.id,
        scheduled_at=datetime(2026, 4, 24, 19, 50, tzinfo=timezone.utc),
        status="scheduled",
        match_code="2026R7CARvGEE",
    )
    run_config = RunConfig(
        config_name="test_config",
        feature_version="features_v1",
        winner_model_version="winner_v1",
        margin_model_version="margin_v1",
        prompt_set_version="prompt_set_v1",
        final_model_provider="heuristic",
        final_model_name="heuristic-final-v1",
        default_temperature=0.2,
        config={},
    )
    session.add(run_config)
    session.flush()
    round_run = RoundRun(
        season_id=season.id,
        round_id=target_round.id,
        run_config_id=run_config.id,
        lock_timestamp=datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc),
        status="running",
    )
    session.add(round_run)
    session.flush()

    service = RoundSourceSyncService(
        session,
        official_connector=FakeOfficialConnector(),
        afl_com_connector=FakeInjuryConnector("afl_com", "Test"),
        footywire_connector=FakeInjuryConnector("footywire", "TBC"),
        bom_connector=FakeBomConnector(),
        odds_connector=FakeOddsConnector(),
        squiggle_connector=FakeSquiggleConnector(),
    )

    summaries = service.snapshot_round(round_id=target_round.id, round_run_id=round_run.id)

    assert any(summary.source_name == "afl_official" and summary.created == 2 for summary in summaries)
    assert session.query(InjurySnapshot).count() == 2
    assert session.query(WeatherSnapshot).count() == 1
    assert session.query(OddsSnapshot).count() == 1
    assert session.query(BenchmarkPrediction).count() == 1
