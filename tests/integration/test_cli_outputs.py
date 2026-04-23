from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from afl_prediction_agent.api.app import app as api_app
from afl_prediction_agent.cli import app
from afl_prediction_agent.ingestion.services import (
    FixtureIngestionService,
    OddsBookInput,
    SnapshotIngestionService,
)
from afl_prediction_agent.core.db.session import get_session
from afl_prediction_agent.orchestration.round_runs import RoundRunService
from afl_prediction_agent.storage.models import PlayerMatchStat, TeamMatchStat
from afl_prediction_agent import cli as cli_module


def test_show_run_and_show_match_commands(session, monkeypatch) -> None:
    fixture_service = FixtureIngestionService(session)
    snapshot_service = SnapshotIngestionService(session)

    competition = fixture_service.get_or_create_competition("AFL", "Australian Football League")
    season = fixture_service.get_or_create_season(competition.id, 2026)
    prior_round = fixture_service.get_or_create_round(
        season_id=season.id,
        round_number=6,
        round_name="Round 6",
    )
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
    venue = fixture_service.upsert_venue(
        name="MCG",
        venue_code="MCG",
        city="Melbourne",
        state_code="VIC",
    )
    prior_match = fixture_service.upsert_match(
        season_id=season.id,
        round_id=prior_round.id,
        home_team_id=blues.id,
        away_team_id=cats.id,
        venue_id=venue.id,
        scheduled_at=datetime(2026, 4, 17, 19, 50, tzinfo=timezone.utc),
        status="completed",
        match_code="2026R6CARvGEE",
        home_score=92,
        away_score=80,
    )
    target_match = fixture_service.upsert_match(
        season_id=season.id,
        round_id=target_round.id,
        home_team_id=blues.id,
        away_team_id=cats.id,
        venue_id=venue.id,
        scheduled_at=datetime(2026, 4, 24, 19, 50, tzinfo=timezone.utc),
        status="scheduled",
        match_code="2026R7CARvGEE",
    )
    session.add_all(
        [
            TeamMatchStat(
                match_id=prior_match.id,
                team_id=blues.id,
                source_name="fitzroy",
                stats={"inside_50": 58},
            ),
            TeamMatchStat(
                match_id=prior_match.id,
                team_id=cats.id,
                source_name="fitzroy",
                stats={"inside_50": 47},
            ),
        ]
    )
    home_player = fixture_service.upsert_player(full_name="Patrick Cripps", current_team_id=blues.id)
    away_player = fixture_service.upsert_player(full_name="Jeremy Cameron", current_team_id=cats.id)
    for player in [home_player, away_player]:
        session.add(
            PlayerMatchStat(
                match_id=prior_match.id,
                team_id=player.current_team_id,
                player_id=player.id,
                source_name="fitzroy",
                stats={"rating": 8.5},
            )
        )
    fetched_at = datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc)
    snapshot_service.store_lineup_snapshot(
        match_id=target_match.id,
        team_id=blues.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={},
        players=[{"player_id": home_player.id, "source_player_name": home_player.full_name}],
    )
    snapshot_service.store_lineup_snapshot(
        match_id=target_match.id,
        team_id=cats.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={},
        players=[{"player_id": away_player.id, "source_player_name": away_player.full_name}],
    )
    snapshot_service.store_odds_snapshot(
        match_id=target_match.id,
        source_name="odds",
        fetched_at=fetched_at,
        payload={},
        books=[
            OddsBookInput("tab", "h2h", 1.72, 2.10),
            OddsBookInput("sb", "h2h", 1.75, 2.05),
        ],
    )
    session.commit()

    run = RoundRunService(session).run_round(round_id=target_round.id, config_name="v1_agentic_default")
    session.commit()

    cli_session_factory = sessionmaker(bind=session.get_bind(), future=True)
    monkeypatch.setattr(cli_module, "SessionLocal", cli_session_factory)

    runner = CliRunner()
    run_result = runner.invoke(app, ["show-run", str(run.id)])
    assert run_result.exit_code == 0
    run_payload = json.loads(run_result.stdout)
    assert run_payload["run"]["run_id"] == str(run.id)
    assert len(run_payload["matches"]) == 1
    assert run_payload["matches"][0]["match_id"] == str(target_match.id)

    match_result = runner.invoke(app, ["show-match", str(run.id), str(target_match.id)])
    assert match_result.exit_code == 0
    match_payload = json.loads(match_result.stdout)
    assert match_payload["run_id"] == str(run.id)
    assert match_payload["match_id"] == str(target_match.id)
    assert match_payload["match_status"] == "completed"


def test_api_lists_run_matches(session) -> None:
    fixture_service = FixtureIngestionService(session)
    snapshot_service = SnapshotIngestionService(session)

    competition = fixture_service.get_or_create_competition("AFL", "Australian Football League")
    season = fixture_service.get_or_create_season(competition.id, 2026)
    prior_round = fixture_service.get_or_create_round(
        season_id=season.id,
        round_number=6,
        round_name="Round 6",
    )
    target_round = fixture_service.get_or_create_round(
        season_id=season.id,
        round_number=7,
        round_name="Round 7",
    )
    blues = fixture_service.upsert_team(team_code="CAR", name="Carlton", short_name="Blues", state_code="VIC")
    cats = fixture_service.upsert_team(team_code="GEE", name="Geelong", short_name="Cats", state_code="VIC")
    venue = fixture_service.upsert_venue(name="MCG", venue_code="MCG", city="Melbourne", state_code="VIC")
    prior_match = fixture_service.upsert_match(
        season_id=season.id,
        round_id=prior_round.id,
        home_team_id=blues.id,
        away_team_id=cats.id,
        venue_id=venue.id,
        scheduled_at=datetime(2026, 4, 17, 19, 50, tzinfo=timezone.utc),
        status="completed",
        match_code="2026R6CARvGEE",
        home_score=92,
        away_score=80,
    )
    target_match = fixture_service.upsert_match(
        season_id=season.id,
        round_id=target_round.id,
        home_team_id=blues.id,
        away_team_id=cats.id,
        venue_id=venue.id,
        scheduled_at=datetime(2026, 4, 24, 19, 50, tzinfo=timezone.utc),
        status="scheduled",
        match_code="2026R7CARvGEE",
    )
    session.add_all(
        [
            TeamMatchStat(match_id=prior_match.id, team_id=blues.id, source_name="fitzroy", stats={"inside_50": 58}),
            TeamMatchStat(match_id=prior_match.id, team_id=cats.id, source_name="fitzroy", stats={"inside_50": 47}),
        ]
    )
    home_player = fixture_service.upsert_player(full_name="Patrick Cripps", current_team_id=blues.id)
    away_player = fixture_service.upsert_player(full_name="Jeremy Cameron", current_team_id=cats.id)
    for player in [home_player, away_player]:
        session.add(
            PlayerMatchStat(
                match_id=prior_match.id,
                team_id=player.current_team_id,
                player_id=player.id,
                source_name="fitzroy",
                stats={"rating": 8.5},
            )
        )
    fetched_at = datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc)
    snapshot_service.store_lineup_snapshot(
        match_id=target_match.id,
        team_id=blues.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={},
        players=[{"player_id": home_player.id, "source_player_name": home_player.full_name}],
    )
    snapshot_service.store_lineup_snapshot(
        match_id=target_match.id,
        team_id=cats.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={},
        players=[{"player_id": away_player.id, "source_player_name": away_player.full_name}],
    )
    session.commit()
    run = RoundRunService(session).run_round(round_id=target_round.id, config_name="v1_agentic_default")
    session.commit()

    api_session_factory = sessionmaker(bind=session.get_bind(), future=True)

    def override_get_session():
        with api_session_factory() as db:
            yield db

    api_app.dependency_overrides[get_session] = override_get_session
    try:
        client = TestClient(api_app)
        response = client.get(f"/runs/{run.id}/matches")
    finally:
        api_app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["match_id"] == str(target_match.id)
    assert payload[0]["match_status"] == "completed"
