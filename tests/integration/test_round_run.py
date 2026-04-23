from __future__ import annotations

from datetime import datetime, timedelta, timezone

from afl_prediction_agent.ingestion.services import FixtureIngestionService, OddsBookInput, SnapshotIngestionService
from afl_prediction_agent.orchestration.round_runs import EvaluationService, RoundRunService
from afl_prediction_agent.storage.models import BenchmarkPrediction, PlayerMatchStat, TeamMatchStat


def test_round_run_and_evaluation_flow(session) -> None:
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
                stats={"inside_50": 58, "clearances": 39},
            ),
            TeamMatchStat(
                match_id=prior_match.id,
                team_id=cats.id,
                source_name="fitzroy",
                stats={"inside_50": 47, "clearances": 31},
            ),
        ]
    )

    home_players = [
        fixture_service.upsert_player(full_name="Patrick Cripps", current_team_id=blues.id),
        fixture_service.upsert_player(full_name="Sam Walsh", current_team_id=blues.id),
    ]
    away_players = [
        fixture_service.upsert_player(full_name="Jeremy Cameron", current_team_id=cats.id),
        fixture_service.upsert_player(full_name="Tom Stewart", current_team_id=cats.id),
    ]

    for player in home_players + away_players:
        session.add(
            PlayerMatchStat(
                match_id=prior_match.id,
                team_id=player.current_team_id,
                player_id=player.id,
                source_name="fitzroy",
                stats={"rating": 8.5, "disposals": 24},
            )
        )

    fetched_at = datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc)
    snapshot_service.store_lineup_snapshot(
        match_id=target_match.id,
        team_id=blues.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={"team": "Carlton"},
        players=[
            {"player_id": home_players[0].id, "source_player_name": home_players[0].full_name},
            {"player_id": home_players[1].id, "source_player_name": home_players[1].full_name},
        ],
    )
    snapshot_service.store_lineup_snapshot(
        match_id=target_match.id,
        team_id=cats.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={"team": "Geelong"},
        players=[
            {"player_id": away_players[0].id, "source_player_name": away_players[0].full_name},
            {"player_id": away_players[1].id, "source_player_name": away_players[1].full_name},
        ],
    )
    snapshot_service.store_injury_snapshot(
        source_name="afl_injuries",
        fetched_at=fetched_at,
        payload={},
        entries=[
            {
                "team_id": cats.id,
                "player_id": away_players[0].id,
                "source_player_name": away_players[0].full_name,
                "status_label": "test",
                "injury_note": "hamstring tightness",
                "estimated_return_text": "round 7",
                "uncertainty_flag": True,
            }
        ],
    )
    snapshot_service.store_weather_snapshot(
        match_id=target_match.id,
        venue_id=venue.id,
        source_name="bom",
        fetched_at=fetched_at,
        payload={},
        temperature_c=16.0,
        rain_probability_pct=35.0,
        rainfall_mm=0.8,
        wind_kmh=18.0,
        weather_text="showers easing",
    )
    snapshot_service.store_odds_snapshot(
        match_id=target_match.id,
        source_name="odds",
        fetched_at=fetched_at,
        payload={},
        books=[
            OddsBookInput("tab", "h2h", 1.72, 2.10),
            OddsBookInput("sb", "h2h", 1.75, 2.05),
            OddsBookInput("pb", "h2h", 1.78, 2.02),
        ],
    )

    round_service = RoundRunService(session)
    run = round_service.run_round(round_id=target_round.id, config_name="v1_agentic_default")
    session.add(
        BenchmarkPrediction(
            round_run_id=run.id,
            match_id=target_match.id,
            source_name="squiggle",
            predicted_winner_team_id=blues.id,
            home_win_probability=0.58,
            away_win_probability=0.42,
            predicted_margin=9.0,
            payload={},
        )
    )
    session.flush()

    assert run.status == "completed"
    detail = round_service.get_run_detail(run.id)
    assert detail.verdict_count == 1
    match_detail = round_service.get_match_run_detail(run.id, target_match.id)
    assert match_detail.final_verdict is not None
    assert len(match_detail.agent_steps) == 7

    target_match.status = "completed"
    target_match.home_score = 88
    target_match.away_score = 76
    target_match.winning_team_id = blues.id
    target_match.actual_margin = 12
    session.flush()

    evaluation_service = EvaluationService(session)
    rows = evaluation_service.evaluate_run(run.id)
    assert len(rows) == 1
    assert rows[0].agent_winner_correct is True
