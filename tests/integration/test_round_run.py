from __future__ import annotations

from datetime import datetime, timedelta, timezone

from afl_prediction_agent.ingestion.services import FixtureIngestionService, OddsBookInput, SnapshotIngestionService
from afl_prediction_agent.orchestration import round_runs as round_runs_module
from afl_prediction_agent.orchestration.round_runs import EvaluationService, ReplayService, RoundRunService
from afl_prediction_agent.storage.models import AuditEvent, BenchmarkPrediction, PlayerMatchStat, TeamMatchStat


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
    assert detail.eligible_match_count == 1
    assert detail.skipped_match_count == 0
    assert detail.processed_match_count == 1
    assert detail.baseline_only_match_count == 0
    assert detail.verdict_count == 1
    match_detail = round_service.get_match_run_detail(run.id, target_match.id)
    assert match_detail.match_status == "completed"
    assert match_detail.analyst_summaries
    assert match_detail.case_summaries
    assert match_detail.bookmaker_snapshot is not None
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
    summaries = evaluation_service.get_season_summaries(season.id)
    assert summaries
    latest_summary = summaries[-1]["summary"]
    assert "bookmaker_favourite" in latest_summary["tracks"]
    assert "squiggle" in latest_summary["tracks"]
    assert "naive_home" in latest_summary["tracks"]
    assert "recent_form" in latest_summary["tracks"]
    assert latest_summary["agent_confidence_bands"]


def test_round_run_skips_only_matches_missing_lineups(session) -> None:
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
    swans = fixture_service.upsert_team(team_code="SYD", name="Sydney", short_name="Swans", state_code="NSW")
    dockers = fixture_service.upsert_team(team_code="FRE", name="Fremantle", short_name="Dockers", state_code="WA")
    venue = fixture_service.upsert_venue(name="MCG", venue_code="MCG", city="Melbourne", state_code="VIC")
    scg = fixture_service.upsert_venue(name="SCG", venue_code="SCG", city="Sydney", state_code="NSW")
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
    skipped_match = fixture_service.upsert_match(
        season_id=season.id,
        round_id=target_round.id,
        home_team_id=swans.id,
        away_team_id=dockers.id,
        venue_id=scg.id,
        scheduled_at=datetime(2026, 4, 25, 6, 10, tzinfo=timezone.utc),
        status="scheduled",
        match_code="2026R7SYDvFRE",
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
    snapshot_service.store_lineup_snapshot(
        match_id=skipped_match.id,
        team_id=swans.id,
        source_name="afl",
        fetched_at=fetched_at,
        payload={},
        players=[{"source_player_name": "Errol Gulden"}],
    )

    run = RoundRunService(session).run_round(round_id=target_round.id, config_name="v1_agentic_default")

    detail = RoundRunService(session).get_run_detail(run.id)
    assert detail.status == "completed_with_warnings"
    assert detail.match_count == 2
    assert detail.eligible_match_count == 1
    assert detail.skipped_match_count == 1
    skipped_detail = RoundRunService(session).get_match_run_detail(run.id, skipped_match.id)
    assert skipped_detail.match_status == "skipped"
    assert skipped_detail.skip_reason == "missing_official_lineup"
    assert session.query(AuditEvent).filter(AuditEvent.event_type == "match_excluded").count() == 1


def test_round_run_preserves_baseline_when_final_agent_fails(session, monkeypatch) -> None:
    fixture_service = FixtureIngestionService(session)
    snapshot_service = SnapshotIngestionService(session)
    competition = fixture_service.get_or_create_competition("AFL", "Australian Football League")
    season = fixture_service.get_or_create_season(competition.id, 2026)
    prior_round = fixture_service.get_or_create_round(season_id=season.id, round_number=6, round_name="Round 6")
    target_round = fixture_service.get_or_create_round(season_id=season.id, round_number=7, round_name="Round 7")
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
        home_score=90,
        away_score=79,
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

    class FinalFailingAdapter:
        def run_structured(
            self,
            *,
            step_name: str,
            prompt: str,
            input_json: dict,
            model_name: str,
            temperature: float | None,
            reasoning_effort: str | None,
            output_schema: dict,
        ):
            if step_name in {
                "form_analyst_v1",
                "selection_analyst_v1",
                "venue_weather_analyst_v1",
                "market_analyst_v1",
            }:
                return type(
                    "Result",
                    (),
                    {
                        "output_json": {
                            "summary": step_name,
                            "signals": [
                                {
                                    "label": step_name,
                                    "leans_to": "home",
                                    "strength": 0.6,
                                    "evidence": "test",
                                }
                            ],
                            "risks": [],
                            "unknowns": [],
                        },
                        "tokens_input": 10,
                        "tokens_output": 10,
                        "provider_meta": {},
                    },
                )()
            if step_name in {"home_case_v1", "away_case_v1"}:
                side = "home" if step_name == "home_case_v1" else "away"
                return type(
                    "Result",
                    (),
                    {
                        "output_json": {
                            "side": side,
                            "case_summary": side,
                            "strongest_points": [
                                {"label": side, "strength": 0.5, "evidence": "test"}
                            ],
                            "weak_points": [],
                            "rebuttals": [],
                        },
                        "tokens_input": 10,
                        "tokens_output": 10,
                        "provider_meta": {},
                    },
                )()
            raise RuntimeError("final step unavailable")

    monkeypatch.setattr(round_runs_module, "get_codex_app_server_client", lambda: None)
    from afl_prediction_agent.agents import runner as runner_module

    monkeypatch.setattr(runner_module, "build_adapter", lambda provider_name: FinalFailingAdapter())

    service = RoundRunService(session)
    run = service.run_round(round_id=target_round.id, config_name="v1_agentic_default")
    detail = service.get_run_detail(run.id)
    assert detail.status == "completed_with_warnings"
    assert detail.baseline_only_match_count == 1
    assert detail.verdict_count == 0
    match_detail = service.get_match_run_detail(run.id, target_match.id)
    assert match_detail.match_status == "baseline_only"
    assert match_detail.final_verdict is None
    assert match_detail.baseline_prediction is not None


def test_replay_round_creates_replay_audit_and_evaluation_summary(session) -> None:
    fixture_service = FixtureIngestionService(session)
    snapshot_service = SnapshotIngestionService(session)
    competition = fixture_service.get_or_create_competition("AFL", "Australian Football League")
    season = fixture_service.get_or_create_season(competition.id, 2026)
    prior_round = fixture_service.get_or_create_round(season_id=season.id, round_number=6, round_name="Round 6")
    target_round = fixture_service.get_or_create_round(season_id=season.id, round_number=7, round_name="Round 7")
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
        status="completed",
        match_code="2026R7CARvGEE",
        home_score=88,
        away_score=76,
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

    replay_service = ReplayService(session)
    run = replay_service.replay_round(round_id=target_round.id, config_name="v1_agentic_default")
    rows = EvaluationService(session).evaluate_run(run.id)

    assert run.status in {"completed", "completed_with_warnings"}
    assert rows
    assert session.query(AuditEvent).filter(AuditEvent.event_type == "historical_replay").count() == 1
