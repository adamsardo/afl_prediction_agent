from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

import pytest

from afl_prediction_agent.agents.codex_app_server import CodexAppServerAuthError, CodexAuthSnapshot, CodexTurnResult
from afl_prediction_agent.ingestion.services import FixtureIngestionService, OddsBookInput, SnapshotIngestionService
from afl_prediction_agent.orchestration import round_runs as round_runs_module
from afl_prediction_agent.orchestration.round_runs import RoundRunService
from afl_prediction_agent.storage.models import PlayerMatchStat, TeamMatchStat
from afl_prediction_agent.agents import adapters as adapters_module


def _seed_round(session):
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

    session.flush()
    return target_round, target_match


class FakeCodexClient:
    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.thread_names: set[str] = set()
        self.step_names: list[str] = []
        self._lock = threading.Lock()

    def preflight_auth(self) -> CodexAuthSnapshot:
        return CodexAuthSnapshot(
            auth_mode="chatgpt",
            email="user@example.com",
            account_plan_type="plus",
            effective_plan_type="pro",
            requires_openai_auth=True,
            supported_plan=True,
            rate_limits={"rateLimits": {"planType": "pro"}},
        )

    def run_turn(
        self,
        *,
        step_name: str,
        prompt: str,
        input_json: dict,
        model_name: str,
        reasoning_effort: str | None,
        output_schema: dict,
    ) -> CodexTurnResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        with self._lock:
            self.thread_names.add(threading.current_thread().name)
            self.step_names.append(step_name)
        dossier = input_json["dossier"]
        home_team_id = dossier["match"]["home_team"]["team_id"]
        away_team_id = dossier["match"]["away_team"]["team_id"]
        if step_name in {
            "form_analyst_v1",
            "selection_analyst_v1",
            "venue_weather_analyst_v1",
            "market_analyst_v1",
        }:
            output = {
                "summary": f"{step_name} summary",
                "signals": [
                    {
                        "label": step_name,
                        "leans_to": "home",
                        "strength": 0.66,
                        "evidence": "grounded in fake codex fixture",
                    }
                ],
                "risks": [],
                "unknowns": [],
            }
        elif step_name == "home_case_v1":
            output = {
                "side": "home",
                "case_summary": "home case",
                "strongest_points": [
                    {
                        "label": "home signal",
                        "strength": 0.7,
                        "evidence": "supported by analysts",
                    }
                ],
                "weak_points": [],
                "rebuttals": [],
            }
        elif step_name == "away_case_v1":
            output = {
                "side": "away",
                "case_summary": "away case",
                "strongest_points": [
                    {
                        "label": "away signal",
                        "strength": 0.4,
                        "evidence": "supported by analysts",
                    }
                ],
                "weak_points": [],
                "rebuttals": [],
            }
        else:
            output = {
                "predicted_winner_team_id": home_team_id,
                "home_win_probability": 0.64,
                "away_win_probability": 0.36,
                "predicted_margin": 11,
                "confidence_score": 74,
                "top_drivers": [
                    {
                        "label": "selection strength",
                        "leans_to": "home",
                        "strength": 0.78,
                        "evidence": "fake codex driver",
                        "source_component": "selection_analyst_v1",
                    }
                ],
                "uncertainty_note": f"away team {away_team_id} still has live chances",
                "rationale_summary": "fake codex final decision",
            }
        return CodexTurnResult(
            output_text=json.dumps(output),
            tokens_input=321,
            tokens_output=123,
            provider_meta={
                "thread_id": f"thr_{step_name}",
                "turn_id": f"turn_{step_name}",
                "transport": "stdio",
                "auth_mode": "chatgpt",
                "plan_type": "pro",
            },
        )


def test_round_run_with_codex_provider_stores_provider_meta(session, monkeypatch) -> None:
    target_round, target_match = _seed_round(session)
    fake_client = FakeCodexClient()
    monkeypatch.setattr(adapters_module, "get_codex_app_server_client", lambda: fake_client)
    monkeypatch.setattr(round_runs_module, "get_codex_app_server_client", lambda: fake_client)

    service = RoundRunService(session)
    run = service.run_round(round_id=target_round.id, config_name="v1_agentic_codex_gpt54")

    assert run.status == "completed"
    match_detail = service.get_match_run_detail(run.id, target_match.id)
    assert len(match_detail.agent_steps) == 7
    assert all(step["model_provider"] == "codex_app_server" for step in match_detail.agent_steps)
    assert all(step["reasoning_effort"] == "xhigh" for step in match_detail.agent_steps)
    assert all(step["provider_meta"]["transport"] == "stdio" for step in match_detail.agent_steps)


def test_round_run_fails_preflight_without_codex_auth(session, monkeypatch) -> None:
    target_round, _ = _seed_round(session)

    class FailingClient:
        def preflight_auth(self):
            raise CodexAppServerAuthError("not logged in")

    monkeypatch.setattr(round_runs_module, "get_codex_app_server_client", lambda: FailingClient())

    service = RoundRunService(session)
    with pytest.raises(CodexAppServerAuthError):
        service.run_round(round_id=target_round.id, config_name="v1_agentic_codex_gpt54")


def test_round_run_emits_progress_and_parallelizes_independent_steps(session, monkeypatch) -> None:
    target_round, _ = _seed_round(session)
    fake_client = FakeCodexClient(delay_seconds=0.05)
    progress_messages: list[str] = []
    monkeypatch.setattr(adapters_module, "get_codex_app_server_client", lambda: fake_client)
    monkeypatch.setattr(round_runs_module, "get_codex_app_server_client", lambda: fake_client)

    started_at = time.perf_counter()
    run = RoundRunService(session).run_round(
        round_id=target_round.id,
        config_name="v1_agentic_codex_gpt54",
        progress_callback=progress_messages.append,
    )
    elapsed = time.perf_counter() - started_at

    assert run.status == "completed"
    assert any(message.startswith("Started run ") for message in progress_messages)
    assert any("Eligible matches: 1/1" in message for message in progress_messages)
    assert any("starting analyst wave" in message for message in progress_messages)
    assert any("completed form_analyst_v1" in message for message in progress_messages)
    assert any("completed final_decision_v1" in message for message in progress_messages)
    assert len(fake_client.thread_names) >= 2
    assert elapsed < 0.3
