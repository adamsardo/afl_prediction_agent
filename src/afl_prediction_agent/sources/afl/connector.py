from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.sources.common.models import (
    FetchEnvelope,
    NormalizedFixtureMatch,
    NormalizedLineupPlayer,
    NormalizedLineupSnapshot,
    NormalizedPlayerMatchStats,
    NormalizedTeam,
    NormalizedTeamMatchStats,
    NormalizedVenue,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coalesce(row: dict[str, Any], *keys: str):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _int_or_none(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


@dataclass(slots=True)
class FitzRoyResponse:
    envelope: FetchEnvelope
    rows: list[dict[str, Any]]


class FitzRoyBridge:
    def __init__(
        self,
        *,
        rscript_bin: str | None = None,
        timeout_seconds: float | None = None,
        bridge_script: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.rscript_bin = rscript_bin or settings.fitzroy_rscript_bin
        self.timeout_seconds = timeout_seconds or settings.fitzroy_timeout_seconds
        self.bridge_script = bridge_script or Path(__file__).with_name("fitzroy_bridge.R")

    def fetch(
        self,
        dataset: str,
        *,
        season_year: int,
        round_number: int | None = None,
        source: str = "AFL",
    ) -> FitzRoyResponse:
        payload = {
            "season": season_year,
            "round_number": round_number,
            "source": source,
            "comp": "AFLM",
        }
        command = [
            self.rscript_bin,
            str(self.bridge_script),
            dataset,
            json.dumps(payload),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "fitzRoy bridge failed")
        rows = json.loads(result.stdout or "[]")
        if isinstance(rows, dict):
            rows = [rows]
        return FitzRoyResponse(
            envelope=FetchEnvelope(
                source_name="afl_official" if source == "AFL" else "afl_tables",
                fetched_at=_utcnow(),
                request_meta={"dataset": dataset, "season_year": season_year, "round_number": round_number},
                response_meta={"row_count": len(rows)},
                raw_payload={"rows": rows},
            ),
            rows=rows,
        )


class FitzRoyConnector:
    source_name = "afl_official"

    def __init__(self, bridge: FitzRoyBridge | None = None, *, source: str = "AFL") -> None:
        self.bridge = bridge or FitzRoyBridge()
        self.source = source

    def fetch_fixtures(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
    ) -> tuple[FetchEnvelope, list[NormalizedFixtureMatch]]:
        response = self.bridge.fetch(
            "fixtures",
            season_year=season_year,
            round_number=round_number,
            source=self.source,
        )
        normalized = [self._fixture_row_to_match(row) for row in response.rows]
        return response.envelope, normalized

    def fetch_results(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
    ) -> tuple[FetchEnvelope, list[NormalizedFixtureMatch]]:
        response = self.bridge.fetch(
            "results",
            season_year=season_year,
            round_number=round_number,
            source=self.source,
        )
        normalized = [self._fixture_row_to_match(row) for row in response.rows]
        return response.envelope, normalized

    def fetch_lineups(
        self,
        *,
        season_year: int,
        round_number: int,
    ) -> tuple[FetchEnvelope, list[NormalizedLineupSnapshot]]:
        response = self.bridge.fetch(
            "lineups",
            season_year=season_year,
            round_number=round_number,
            source=self.source,
        )
        snapshots: list[NormalizedLineupSnapshot] = []
        for row in response.rows:
            side = str(_coalesce(row, "match.homeaway", "home_away", "homeAway") or "").lower()
            if side.startswith("h"):
                side_name = "home"
            elif side.startswith("a"):
                side_name = "away"
            else:
                side_name = "home" if str(_coalesce(row, "team", "team.name", "team_name")) == str(_coalesce(row, "home.team.name", "home_team")) else "away"
            snapshots.append(
                NormalizedLineupSnapshot(
                    match_code=str(_coalesce(row, "providerId", "match.id", "id", "match_id") or ""),
                    home_or_away=side_name,
                    team=NormalizedTeam(
                        name=str(_coalesce(row, "team.name", "team", "team_name")),
                        short_name=_coalesce(row, "team.abbreviation", "team.shortName", "team_short_name"),
                        team_code=_coalesce(row, "team.abbreviation", "team_code"),
                        external_id=str(_coalesce(row, "team.providerId", "team.id", "team_id") or ""),
                    ),
                    players=[
                        NormalizedLineupPlayer(
                            source_player_name=str(_coalesce(row, "player.name", "player", "player_name")),
                            source_player_id=str(_coalesce(row, "player.providerId", "player.id", "player_id") or "")
                            or None,
                            slot_label=_coalesce(row, "position", "slot", "slot_label"),
                            named_role=_coalesce(row, "role", "named_role"),
                            is_selected=bool(_coalesce(row, "isSelected", "is_selected") if _coalesce(row, "isSelected", "is_selected") is not None else True),
                            is_interchange=bool(_coalesce(row, "isInterchange", "is_interchange") or False),
                            is_emergency=bool(_coalesce(row, "isEmergency", "is_emergency") or False),
                            is_sub=bool(_coalesce(row, "isSub", "is_sub") or False),
                        )
                    ],
                    published_at=None,
                )
            )
        # fitzRoy lineup responses are often one row per player; group by match/team/side
        grouped: dict[tuple[str, str, str], NormalizedLineupSnapshot] = {}
        for snapshot in snapshots:
            key = (
                snapshot.match_code or "",
                snapshot.team.external_id or snapshot.team.name,
                snapshot.home_or_away,
            )
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = snapshot
            else:
                existing.players.extend(snapshot.players)
        response.envelope.response_meta["snapshot_count"] = len(grouped)
        return response.envelope, list(grouped.values())

    def fetch_team_stats(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
    ) -> tuple[FetchEnvelope, list[NormalizedTeamMatchStats]]:
        response = self.bridge.fetch(
            "team_stats",
            season_year=season_year,
            round_number=round_number,
            source=self.source,
        )
        normalized: list[NormalizedTeamMatchStats] = []
        for row in response.rows:
            normalized.append(
                NormalizedTeamMatchStats(
                    match_code=str(_coalesce(row, "providerId", "match.id", "match_id", "id") or ""),
                    source_match_id=str(_coalesce(row, "providerId", "match.id", "match_id", "id") or ""),
                    home_team_name=str(_coalesce(row, "home.team.name", "home_team", "Home.Team")),
                    away_team_name=str(_coalesce(row, "away.team.name", "away_team", "Away.Team")),
                    team_name=str(_coalesce(row, "team", "team.name", "team_name")),
                    stats={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "providerId",
                            "match.id",
                            "match_id",
                            "id",
                            "home.team.name",
                            "away.team.name",
                            "team",
                            "team.name",
                            "team_name",
                        }
                    },
                )
            )
        return response.envelope, normalized

    def fetch_player_stats(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
    ) -> tuple[FetchEnvelope, list[NormalizedPlayerMatchStats]]:
        response = self.bridge.fetch(
            "player_stats",
            season_year=season_year,
            round_number=round_number,
            source=self.source,
        )
        normalized: list[NormalizedPlayerMatchStats] = []
        for row in response.rows:
            normalized.append(
                NormalizedPlayerMatchStats(
                    match_code=str(_coalesce(row, "providerId", "match.id", "match_id", "id") or ""),
                    source_match_id=str(_coalesce(row, "providerId", "match.id", "match_id", "id") or ""),
                    team_name=str(_coalesce(row, "team", "team.name", "team_name")),
                    player_name=str(_coalesce(row, "player.name", "player", "player_name")),
                    source_player_id=str(_coalesce(row, "player.providerId", "player.id", "player_id") or "")
                    or None,
                    stats={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "providerId",
                            "match.id",
                            "match_id",
                            "id",
                            "team",
                            "team.name",
                            "team_name",
                            "player",
                            "player.name",
                            "player_name",
                            "player.providerId",
                            "player.id",
                            "player_id",
                        }
                    },
                )
            )
        return response.envelope, normalized

    def _fixture_row_to_match(self, row: dict[str, Any]) -> NormalizedFixtureMatch:
        scheduled_at = _coalesce(
            row,
            "utcStartTime",
            "localStartTime",
            "date",
            "scheduled_at",
            "startTime",
        )
        if isinstance(scheduled_at, str):
            scheduled = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        else:
            scheduled = _utcnow()
        return NormalizedFixtureMatch(
            season_year=int(_coalesce(row, "season", "season.year", "seasonYear") or scheduled.year),
            round_number=int(_coalesce(row, "round.roundNumber", "round_number", "round") or 0),
            round_name=str(
                _coalesce(row, "round.name", "round_name", "roundName")
                or f"Round {_coalesce(row, 'round.roundNumber', 'round_number', 'round')}"
            ),
            scheduled_at=scheduled,
            home_team=NormalizedTeam(
                name=str(_coalesce(row, "home.team.name", "home_team", "home.name")),
                short_name=_coalesce(row, "home.team.nickname", "home.short_name"),
                team_code=_coalesce(row, "home.team.abbreviation", "home.team.code", "home_code"),
                external_id=str(_coalesce(row, "home.team.providerId", "home.team.id", "home_team_id") or "")
                or None,
                state_code=_coalesce(row, "home.team.state", "home_state"),
            ),
            away_team=NormalizedTeam(
                name=str(_coalesce(row, "away.team.name", "away_team", "away.name")),
                short_name=_coalesce(row, "away.team.nickname", "away.short_name"),
                team_code=_coalesce(row, "away.team.abbreviation", "away.team.code", "away_code"),
                external_id=str(_coalesce(row, "away.team.providerId", "away.team.id", "away_team_id") or "")
                or None,
                state_code=_coalesce(row, "away.team.state", "away_state"),
            ),
            venue=NormalizedVenue(
                name=str(_coalesce(row, "venue.name", "venue", "venue_name")),
                city=_coalesce(row, "venue.city", "venue_city"),
                state_code=_coalesce(row, "venue.state", "venue_state"),
                external_id=str(_coalesce(row, "venue.providerId", "venue.id", "venue_id") or "")
                or None,
            ),
            is_finals=bool(_coalesce(row, "round.isFinals", "is_finals") or False),
            match_code=str(_coalesce(row, "providerId", "id", "match.id", "match_id") or "") or None,
            status=str(_coalesce(row, "status", "match_status") or "scheduled"),
            home_score=_int_or_none(_coalesce(row, "home.score.total", "home_score")),
            away_score=_int_or_none(_coalesce(row, "away.score.total", "away_score")),
        )

