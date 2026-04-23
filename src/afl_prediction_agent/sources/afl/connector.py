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


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_datetime_or_now(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return _utcnow()


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
            side = str(_coalesce(row, "teamType", "match.homeaway", "home_away", "homeAway") or "").lower()
            if side.startswith("h"):
                side_name = "home"
            elif side.startswith("a"):
                side_name = "away"
            else:
                side_name = "home" if str(_coalesce(row, "teamName", "team", "team.name", "team_name")) == str(_coalesce(row, "home.team.name", "home_team")) else "away"
            given_name = _coalesce(row, "player.playerName.givenName")
            surname = _coalesce(row, "player.playerName.surname")
            combined_name = " ".join(part for part in [given_name, surname] if part)
            snapshots.append(
                NormalizedLineupSnapshot(
                    match_code=str(_coalesce(row, "providerId", "match.id", "id", "match_id") or ""),
                    home_or_away=side_name,
                    team=NormalizedTeam(
                        name=str(_coalesce(row, "teamName", "team.name", "team", "team_name")),
                        short_name=_coalesce(row, "teamNickname", "team.shortName", "team_short_name"),
                        team_code=_coalesce(row, "teamAbbr", "team.abbreviation", "team_code"),
                        external_id=str(_coalesce(row, "teamId", "team.providerId", "team.id", "team_id") or ""),
                    ),
                    players=[
                        NormalizedLineupPlayer(
                            source_player_name=str(_coalesce(row, "player.name", "player", "player_name") or combined_name),
                            source_player_id=str(_coalesce(row, "player.playerId", "player.providerId", "player.id", "player_id") or "")
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
        try:
            response = self.bridge.fetch(
                "team_stats",
                season_year=season_year,
                round_number=round_number,
                source=self.source,
            )
        except RuntimeError as exc:
            if self.source != "AFL":
                raise
            return self._derive_team_stats_from_player_stats(
                season_year=season_year,
                round_number=round_number,
            )
        if not self._rows_support_match_level_team_stats(response.rows):
            return self._derive_team_stats_from_player_stats(
                season_year=season_year,
                round_number=round_number,
            )
        normalized: list[NormalizedTeamMatchStats] = []
        for row in response.rows:
            normalized.append(
                NormalizedTeamMatchStats(
                    match_code=str(_coalesce(row, "providerId", "match.id", "match_id", "id", "Game") or ""),
                    source_match_id=str(_coalesce(row, "providerId", "match.id", "match_id", "id", "Game") or ""),
                    home_team_name=str(_coalesce(row, "home.team.name", "home_team", "Home.Team") or ""),
                    away_team_name=str(_coalesce(row, "away.team.name", "away_team", "Away.Team") or ""),
                    team_name=str(_coalesce(row, "Team", "team", "team.name", "team_name")),
                    stats={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "providerId",
                            "match.id",
                            "match_id",
                            "id",
                            "Game",
                            "home.team.name",
                            "away.team.name",
                            "Home.Team",
                            "Away.Team",
                            "Team",
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
            given_name = _coalesce(
                row,
                "player.givenName",
                "player.player.player.givenName",
                "player.playerName.givenName",
                "First.name",
            )
            surname = _coalesce(
                row,
                "player.surname",
                "player.player.player.surname",
                "player.playerName.surname",
                "Surname",
            )
            combined_name = " ".join(part for part in [given_name, surname] if part)
            normalized.append(
                NormalizedPlayerMatchStats(
                    match_code=str(
                        _coalesce(row, "providerId", "match.id", "match_id", "id")
                        or self._synthetic_match_code(row)
                        or ""
                    ),
                    source_match_id=str(
                        _coalesce(row, "providerId", "match.id", "match_id", "id")
                        or self._synthetic_match_code(row)
                        or ""
                    ),
                    team_name=str(_coalesce(row, "team.name", "team", "team_name", "Team", "Playing.for")),
                    player_name=str(_coalesce(row, "player.name", "player", "player_name") or combined_name),
                    source_player_id=str(
                        _coalesce(
                            row,
                            "player.playerId",
                            "player.player.player.playerId",
                            "player.playerId",
                            "player.providerId",
                            "player.id",
                            "player_id",
                            "ID",
                        )
                        or ""
                    )
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

    def _rows_support_match_level_team_stats(self, rows: list[dict[str, Any]]) -> bool:
        for row in rows:
            match_id = _coalesce(row, "providerId", "match.id", "match_id", "id", "Game")
            home_team = _coalesce(row, "home.team.name", "Home.Team")
            away_team = _coalesce(row, "away.team.name", "Away.Team")
            if match_id and (home_team or away_team):
                return True
        return False

    def _derive_team_stats_from_player_stats(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
    ) -> tuple[FetchEnvelope, list[NormalizedTeamMatchStats]]:
        response = self.bridge.fetch(
            "player_stats",
            season_year=season_year,
            round_number=round_number,
            source=self.source,
        )
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in response.rows:
            team_name = str(_coalesce(row, "team.name", "team", "team_name", "Team", "Playing.for") or "")
            if not team_name:
                continue
            match_code = str(
                _coalesce(row, "providerId", "match.id", "match_id", "id")
                or self._synthetic_match_code(row)
                or ""
            )
            if not match_code:
                continue
            key = (match_code, team_name)
            bucket = grouped.setdefault(
                key,
                {
                    "match_code": match_code,
                    "source_match_id": match_code,
                    "home_team_name": str(_coalesce(row, "home.team.name", "Home.Team", "Home.team") or ""),
                    "away_team_name": str(_coalesce(row, "away.team.name", "Away.Team", "Away.team") or ""),
                    "team_name": team_name,
                    "stats": {
                        "inside_50": 0.0,
                        "clearances": 0.0,
                        "contested_possessions": 0.0,
                        "marks_inside_50": 0.0,
                        "tackles_inside_50": 0.0,
                        "disposals": 0.0,
                        "kicks": 0.0,
                        "handballs": 0.0,
                        "marks": 0.0,
                        "goals": 0.0,
                        "behinds": 0.0,
                        "hitouts": 0.0,
                        "rebound_50": 0.0,
                        "score_involvements": 0.0,
                        "metres_gained": 0.0,
                        "tackles": 0.0,
                        "one_percenters": 0.0,
                        "uncontested_possessions": 0.0,
                        "clangers": 0.0,
                        "frees_for": 0.0,
                        "frees_against": 0.0,
                    },
                },
            )
            canonical = bucket["stats"]
            canonical["inside_50"] += _float_or_none(_coalesce(row, "inside50s", "Inside.50s")) or 0.0
            canonical["clearances"] += _float_or_none(
                _coalesce(row, "clearances.totalClearances", "Clearances")
            ) or 0.0
            canonical["contested_possessions"] += _float_or_none(
                _coalesce(row, "contestedPossessions", "Contested.Possessions")
            ) or 0.0
            canonical["marks_inside_50"] += _float_or_none(
                _coalesce(row, "marksInside50", "Marks.Inside.50")
            ) or 0.0
            canonical["tackles_inside_50"] += _float_or_none(
                _coalesce(row, "tacklesInside50")
            ) or 0.0
            canonical["disposals"] += _float_or_none(_coalesce(row, "disposals", "Disposals")) or 0.0
            canonical["kicks"] += _float_or_none(_coalesce(row, "kicks", "Kicks")) or 0.0
            canonical["handballs"] += _float_or_none(_coalesce(row, "handballs", "Handballs")) or 0.0
            canonical["marks"] += _float_or_none(_coalesce(row, "marks", "Marks")) or 0.0
            canonical["goals"] += _float_or_none(_coalesce(row, "goals", "Goals")) or 0.0
            canonical["behinds"] += _float_or_none(_coalesce(row, "behinds", "Behinds")) or 0.0
            canonical["hitouts"] += _float_or_none(_coalesce(row, "hitouts", "Hit.Outs")) or 0.0
            canonical["rebound_50"] += _float_or_none(_coalesce(row, "rebound50s", "Rebounds")) or 0.0
            canonical["score_involvements"] += _float_or_none(_coalesce(row, "scoreInvolvements")) or 0.0
            canonical["metres_gained"] += _float_or_none(_coalesce(row, "metresGained")) or 0.0
            canonical["tackles"] += _float_or_none(_coalesce(row, "tackles", "Tackles")) or 0.0
            canonical["one_percenters"] += _float_or_none(_coalesce(row, "onePercenters", "One.Percenters")) or 0.0
            canonical["uncontested_possessions"] += _float_or_none(
                _coalesce(row, "uncontestedPossessions", "Uncontested.Possessions")
            ) or 0.0
            canonical["clangers"] += _float_or_none(_coalesce(row, "clangers", "Clangers")) or 0.0
            canonical["frees_for"] += _float_or_none(_coalesce(row, "freesFor", "Frees.For")) or 0.0
            canonical["frees_against"] += _float_or_none(_coalesce(row, "freesAgainst", "Frees.Against")) or 0.0
        normalized = [
            NormalizedTeamMatchStats(
                match_code=bucket["match_code"],
                source_match_id=bucket["source_match_id"],
                home_team_name=bucket["home_team_name"],
                away_team_name=bucket["away_team_name"],
                team_name=bucket["team_name"],
                stats=bucket["stats"],
            )
            for bucket in grouped.values()
        ]
        response.envelope.response_meta["derived_from_player_stats"] = True
        response.envelope.response_meta["row_count"] = len(normalized)
        response.envelope.raw_payload = {
            "derived_from_player_stats": True,
            "rows": response.rows,
        }
        return response.envelope, normalized

    def _synthetic_match_code(self, row: dict[str, Any]) -> str | None:
        season = _coalesce(row, "Season", "round.year", "season")
        round_number = _coalesce(row, "Round", "round.roundNumber", "round")
        match_date = _coalesce(row, "Date", "utcStartTime", "match.utcStartTime")
        home_team = _coalesce(row, "Home.Team", "Home.team", "home.team.name")
        away_team = _coalesce(row, "Away.Team", "Away.team", "away.team.name")
        if not all([season, round_number, match_date, home_team, away_team]):
            return None
        return f"{season}:{round_number}:{match_date}:{home_team}:{away_team}"

    def _fixture_row_to_match(self, row: dict[str, Any]) -> NormalizedFixtureMatch:
        scheduled = _parse_datetime_or_now(
            _coalesce(
                row,
                "match.utcStartTime",
                "utcStartTime",
                "match.date",
                "Date",
                "localStartTime",
                "date",
                "scheduled_at",
                "startTime",
            )
        )
        home_score = _int_or_none(
            _coalesce(
                row,
                "homeTeamScore.matchScore.totalScore",
                "home.score.total",
                "Home.Points",
                "home_score",
            )
        )
        away_score = _int_or_none(
            _coalesce(
                row,
                "awayTeamScore.matchScore.totalScore",
                "away.score.total",
                "Away.Points",
                "away_score",
            )
        )
        round_number = _coalesce(row, "round.roundNumber", "Round.Number", "round_number", "round")
        season_year = _coalesce(row, "round.year", "Season", "season", "season.year", "seasonYear")
        status = _coalesce(row, "match.status", "status", "match_status")
        if status is None and home_score is not None and away_score is not None:
            status = "CONCLUDED"
        return NormalizedFixtureMatch(
            season_year=int(season_year or scheduled.year),
            round_number=int(round_number or 0),
            round_name=str(
                _coalesce(row, "round.name", "Round", "round_name", "roundName")
                or f"Round {round_number}"
            ),
            scheduled_at=scheduled,
            home_team=NormalizedTeam(
                name=str(_coalesce(row, "match.homeTeam.name", "home.team.name", "Home.Team", "home_team", "home.name")),
                short_name=_coalesce(row, "match.homeTeam.nickname", "home.team.nickname", "home.short_name"),
                team_code=_coalesce(row, "match.homeTeam.abbr", "home.team.abbreviation", "home.team.code", "home_code"),
                external_id=str(_coalesce(row, "match.homeTeam.teamId", "home.team.providerId", "home.team.id", "home_team_id") or "")
                or None,
                state_code=_coalesce(row, "match.homeTeam.state", "home.team.state", "home_state"),
            ),
            away_team=NormalizedTeam(
                name=str(_coalesce(row, "match.awayTeam.name", "away.team.name", "Away.Team", "away_team", "away.name")),
                short_name=_coalesce(row, "match.awayTeam.nickname", "away.team.nickname", "away.short_name"),
                team_code=_coalesce(row, "match.awayTeam.abbr", "away.team.abbreviation", "away.team.code", "away_code"),
                external_id=str(_coalesce(row, "match.awayTeam.teamId", "away.team.providerId", "away.team.id", "away_team_id") or "")
                or None,
                state_code=_coalesce(row, "match.awayTeam.state", "away.team.state", "away_state"),
            ),
            venue=NormalizedVenue(
                name=str(_coalesce(row, "venue.name", "Venue", "venue", "venue_name")),
                city=_coalesce(row, "venue.address", "venue.city", "venue_city"),
                state_code=_coalesce(row, "venue.state", "venue_state"),
                external_id=str(_coalesce(row, "venue.venueId", "venue.providerId", "venue.id", "venue_id", "match.venue") or "")
                or None,
                timezone_name=_coalesce(row, "venue.timeZone") or "Australia/Melbourne",
                latitude=_coalesce(row, "venue.latitude"),
                longitude=_coalesce(row, "venue.longitude"),
            ),
            is_finals=bool(_coalesce(row, "round.isFinals", "is_finals") or False),
            match_code=str(_coalesce(row, "match.matchId", "Game", "providerId", "id", "match.id", "match_id") or "")
            or None,
            status=str(status or "scheduled"),
            home_score=home_score,
            away_score=away_score,
        )
