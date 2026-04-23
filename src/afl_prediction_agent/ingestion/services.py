from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from afl_prediction_agent.storage.models import (
    AuditEvent,
    BenchmarkPrediction,
    Competition,
    InjurySnapshot,
    InjurySnapshotEntry,
    LineupSnapshot,
    LineupSnapshotPlayer,
    Match,
    OddsSnapshot,
    OddsSnapshotBook,
    Player,
    Round,
    Season,
    SourceFetchLog,
    Team,
    TeamMatchStat,
    Venue,
    WeatherSnapshot,
    PlayerMatchStat,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


@dataclass(slots=True)
class OddsBookInput:
    bookmaker_key: str
    market_key: str
    home_price: float
    away_price: float


class SourceFetchLogger:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(
        self,
        *,
        source_name: str,
        entity_type: str,
        entity_key: str | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> SourceFetchLog:
        log = SourceFetchLog(
            source_name=source_name,
            entity_type=entity_type,
            entity_key=entity_key,
            requested_at=_utcnow(),
            status="started",
            request_meta=request_meta or {},
            response_meta={},
        )
        self.session.add(log)
        self.session.flush()
        return log

    def finish(
        self,
        log: SourceFetchLog,
        *,
        status: str,
        response_meta: dict[str, Any] | None = None,
        raw_payload: Any | None = None,
        error_message: str | None = None,
    ) -> SourceFetchLog:
        log.status = status
        log.completed_at = _utcnow()
        log.response_meta = response_meta or {}
        log.raw_payload = raw_payload
        log.error_message = error_message
        self.session.flush()
        return log


class FixtureIngestionService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create_competition(self, code: str, name: str) -> Competition:
        competition = self.session.scalar(
            select(Competition).where(Competition.code == code)
        )
        if competition is None:
            competition = Competition(code=code, name=name)
            self.session.add(competition)
            self.session.flush()
        return competition

    def get_or_create_season(self, competition_id, season_year: int) -> Season:
        season = self.session.scalar(
            select(Season).where(
                Season.competition_id == competition_id,
                Season.season_year == season_year,
            )
        )
        if season is None:
            season = Season(competition_id=competition_id, season_year=season_year)
            self.session.add(season)
            self.session.flush()
        return season

    def get_or_create_round(
        self,
        *,
        season_id,
        round_number: int,
        round_name: str,
        is_finals: bool = False,
        starts_at: datetime | None = None,
        ends_at: datetime | None = None,
    ) -> Round:
        round_obj = self.session.scalar(
            select(Round).where(
                Round.season_id == season_id,
                Round.round_number == round_number,
                Round.is_finals == is_finals,
            )
        )
        if round_obj is None:
            round_obj = Round(
                season_id=season_id,
                round_number=round_number,
                round_name=round_name,
                is_finals=is_finals,
                starts_at=starts_at,
                ends_at=ends_at,
            )
            self.session.add(round_obj)
            self.session.flush()
        return round_obj

    def upsert_team(
        self,
        *,
        team_code: str,
        name: str,
        short_name: str,
        state_code: str | None = None,
    ) -> Team:
        team = self.session.scalar(select(Team).where(Team.team_code == team_code))
        if team is None:
            team = Team(
                team_code=team_code,
                name=name,
                short_name=short_name,
                slug=_slugify(name),
                state_code=state_code,
            )
            self.session.add(team)
        else:
            team.name = name
            team.short_name = short_name
            team.state_code = state_code
        self.session.flush()
        return team

    def upsert_venue(
        self,
        *,
        name: str,
        venue_code: str | None = None,
        city: str | None = None,
        state_code: str | None = None,
        timezone_name: str = "Australia/Melbourne",
        bom_location_code: str | None = None,
        bom_station_id: str | None = None,
    ) -> Venue:
        statement = select(Venue)
        if venue_code is not None:
            statement = statement.where(Venue.venue_code == venue_code)
        else:
            statement = statement.where(Venue.name == name)
        venue = self.session.scalar(statement)
        if venue is None:
            venue = Venue(
                venue_code=venue_code,
                name=name,
                city=city,
                state_code=state_code,
                timezone=timezone_name,
                bom_location_code=bom_location_code,
                bom_station_id=bom_station_id,
            )
            self.session.add(venue)
        else:
            venue.city = city
            venue.state_code = state_code
            venue.timezone = timezone_name
            venue.bom_location_code = bom_location_code
            venue.bom_station_id = bom_station_id
        self.session.flush()
        return venue

    def upsert_player(
        self,
        *,
        full_name: str,
        player_code: str | None = None,
        current_team_id=None,
        active: bool = True,
    ) -> Player:
        statement = select(Player)
        if player_code is not None:
            statement = statement.where(Player.player_code == player_code)
        else:
            statement = statement.where(Player.full_name == full_name)
        player = self.session.scalar(statement)
        if player is None:
            first_name, _, last_name = full_name.partition(" ")
            player = Player(
                player_code=player_code,
                full_name=full_name,
                first_name=first_name or None,
                last_name=last_name or None,
                current_team_id=current_team_id,
                active=active,
            )
            self.session.add(player)
        else:
            player.current_team_id = current_team_id
            player.active = active
        self.session.flush()
        return player

    def upsert_match(
        self,
        *,
        season_id,
        round_id,
        home_team_id,
        away_team_id,
        venue_id,
        scheduled_at: datetime,
        status: str,
        match_code: str | None = None,
        home_score: int | None = None,
        away_score: int | None = None,
    ) -> Match:
        statement = select(Match)
        if match_code is not None:
            statement = statement.where(Match.match_code == match_code)
        else:
            statement = statement.where(
                Match.round_id == round_id,
                Match.home_team_id == home_team_id,
                Match.away_team_id == away_team_id,
                Match.scheduled_at == scheduled_at,
            )
        match = self.session.scalar(statement)
        winning_team_id = None
        actual_margin = None
        if home_score is not None and away_score is not None:
            actual_margin = home_score - away_score
            if home_score > away_score:
                winning_team_id = home_team_id
            elif away_score > home_score:
                winning_team_id = away_team_id
        if match is None:
            match = Match(
                season_id=season_id,
                round_id=round_id,
                match_code=match_code,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                venue_id=venue_id,
                scheduled_at=scheduled_at,
                status=status,
                home_score=home_score,
                away_score=away_score,
                winning_team_id=winning_team_id,
                actual_margin=actual_margin,
            )
            self.session.add(match)
        else:
            match.venue_id = venue_id
            match.scheduled_at = scheduled_at
            match.status = status
            match.home_score = home_score
            match.away_score = away_score
            match.winning_team_id = winning_team_id
            match.actual_margin = actual_margin
        self.session.flush()
        return match


class SnapshotIngestionService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def store_lineup_snapshot(
        self,
        *,
        match_id,
        team_id,
        source_name: str,
        fetched_at: datetime,
        payload: dict[str, Any],
        players: list[dict[str, Any]],
        round_run_id=None,
    ) -> LineupSnapshot:
        snapshot = LineupSnapshot(
            round_run_id=round_run_id,
            match_id=match_id,
            team_id=team_id,
            source_name=source_name,
            fetched_at=fetched_at,
            payload=payload,
        )
        self.session.add(snapshot)
        self.session.flush()
        for player_row in players:
            self.session.add(
                LineupSnapshotPlayer(
                    lineup_snapshot_id=snapshot.id,
                    player_id=player_row.get("player_id"),
                    source_player_name=player_row["source_player_name"],
                    slot_label=player_row.get("slot_label"),
                    named_role=player_row.get("named_role"),
                    is_selected=player_row.get("is_selected", True),
                    is_interchange=player_row.get("is_interchange", False),
                    is_emergency=player_row.get("is_emergency", False),
                    is_sub=player_row.get("is_sub", False),
                    mapping_status=player_row.get("mapping_status", "mapped"),
                )
            )
        self.session.flush()
        return snapshot

    def store_benchmark_prediction(
        self,
        *,
        round_run_id,
        match_id,
        source_name: str,
        predicted_winner_team_id=None,
        home_win_probability: float | None = None,
        away_win_probability: float | None = None,
        predicted_margin: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> BenchmarkPrediction:
        row = BenchmarkPrediction(
            round_run_id=round_run_id,
            match_id=match_id,
            source_name=source_name,
            predicted_winner_team_id=predicted_winner_team_id,
            home_win_probability=home_win_probability,
            away_win_probability=away_win_probability,
            predicted_margin=predicted_margin,
            payload=payload or {},
        )
        self.session.add(row)
        self.session.flush()
        return row

    def store_injury_snapshot(
        self,
        *,
        source_name: str,
        fetched_at: datetime,
        payload: dict[str, Any],
        entries: list[dict[str, Any]],
        round_run_id=None,
    ) -> InjurySnapshot:
        snapshot = InjurySnapshot(
            round_run_id=round_run_id,
            source_name=source_name,
            fetched_at=fetched_at,
            payload=payload,
        )
        self.session.add(snapshot)
        self.session.flush()
        for entry in entries:
            self.session.add(
                InjurySnapshotEntry(
                    injury_snapshot_id=snapshot.id,
                    team_id=entry.get("team_id"),
                    player_id=entry.get("player_id"),
                    source_player_name=entry["source_player_name"],
                    status_label=entry["status_label"],
                    injury_note=entry.get("injury_note"),
                    estimated_return_text=entry.get("estimated_return_text"),
                    uncertainty_flag=entry.get("uncertainty_flag", False),
                    mapping_status=entry.get("mapping_status", "mapped"),
                )
            )
        self.session.flush()
        return snapshot

    def store_weather_snapshot(
        self,
        *,
        match_id,
        venue_id,
        source_name: str,
        fetched_at: datetime,
        payload: dict[str, Any],
        round_run_id=None,
        temperature_c: float | None = None,
        rain_probability_pct: float | None = None,
        rainfall_mm: float | None = None,
        wind_kmh: float | None = None,
        weather_text: str | None = None,
        severe_flag: bool = False,
    ) -> WeatherSnapshot:
        snapshot = WeatherSnapshot(
            round_run_id=round_run_id,
            match_id=match_id,
            venue_id=venue_id,
            source_name=source_name,
            fetched_at=fetched_at,
            temperature_c=temperature_c,
            rain_probability_pct=rain_probability_pct,
            rainfall_mm=rainfall_mm,
            wind_kmh=wind_kmh,
            weather_text=weather_text,
            severe_flag=severe_flag,
            payload=payload,
        )
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def store_odds_snapshot(
        self,
        *,
        match_id,
        source_name: str,
        fetched_at: datetime,
        payload: dict[str, Any],
        books: list[OddsBookInput],
        round_run_id=None,
    ) -> OddsSnapshot:
        home_probs = []
        away_probs = []
        for book in books:
            raw_home = 1.0 / book.home_price
            raw_away = 1.0 / book.away_price
            total = raw_home + raw_away
            home_probs.append(raw_home / total)
            away_probs.append(raw_away / total)
        snapshot = OddsSnapshot(
            round_run_id=round_run_id,
            match_id=match_id,
            source_name=source_name,
            fetched_at=fetched_at,
            home_median_price=median([book.home_price for book in books]) if books else None,
            away_median_price=median([book.away_price for book in books]) if books else None,
            home_implied_probability=median(home_probs) if home_probs else None,
            away_implied_probability=median(away_probs) if away_probs else None,
            bookmaker_count=len(books),
            payload=payload,
        )
        self.session.add(snapshot)
        self.session.flush()
        for book in books:
            raw_home = 1.0 / book.home_price
            raw_away = 1.0 / book.away_price
            overround = (raw_home + raw_away) * 100.0
            self.session.add(
                OddsSnapshotBook(
                    odds_snapshot_id=snapshot.id,
                    bookmaker_key=book.bookmaker_key,
                    market_key=book.market_key,
                    home_price=book.home_price,
                    away_price=book.away_price,
                    overround_pct=overround,
                )
            )
        self.session.flush()
        return snapshot

    def create_audit_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        round_run_id=None,
        match_id=None,
    ) -> AuditEvent:
        event = AuditEvent(
            round_run_id=round_run_id,
            match_id=match_id,
            event_type=event_type,
            payload=payload,
        )
        self.session.add(event)
        self.session.flush()
        return event


class StatsIngestionService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_team_match_stat(
        self,
        *,
        match_id,
        team_id,
        source_name: str,
        stats: dict[str, Any],
    ) -> TeamMatchStat:
        row = self.session.scalar(
            select(TeamMatchStat).where(
                TeamMatchStat.match_id == match_id,
                TeamMatchStat.team_id == team_id,
                TeamMatchStat.source_name == source_name,
            )
        )
        if row is None:
            row = TeamMatchStat(
                match_id=match_id,
                team_id=team_id,
                source_name=source_name,
                stats=stats,
            )
            self.session.add(row)
        else:
            row.stats = stats
        self.session.flush()
        return row

    def upsert_player_match_stat(
        self,
        *,
        match_id,
        team_id,
        player_id,
        source_name: str,
        stats: dict[str, Any],
    ) -> PlayerMatchStat:
        row = self.session.scalar(
            select(PlayerMatchStat).where(
                PlayerMatchStat.match_id == match_id,
                PlayerMatchStat.player_id == player_id,
                PlayerMatchStat.source_name == source_name,
            )
        )
        if row is None:
            row = PlayerMatchStat(
                match_id=match_id,
                team_id=team_id,
                player_id=player_id,
                source_name=source_name,
                stats=stats,
            )
            self.session.add(row)
        else:
            row.team_id = team_id
            row.stats = stats
        self.session.flush()
        return row
