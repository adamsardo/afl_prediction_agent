from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from afl_prediction_agent.ingestion.services import (
    FixtureIngestionService,
    SnapshotIngestionService,
    SourceFetchLogger,
    StatsIngestionService,
    OddsBookInput,
)
from afl_prediction_agent.sources.afl.connector import FitzRoyConnector
from afl_prediction_agent.sources.afl_com.injuries import AFLComInjuryConnector
from afl_prediction_agent.sources.afl_tables.connector import AflTablesConnector
from afl_prediction_agent.sources.bom.connector import BomWeatherConnector
from afl_prediction_agent.sources.common.mapping import CanonicalMappingService
from afl_prediction_agent.sources.footywire.injuries import FootyWireInjuryConnector
from afl_prediction_agent.sources.odds.the_odds_api import TheOddsApiConnector
from afl_prediction_agent.sources.squiggle.api import SquiggleConnector
from afl_prediction_agent.storage.models import (
    InjurySnapshot,
    InjurySnapshotEntry,
    LineupSnapshotPlayer,
    Match,
    Round,
    Season,
    Team,
    Venue,
)


@dataclass(slots=True)
class IngestSummary:
    source_name: str
    created: int
    skipped: int = 0
    errors: list[str] | None = None


class RoundSourceSyncService:
    def __init__(
        self,
        session: Session,
        *,
        official_connector: FitzRoyConnector | None = None,
        archive_connector: AflTablesConnector | None = None,
        afl_com_connector: AFLComInjuryConnector | None = None,
        footywire_connector: FootyWireInjuryConnector | None = None,
        bom_connector: BomWeatherConnector | None = None,
        odds_connector: TheOddsApiConnector | None = None,
        squiggle_connector: SquiggleConnector | None = None,
    ) -> None:
        self.session = session
        self.fixture_service = FixtureIngestionService(session)
        self.snapshot_service = SnapshotIngestionService(session)
        self.stats_service = StatsIngestionService(session)
        self.fetch_logger = SourceFetchLogger(session)
        self.mapping = CanonicalMappingService(session)
        self.official_connector = official_connector or FitzRoyConnector()
        self.archive_connector = archive_connector or AflTablesConnector()
        self.afl_com_connector = afl_com_connector or AFLComInjuryConnector()
        self.footywire_connector = footywire_connector or FootyWireInjuryConnector()
        self.bom_connector = bom_connector or BomWeatherConnector()
        self.odds_connector = odds_connector or TheOddsApiConnector()
        self.squiggle_connector = squiggle_connector or SquiggleConnector()

    def ingest_fixtures(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
        use_archive_fallback: bool = False,
    ) -> IngestSummary:
        summary = self._ingest_matches_from_connector(
            connector=self.official_connector,
            season_year=season_year,
            round_number=round_number,
            fetch_kind="fixtures",
        )
        if summary.created == 0 and use_archive_fallback:
            return self._ingest_matches_from_connector(
                connector=self.archive_connector,
                season_year=season_year,
                round_number=round_number,
                fetch_kind="fixtures",
            )
        return summary

    def ingest_results(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
        use_archive_fallback: bool = False,
    ) -> IngestSummary:
        summary = self._ingest_matches_from_connector(
            connector=self.official_connector,
            season_year=season_year,
            round_number=round_number,
            fetch_kind="results",
        )
        if summary.created == 0 and use_archive_fallback:
            return self._ingest_matches_from_connector(
                connector=self.archive_connector,
                season_year=season_year,
                round_number=round_number,
                fetch_kind="results",
            )
        return summary

    def ingest_lineups(self, *, round_id, round_run_id=None) -> IngestSummary:
        round_obj, season = self._round_and_season(round_id)
        log = self.fetch_logger.start(
            source_name=self.official_connector.source_name,
            entity_type="lineups",
            entity_key=str(round_obj.id),
            request_meta={"season_year": season.season_year, "round_number": round_obj.round_number},
        )
        try:
            envelope, snapshots = self.official_connector.fetch_lineups(
                season_year=season.season_year,
                round_number=round_obj.round_number,
            )
            created = 0
            skipped = 0
            for snapshot in snapshots:
                match = self._match_for_round(
                    round_obj=round_obj,
                    match_code=snapshot.match_code,
                    home_team_name=snapshot.team.name if snapshot.home_or_away == "home" else None,
                    away_team_name=snapshot.team.name if snapshot.home_or_away == "away" else None,
                )
                if match is None:
                    skipped += 1
                    continue
                team_id = match.home_team_id if snapshot.home_or_away == "home" else match.away_team_id
                players = []
                for player in snapshot.players:
                    resolved = self.mapping.resolve_player(
                        source_name=self.official_connector.source_name,
                        external_id=player.source_player_id,
                        full_name=player.source_player_name,
                        current_team_id=team_id,
                        create_missing=True,
                    )
                    players.append(
                        {
                            "player_id": resolved.canonical_id,
                            "source_player_name": player.source_player_name,
                            "slot_label": player.slot_label,
                            "named_role": player.named_role,
                            "is_selected": player.is_selected,
                            "is_interchange": player.is_interchange,
                            "is_emergency": player.is_emergency,
                            "is_sub": player.is_sub,
                            "mapping_status": resolved.mapping_status,
                        }
                    )
                self.snapshot_service.store_lineup_snapshot(
                    match_id=match.id,
                    team_id=team_id,
                    source_name=self.official_connector.source_name,
                    fetched_at=envelope.fetched_at,
                    payload={"team": snapshot.team.name, "match_code": snapshot.match_code},
                    players=players,
                    round_run_id=round_run_id,
                )
                created += 1
            self.fetch_logger.finish(
                log,
                status="succeeded",
                response_meta={"snapshot_count": created, "skipped": skipped},
                raw_payload=envelope.raw_payload,
            )
            return IngestSummary(self.official_connector.source_name, created=created, skipped=skipped)
        except Exception as exc:
            self.fetch_logger.finish(log, status="failed", error_message=str(exc))
            raise

    def ingest_stats(
        self,
        *,
        season_year: int,
        round_number: int | None = None,
        source_track: str = "official",
    ) -> IngestSummary:
        connector = self.official_connector if source_track == "official" else self.archive_connector
        team_log = self.fetch_logger.start(
            source_name=connector.source_name,
            entity_type="team_stats",
            entity_key=f"{season_year}:{round_number or 'all'}",
            request_meta={"season_year": season_year, "round_number": round_number},
        )
        try:
            team_envelope, team_rows = connector.fetch_team_stats(
                season_year=season_year,
                round_number=round_number,
            )
            player_envelope, player_rows = connector.fetch_player_stats(
                season_year=season_year,
                round_number=round_number,
            )
            created = 0
            skipped = 0
            for row in team_rows:
                match = self._match_by_code_or_team_names(
                    match_code=row.match_code,
                    home_team_name=row.home_team_name,
                    away_team_name=row.away_team_name,
                )
                if match is None:
                    skipped += 1
                    continue
                team_result = self.mapping.resolve_team(
                    source_name=connector.source_name,
                    name=row.team_name,
                    create_missing=False,
                )
                if team_result.canonical_id is None:
                    skipped += 1
                    continue
                self.stats_service.upsert_team_match_stat(
                    match_id=match.id,
                    team_id=team_result.canonical_id,
                    source_name=connector.source_name,
                    stats=row.stats,
                )
                created += 1
            for row in player_rows:
                match = self._match_by_code_or_team_names(
                    match_code=row.match_code,
                    home_team_name=row.team_name,
                    away_team_name=row.team_name,
                )
                if match is None:
                    skipped += 1
                    continue
                team_result = self.mapping.resolve_team(
                    source_name=connector.source_name,
                    name=row.team_name,
                    create_missing=False,
                )
                if team_result.canonical_id is None:
                    skipped += 1
                    continue
                player_result = self.mapping.resolve_player(
                    source_name=connector.source_name,
                    external_id=row.source_player_id,
                    full_name=row.player_name,
                    current_team_id=team_result.canonical_id,
                    create_missing=source_track == "official",
                )
                if player_result.canonical_id is None:
                    skipped += 1
                    continue
                self.stats_service.upsert_player_match_stat(
                    match_id=match.id,
                    team_id=team_result.canonical_id,
                    player_id=player_result.canonical_id,
                    source_name=connector.source_name,
                    stats=row.stats,
                )
                created += 1
            self.fetch_logger.finish(
                team_log,
                status="succeeded",
                response_meta={
                    "team_rows": len(team_rows),
                    "player_rows": len(player_rows),
                    "created": created,
                    "skipped": skipped,
                },
                raw_payload={
                    "team_stats": team_envelope.raw_payload,
                    "player_stats": player_envelope.raw_payload,
                },
            )
            return IngestSummary(connector.source_name, created=created, skipped=skipped)
        except Exception as exc:
            self.fetch_logger.finish(team_log, status="failed", error_message=str(exc))
            raise

    def ingest_injuries(self, *, round_run_id=None) -> list[IngestSummary]:
        summaries = [
            self._ingest_injury_source(self.afl_com_connector, round_run_id=round_run_id),
            self._ingest_injury_source(self.footywire_connector, round_run_id=round_run_id),
        ]
        self._audit_injury_disagreements(round_run_id=round_run_id)
        return summaries

    def ingest_weather(self, *, round_id, round_run_id=None) -> IngestSummary:
        round_obj, _ = self._round_and_season(round_id)
        matches = self.session.scalars(
            select(Match).where(Match.round_id == round_obj.id).order_by(Match.scheduled_at.asc())
        ).all()
        created = 0
        skipped = 0
        errors: list[str] = []
        for match in matches:
            venue = self.session.get(Venue, match.venue_id)
            assert venue is not None
            log = self.fetch_logger.start(
                source_name=self.bom_connector.source_name,
                entity_type="weather",
                entity_key=str(match.id),
                request_meta={"venue_name": venue.name},
            )
            try:
                envelope, snapshot = self.bom_connector.fetch_weather_for_venue(
                    venue_name=venue.name,
                    scheduled_at=match.scheduled_at,
                )
                mapping = self.bom_connector.mapping_for_venue(venue.name)
                venue.bom_station_id = mapping.station_id
                venue.bom_location_code = mapping.forecast_location_name or mapping.forecast_district_name
                self.snapshot_service.store_weather_snapshot(
                    match_id=match.id,
                    venue_id=venue.id,
                    source_name=self.bom_connector.source_name,
                    fetched_at=envelope.fetched_at,
                    payload=envelope.raw_payload,
                    round_run_id=round_run_id,
                    temperature_c=snapshot.temperature_c,
                    rain_probability_pct=snapshot.rain_probability_pct,
                    rainfall_mm=snapshot.rainfall_mm,
                    wind_kmh=snapshot.wind_kmh,
                    weather_text=snapshot.weather_text,
                    severe_flag=snapshot.severe_flag,
                )
                if snapshot.rain_probability_pct is None:
                    self.snapshot_service.create_audit_event(
                        event_type="weather_forecast_coverage_gap",
                        payload={
                            "match_id": str(match.id),
                            "venue_name": venue.name,
                            "reason": "forecast_product_missing_or_unresolved",
                        },
                        round_run_id=round_run_id,
                        match_id=match.id,
                    )
                self.fetch_logger.finish(
                    log,
                    status="succeeded",
                    response_meta=envelope.response_meta,
                    raw_payload=envelope.raw_payload,
                )
                created += 1
            except Exception as exc:
                skipped += 1
                errors.append(str(exc))
                self.fetch_logger.finish(log, status="failed", error_message=str(exc))
                self.snapshot_service.create_audit_event(
                    event_type="weather_source_failed",
                    payload={"match_id": str(match.id), "error": str(exc)},
                    round_run_id=round_run_id,
                    match_id=match.id,
                )
        return IngestSummary(self.bom_connector.source_name, created=created, skipped=skipped, errors=errors or None)

    def ingest_odds(self, *, round_id, round_run_id=None, as_of: datetime | None = None) -> IngestSummary:
        round_obj, _ = self._round_and_season(round_id)
        log = self.fetch_logger.start(
            source_name=self.odds_connector.source_name,
            entity_type="odds",
            entity_key=str(round_obj.id),
            request_meta={"round_id": str(round_obj.id), "as_of": as_of.isoformat() if as_of else None},
        )
        try:
            envelope, snapshots = self.odds_connector.fetch_head_to_head(as_of=as_of)
            created = 0
            skipped = 0
            errors: list[str] = []
            for snapshot in snapshots:
                match = self._match_for_round(
                    round_obj=round_obj,
                    match_code=None,
                    home_team_name=snapshot.home_team_name,
                    away_team_name=snapshot.away_team_name,
                )
                if match is None:
                    skipped += 1
                    continue
                try:
                    self.snapshot_service.store_odds_snapshot(
                        match_id=match.id,
                        source_name=self.odds_connector.source_name,
                        fetched_at=envelope.fetched_at,
                        payload={
                            "home_team_name": snapshot.home_team_name,
                            "away_team_name": snapshot.away_team_name,
                            "commence_time": snapshot.commence_time.isoformat() if snapshot.commence_time else None,
                        },
                        books=[
                            OddsBookInput(
                                bookmaker_key=book.bookmaker_key,
                                market_key=book.market_key,
                                home_price=book.home_price,
                                away_price=book.away_price,
                            )
                            for book in snapshot.books
                        ],
                        round_run_id=round_run_id,
                    )
                    created += 1
                except Exception as exc:
                    skipped += 1
                    errors.append(str(exc))
            self.fetch_logger.finish(
                log,
                status="succeeded",
                response_meta={"created": created, "skipped": skipped, **envelope.response_meta},
                raw_payload=envelope.raw_payload,
            )
            return IngestSummary(self.odds_connector.source_name, created=created, skipped=skipped, errors=errors or None)
        except Exception as exc:
            self.fetch_logger.finish(log, status="failed", error_message=str(exc))
            self.snapshot_service.create_audit_event(
                event_type="odds_source_failed",
                payload={"round_id": str(round_obj.id), "error": str(exc)},
                round_run_id=round_run_id,
                match_id=None,
            )
            return IngestSummary(self.odds_connector.source_name, created=0, skipped=0, errors=[str(exc)])

    def ingest_benchmarks(self, *, round_id, round_run_id) -> IngestSummary:
        round_obj, season = self._round_and_season(round_id)
        log = self.fetch_logger.start(
            source_name=self.squiggle_connector.source_name,
            entity_type="benchmarks",
            entity_key=str(round_run_id),
            request_meta={"season_year": season.season_year, "round_number": round_obj.round_number},
        )
        try:
            envelope, predictions = self.squiggle_connector.fetch_predictions(
                season_year=season.season_year,
                round_number=round_obj.round_number,
            )
            created = 0
            skipped = 0
            for prediction in predictions:
                match = self._match_for_round(
                    round_obj=round_obj,
                    match_code=prediction.match_code,
                    home_team_name=prediction.home_team_name,
                    away_team_name=prediction.away_team_name,
                )
                if match is None:
                    skipped += 1
                    continue
                winner_team_id = None
                if prediction.predicted_winner_name:
                    winner_result = self.mapping.resolve_team(
                        source_name=self.squiggle_connector.source_name,
                        name=prediction.predicted_winner_name,
                        create_missing=False,
                    )
                    winner_team_id = winner_result.canonical_id
                self.snapshot_service.store_benchmark_prediction(
                    round_run_id=round_run_id,
                    match_id=match.id,
                    source_name=prediction.source_name,
                    predicted_winner_team_id=winner_team_id,
                    home_win_probability=prediction.home_win_probability,
                    away_win_probability=prediction.away_win_probability,
                    predicted_margin=prediction.predicted_margin,
                    payload={
                        "home_team_name": prediction.home_team_name,
                        "away_team_name": prediction.away_team_name,
                        "match_code": prediction.match_code,
                    },
                )
                created += 1
            self.fetch_logger.finish(
                log,
                status="succeeded",
                response_meta={"created": created, "skipped": skipped, **envelope.response_meta},
                raw_payload=envelope.raw_payload,
            )
            return IngestSummary(self.squiggle_connector.source_name, created=created, skipped=skipped)
        except Exception as exc:
            self.fetch_logger.finish(log, status="failed", error_message=str(exc))
            raise

    def snapshot_round(self, *, round_id, round_run_id=None) -> list[IngestSummary]:
        summaries: list[IngestSummary] = []
        lineups = self.ingest_lineups(round_id=round_id, round_run_id=round_run_id)
        summaries.append(lineups)
        if lineups.created == 0:
            raise ValueError("Official lineups could not be captured for this round")
        summaries.extend(self.ingest_injuries(round_run_id=round_run_id))
        summaries.append(self.ingest_weather(round_id=round_id, round_run_id=round_run_id))
        summaries.append(self.ingest_odds(round_id=round_id, round_run_id=round_run_id))
        if round_run_id is not None:
            try:
                summaries.append(self.ingest_benchmarks(round_id=round_id, round_run_id=round_run_id))
            except Exception as exc:
                self.snapshot_service.create_audit_event(
                    event_type="benchmark_source_failed",
                    payload={"round_id": str(round_id), "error": str(exc)},
                    round_run_id=round_run_id,
                    match_id=None,
                )
                summaries.append(
                    IngestSummary(
                        self.squiggle_connector.source_name,
                        created=0,
                        errors=[str(exc)],
                    )
                )
        return summaries

    def review_unresolved(self) -> dict[str, list[dict[str, Any]]]:
        lineup_rows = self.session.scalars(
            select(LineupSnapshotPlayer).where(LineupSnapshotPlayer.mapping_status != "mapped")
        ).all()
        injury_rows = self.session.scalars(
            select(InjurySnapshotEntry).where(InjurySnapshotEntry.mapping_status != "mapped")
        ).all()
        missing_bom = self.session.scalars(
            select(Venue).where(or_(Venue.bom_station_id.is_(None), Venue.bom_location_code.is_(None)))
        ).all()
        return {
            "lineup_players": [
                {"source_player_name": row.source_player_name, "mapping_status": row.mapping_status}
                for row in lineup_rows
            ],
            "injuries": [
                {
                    "source_player_name": row.source_player_name,
                    "status_label": row.status_label,
                    "mapping_status": row.mapping_status,
                }
                for row in injury_rows
            ],
            "venues": [
                {
                    "venue_name": row.name,
                    "bom_station_id": row.bom_station_id,
                    "bom_location_code": row.bom_location_code,
                }
                for row in missing_bom
            ],
        }

    def _ingest_matches_from_connector(
        self,
        *,
        connector,
        season_year: int,
        round_number: int | None,
        fetch_kind: str,
    ) -> IngestSummary:
        log = self.fetch_logger.start(
            source_name=connector.source_name,
            entity_type=fetch_kind,
            entity_key=f"{season_year}:{round_number or 'all'}",
            request_meta={"season_year": season_year, "round_number": round_number},
        )
        try:
            fetch_method = connector.fetch_fixtures if fetch_kind == "fixtures" else connector.fetch_results
            envelope, matches = fetch_method(season_year=season_year, round_number=round_number)
            competition = self.fixture_service.get_or_create_competition("AFL", "Australian Football League")
            season = self.fixture_service.get_or_create_season(competition.id, season_year)
            created = 0
            for fixture in matches:
                round_obj = self.fixture_service.get_or_create_round(
                    season_id=season.id,
                    round_number=fixture.round_number,
                    round_name=fixture.round_name,
                    is_finals=fixture.is_finals,
                )
                home_result = self.mapping.resolve_team(
                    source_name=connector.source_name,
                    external_id=fixture.home_team.external_id,
                    name=fixture.home_team.name,
                    short_name=fixture.home_team.short_name,
                    team_code=fixture.home_team.team_code,
                    state_code=fixture.home_team.state_code,
                    create_missing=True,
                )
                away_result = self.mapping.resolve_team(
                    source_name=connector.source_name,
                    external_id=fixture.away_team.external_id,
                    name=fixture.away_team.name,
                    short_name=fixture.away_team.short_name,
                    team_code=fixture.away_team.team_code,
                    state_code=fixture.away_team.state_code,
                    create_missing=True,
                )
                venue_result = self.mapping.resolve_venue(
                    source_name=connector.source_name,
                    external_id=fixture.venue.external_id,
                    name=fixture.venue.name,
                    city=fixture.venue.city,
                    state_code=fixture.venue.state_code,
                    timezone_name=fixture.venue.timezone_name,
                    create_missing=True,
                )
                if home_result.canonical_id is None or away_result.canonical_id is None or venue_result.canonical_id is None:
                    continue
                if connector.source_name == "afl_tables":
                    existing_match = self._match_by_code_or_team_names(
                        match_code=fixture.match_code,
                        home_team_name=fixture.home_team.name,
                        away_team_name=fixture.away_team.name,
                        round_id=round_obj.id,
                    )
                    if existing_match is not None:
                        continue
                self.fixture_service.upsert_match(
                    season_id=season.id,
                    round_id=round_obj.id,
                    home_team_id=home_result.canonical_id,
                    away_team_id=away_result.canonical_id,
                    venue_id=venue_result.canonical_id,
                    scheduled_at=fixture.scheduled_at,
                    status=fixture.status,
                    match_code=fixture.match_code,
                    home_score=fixture.home_score,
                    away_score=fixture.away_score,
                )
                created += 1
            self.fetch_logger.finish(
                log,
                status="succeeded",
                response_meta={"match_count": created, **envelope.response_meta},
                raw_payload=envelope.raw_payload,
            )
            return IngestSummary(connector.source_name, created=created)
        except Exception as exc:
            self.fetch_logger.finish(log, status="failed", error_message=str(exc))
            raise

    def _ingest_injury_source(self, connector, *, round_run_id=None) -> IngestSummary:
        log = self.fetch_logger.start(
            source_name=connector.source_name,
            entity_type="injuries",
            entity_key="current",
            request_meta={"source": connector.source_name},
        )
        try:
            envelope, snapshot = connector.fetch_injuries()
            entries = []
            for entry in snapshot.entries:
                team_result = self.mapping.resolve_team(
                    source_name=connector.source_name,
                    name=entry.team_name,
                    create_missing=False,
                )
                player_result = self.mapping.resolve_player(
                    source_name=connector.source_name,
                    external_id=entry.source_player_id,
                    full_name=entry.source_player_name,
                    current_team_id=team_result.canonical_id,
                    create_missing=False,
                )
                entries.append(
                    {
                        "team_id": team_result.canonical_id,
                        "player_id": player_result.canonical_id,
                        "source_player_name": entry.source_player_name,
                        "status_label": entry.status_label,
                        "injury_note": entry.injury_note,
                        "estimated_return_text": entry.estimated_return_text,
                        "uncertainty_flag": entry.uncertainty_flag,
                        "mapping_status": "mapped"
                        if team_result.canonical_id is not None and player_result.canonical_id is not None
                        else "unresolved",
                    }
                )
            self.snapshot_service.store_injury_snapshot(
                source_name=connector.source_name,
                fetched_at=envelope.fetched_at,
                payload=envelope.raw_payload,
                entries=entries,
                round_run_id=round_run_id,
            )
            self.fetch_logger.finish(
                log,
                status="succeeded",
                response_meta=envelope.response_meta,
                raw_payload=envelope.raw_payload,
            )
            return IngestSummary(connector.source_name, created=len(entries))
        except Exception as exc:
            self.fetch_logger.finish(log, status="failed", error_message=str(exc))
            self.snapshot_service.create_audit_event(
                event_type="injury_source_failed",
                payload={"source_name": connector.source_name, "error": str(exc)},
                round_run_id=round_run_id,
                match_id=None,
            )
            return IngestSummary(connector.source_name, created=0, errors=[str(exc)])

    def _audit_injury_disagreements(self, *, round_run_id=None) -> None:
        latest_primary = self.session.scalar(
            select(InjurySnapshot)
            .where(InjurySnapshot.source_name == self.afl_com_connector.source_name)
            .order_by(InjurySnapshot.fetched_at.desc())
        )
        latest_secondary = self.session.scalar(
            select(InjurySnapshot)
            .where(InjurySnapshot.source_name == self.footywire_connector.source_name)
            .order_by(InjurySnapshot.fetched_at.desc())
        )
        if latest_primary is None or latest_secondary is None:
            return
        primary_entries = self.session.scalars(
            select(InjurySnapshotEntry).where(InjurySnapshotEntry.injury_snapshot_id == latest_primary.id)
        ).all()
        secondary_entries = self.session.scalars(
            select(InjurySnapshotEntry).where(InjurySnapshotEntry.injury_snapshot_id == latest_secondary.id)
        ).all()
        secondary_by_key = {
            (row.team_id, row.player_id or row.source_player_name.lower()): row for row in secondary_entries
        }
        for row in primary_entries:
            key = (row.team_id, row.player_id or row.source_player_name.lower())
            other = secondary_by_key.get(key)
            if other is None:
                continue
            if (
                row.status_label != other.status_label
                or (row.estimated_return_text or "") != (other.estimated_return_text or "")
            ):
                self.snapshot_service.create_audit_event(
                    event_type="injury_source_disagreement",
                    payload={
                        "player_name": row.source_player_name,
                        "primary_status": row.status_label,
                        "secondary_status": other.status_label,
                        "primary_return": row.estimated_return_text,
                        "secondary_return": other.estimated_return_text,
                    },
                    round_run_id=round_run_id,
                    match_id=None,
                )

    def _round_and_season(self, round_id) -> tuple[Round, Season]:
        round_obj = self.session.get(Round, round_id)
        if round_obj is None:
            raise ValueError(f"Round {round_id} not found")
        season = self.session.get(Season, round_obj.season_id)
        assert season is not None
        return round_obj, season

    def _match_for_round(
        self,
        *,
        round_obj: Round,
        match_code: str | None,
        home_team_name: str | None,
        away_team_name: str | None,
    ) -> Match | None:
        if match_code:
            direct = self.session.scalar(
                select(Match).where(Match.round_id == round_obj.id, Match.match_code == match_code)
            )
            if direct is not None:
                return direct
        return self._match_by_code_or_team_names(
            match_code=match_code,
            home_team_name=home_team_name,
            away_team_name=away_team_name,
            round_id=round_obj.id,
        )

    def _match_by_code_or_team_names(
        self,
        *,
        match_code: str | None,
        home_team_name: str | None,
        away_team_name: str | None,
        round_id=None,
    ) -> Match | None:
        statement = select(Match)
        if round_id is not None:
            statement = statement.where(Match.round_id == round_id)
        if match_code:
            match = self.session.scalar(statement.where(Match.match_code == match_code))
            if match is not None:
                return match
        if not home_team_name or not away_team_name:
            names = [name for name in (home_team_name, away_team_name) if name]
            if len(names) != 1:
                return None
            result = self.mapping.resolve_team(
                source_name="lookup",
                name=names[0],
                create_missing=False,
            )
            if result.canonical_id is None:
                return None
            return self.session.scalar(
                statement.where(
                    or_(
                        Match.home_team_id == result.canonical_id,
                        Match.away_team_id == result.canonical_id,
                    )
                )
            )
        home_result = self.mapping.resolve_team(
            source_name="lookup",
            name=home_team_name,
            create_missing=False,
        )
        away_result = self.mapping.resolve_team(
            source_name="lookup",
            name=away_team_name,
            create_missing=False,
        )
        if home_result.canonical_id is None or away_result.canonical_id is None:
            return None
        if home_result.canonical_id == away_result.canonical_id:
            return self.session.scalar(
                statement.where(
                    or_(
                        Match.home_team_id == home_result.canonical_id,
                        Match.away_team_id == home_result.canonical_id,
                    )
                )
            )
        return self.session.scalar(
            statement.where(
                Match.home_team_id == home_result.canonical_id,
                Match.away_team_id == away_result.canonical_id,
            )
        )
