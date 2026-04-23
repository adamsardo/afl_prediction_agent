from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from afl_prediction_agent.storage.models import (
    BenchmarkPrediction,
    InjurySnapshot,
    Match,
    OddsSnapshot,
    Round,
    Season,
    Team,
    Venue,
    WeatherSnapshot,
    LineupSnapshot,
)


@dataclass(slots=True)
class LoadedMatchContext:
    match: Match
    season: Season
    round: Round
    home_team: Team
    away_team: Team
    venue: Venue
    home_lineup: LineupSnapshot | None
    away_lineup: LineupSnapshot | None
    injury_snapshot: InjurySnapshot | None
    weather_snapshot: WeatherSnapshot | None
    odds_snapshot: OddsSnapshot | None
    benchmark_prediction: BenchmarkPrediction | None
def _load_latest_by_run_or_time(
    session: Session,
    model,
    *filters,
    round_run_id,
    lock_timestamp: datetime,
):
    if round_run_id is not None:
        run_specific = session.scalar(
            select(model)
            .where(model.round_run_id == round_run_id, *filters)
            .order_by(desc(model.fetched_at))
        )
        if run_specific is not None:
            return run_specific
    return session.scalar(
        select(model)
        .where(*filters, model.fetched_at <= lock_timestamp)
        .order_by(desc(model.fetched_at))
    )


def load_match_context(
    session: Session,
    *,
    match: Match,
    lock_timestamp: datetime,
    round_run_id,
) -> LoadedMatchContext:
    season = session.get(Season, match.season_id)
    round_obj = session.get(Round, match.round_id)
    home_team = session.get(Team, match.home_team_id)
    away_team = session.get(Team, match.away_team_id)
    venue = session.get(Venue, match.venue_id)
    assert season is not None
    assert round_obj is not None
    assert home_team is not None
    assert away_team is not None
    assert venue is not None

    home_lineup = _load_latest_by_run_or_time(
        session,
        LineupSnapshot,
        LineupSnapshot.match_id == match.id,
        LineupSnapshot.team_id == match.home_team_id,
        round_run_id=round_run_id,
        lock_timestamp=lock_timestamp,
    )
    away_lineup = _load_latest_by_run_or_time(
        session,
        LineupSnapshot,
        LineupSnapshot.match_id == match.id,
        LineupSnapshot.team_id == match.away_team_id,
        round_run_id=round_run_id,
        lock_timestamp=lock_timestamp,
    )
    injury_snapshot = _load_latest_by_run_or_time(
        session,
        InjurySnapshot,
        round_run_id=round_run_id,
        lock_timestamp=lock_timestamp,
    )
    weather_snapshot = _load_latest_by_run_or_time(
        session,
        WeatherSnapshot,
        WeatherSnapshot.match_id == match.id,
        round_run_id=round_run_id,
        lock_timestamp=lock_timestamp,
    )
    odds_snapshot = _load_latest_by_run_or_time(
        session,
        OddsSnapshot,
        OddsSnapshot.match_id == match.id,
        round_run_id=round_run_id,
        lock_timestamp=lock_timestamp,
    )

    benchmark_prediction = None
    if round_run_id is not None:
        benchmark_prediction = session.scalar(
            select(BenchmarkPrediction).where(
                BenchmarkPrediction.round_run_id == round_run_id,
                BenchmarkPrediction.match_id == match.id,
            )
        )

    return LoadedMatchContext(
        match=match,
        season=season,
        round=round_obj,
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        injury_snapshot=injury_snapshot,
        weather_snapshot=weather_snapshot,
        odds_snapshot=odds_snapshot,
        benchmark_prediction=benchmark_prediction,
    )
