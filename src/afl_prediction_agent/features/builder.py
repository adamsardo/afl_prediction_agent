from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any

import polars as pl
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from afl_prediction_agent.storage.context import LoadedMatchContext
from afl_prediction_agent.storage.models import (
    InjurySnapshotEntry,
    LineupSnapshot,
    LineupSnapshotPlayer,
    Match,
    PlayerMatchStat,
    TeamMatchStat,
)


@dataclass(slots=True)
class FeatureBuildResult:
    features: dict[str, Any]
    input_hash: str
    uncertainties: list[str]
    form_section: dict[str, Any]
    selection_section: dict[str, Any]
    venue_weather_section: dict[str, Any]
    market_section: dict[str, Any]
    benchmarks_section: dict[str, Any]


def _match_team_score(match: Match, team_id) -> tuple[int, int]:
    if match.home_score is None or match.away_score is None:
        return 0, 0
    if match.home_team_id == team_id:
        return match.home_score, match.away_score
    return match.away_score, match.home_score


def _extract_numeric_stat(stats: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_player_rating(stats: dict[str, Any]) -> float:
    primary = _extract_numeric_stat(
        stats,
        [
            "rating",
            "afl_rating",
            "player_rating",
            "coach_rating",
        ],
    )
    if primary is not None:
        return primary
    disposals = _extract_numeric_stat(stats, ["disposals", "kicks", "possessions"]) or 0.0
    score_involvements = _extract_numeric_stat(stats, ["score_involvements"]) or 0.0
    goals = _extract_numeric_stat(stats, ["goals"]) or 0.0
    tackles = _extract_numeric_stat(stats, ["tackles"]) or 0.0
    return disposals * 0.4 + score_involvements * 1.1 + goals * 2.5 + tackles * 0.6


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class FeatureBuilder:
    def __init__(self, feature_version: str) -> None:
        self.feature_version = feature_version

    def build_for_match(
        self,
        session: Session,
        context: LoadedMatchContext,
    ) -> FeatureBuildResult:
        home_form = self._team_form(
            session,
            team_id=context.home_team.id,
            before=context.match.scheduled_at,
        )
        away_form = self._team_form(
            session,
            team_id=context.away_team.id,
            before=context.match.scheduled_at,
        )
        team_stat_edges = self._team_stat_edges(
            session,
            home_team_id=context.home_team.id,
            away_team_id=context.away_team.id,
            before=context.match.scheduled_at,
        )
        selection_features = self._selection_features(session, context)
        venue_weather = self._venue_weather_features(context)
        market_features = self._market_features(context)

        features: dict[str, Any] = {
            "feature_version": self.feature_version,
            "home_recent_win_rate": home_form["win_rate"],
            "away_recent_win_rate": away_form["win_rate"],
            "home_recent_avg_margin": home_form["avg_margin"],
            "away_recent_avg_margin": away_form["avg_margin"],
            "home_recent_avg_points_for": home_form["avg_points_for"],
            "away_recent_avg_points_for": away_form["avg_points_for"],
            "home_recent_avg_points_against": home_form["avg_points_against"],
            "away_recent_avg_points_against": away_form["avg_points_against"],
            "home_rest_days": home_form["rest_days"],
            "away_rest_days": away_form["rest_days"],
            "home_is_interstate": venue_weather["travel_context"]["home_interstate"],
            "away_is_interstate": venue_weather["travel_context"]["away_interstate"],
            "home_ground_edge": venue_weather["home_ground_edge"],
            "home_named_changes": selection_features["home_named_changes"],
            "away_named_changes": selection_features["away_named_changes"],
            "home_lineup_strength": selection_features["home_lineup_strength"],
            "away_lineup_strength": selection_features["away_lineup_strength"],
            "home_injury_count": selection_features["home_injury_count"],
            "away_injury_count": selection_features["away_injury_count"],
            "market_home_implied_probability": market_features["home_implied_probability"],
            "market_away_implied_probability": market_features["away_implied_probability"],
            "bookmaker_count": market_features["bookmaker_count"],
            "weather_rain_probability_pct": venue_weather["forecast"]["rain_probability_pct"],
            "weather_rainfall_mm": venue_weather["forecast"]["rainfall_mm"],
            "weather_wind_kmh": venue_weather["forecast"]["wind_kmh"],
            "weather_temperature_c": venue_weather["forecast"]["temperature_c"],
            "squiggle_home_probability": market_features["benchmark_home_probability"],
            "squiggle_predicted_margin": market_features["benchmark_predicted_margin"],
            "team_stat_edges": team_stat_edges,
        }

        uncertainties = self._uncertainties(context, selection_features, market_features)
        input_hash = hashlib.sha256(
            json.dumps(
                {
                    "match_id": str(context.match.id),
                    "home_lineup": str(context.home_lineup.id) if context.home_lineup else None,
                    "away_lineup": str(context.away_lineup.id) if context.away_lineup else None,
                    "injury_snapshot": str(context.injury_snapshot.id)
                    if context.injury_snapshot
                    else None,
                    "weather_snapshot": str(context.weather_snapshot.id)
                    if context.weather_snapshot
                    else None,
                    "odds_snapshot": str(context.odds_snapshot.id) if context.odds_snapshot else None,
                    "benchmark_prediction": str(context.benchmark_prediction.id)
                    if context.benchmark_prediction
                    else None,
                    "features": features,
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        return FeatureBuildResult(
            features=features,
            input_hash=input_hash,
            uncertainties=uncertainties,
            form_section={
                "home_recent_form": home_form,
                "away_recent_form": away_form,
                "team_stat_edges": team_stat_edges,
            },
            selection_section={
                "home_named_changes": selection_features["home_named_changes"],
                "away_named_changes": selection_features["away_named_changes"],
                "home_lineup_strength": selection_features["home_lineup_strength"],
                "away_lineup_strength": selection_features["away_lineup_strength"],
                "key_absences": selection_features["key_absences"],
            },
            venue_weather_section=venue_weather,
            market_section={
                "home_implied_probability": market_features["home_implied_probability"],
                "away_implied_probability": market_features["away_implied_probability"],
                "bookmaker_count": market_features["bookmaker_count"],
            },
            benchmarks_section={
                "squiggle": {
                    "home_win_probability": market_features["benchmark_home_probability"],
                    "predicted_margin": market_features["benchmark_predicted_margin"],
                }
            },
        )

    def _team_form(self, session: Session, *, team_id, before: datetime) -> dict[str, float]:
        matches = session.scalars(
            select(Match)
            .where(
                Match.scheduled_at < before,
                Match.status == "completed",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            )
            .order_by(Match.scheduled_at.desc())
            .limit(5)
        ).all()
        if not matches:
            return {
                "win_rate": 0.5,
                "avg_margin": 0.0,
                "avg_points_for": 0.0,
                "avg_points_against": 0.0,
                "rest_days": 7.0,
            }

        rows: list[dict[str, float]] = []
        for match in matches:
            team_score, opp_score = _match_team_score(match, team_id)
            rows.append(
                {
                    "win": 1.0 if team_score > opp_score else 0.0,
                    "margin": float(team_score - opp_score),
                    "points_for": float(team_score),
                    "points_against": float(opp_score),
                }
            )
        df = pl.DataFrame(rows)
        latest_match = matches[0]
        rest_days = max(
            (_as_utc(before) - _as_utc(latest_match.scheduled_at)).total_seconds() / 86400.0,
            0.0,
        )
        return {
            "win_rate": float(df["win"].mean()),
            "avg_margin": float(df["margin"].mean()),
            "avg_points_for": float(df["points_for"].mean()),
            "avg_points_against": float(df["points_against"].mean()),
            "rest_days": rest_days,
        }

    def _selection_features(
        self,
        session: Session,
        context: LoadedMatchContext,
    ) -> dict[str, Any]:
        home_players = self._lineup_players(session, context.home_lineup)
        away_players = self._lineup_players(session, context.away_lineup)
        home_previous = self._previous_lineup_players(
            session,
            team_id=context.home_team.id,
            before=context.match.scheduled_at,
            exclude_snapshot_id=context.home_lineup.id if context.home_lineup else None,
        )
        away_previous = self._previous_lineup_players(
            session,
            team_id=context.away_team.id,
            before=context.match.scheduled_at,
            exclude_snapshot_id=context.away_lineup.id if context.away_lineup else None,
        )

        home_named_changes = self._count_named_changes(home_players, home_previous)
        away_named_changes = self._count_named_changes(away_players, away_previous)
        home_lineup_strength = self._lineup_strength(session, home_players)
        away_lineup_strength = self._lineup_strength(session, away_players)
        home_injuries = self._team_injuries(session, context, context.home_team.id)
        away_injuries = self._team_injuries(session, context, context.away_team.id)
        key_absences = [
            *home_injuries["key_absences"],
            *away_injuries["key_absences"],
        ]
        return {
            "home_named_changes": home_named_changes,
            "away_named_changes": away_named_changes,
            "home_lineup_strength": home_lineup_strength,
            "away_lineup_strength": away_lineup_strength,
            "home_injury_count": home_injuries["injury_count"],
            "away_injury_count": away_injuries["injury_count"],
            "key_absences": key_absences,
        }

    def _venue_weather_features(self, context: LoadedMatchContext) -> dict[str, Any]:
        forecast = {
            "temperature_c": float(context.weather_snapshot.temperature_c)
            if context.weather_snapshot and context.weather_snapshot.temperature_c is not None
            else None,
            "rain_probability_pct": float(context.weather_snapshot.rain_probability_pct)
            if context.weather_snapshot and context.weather_snapshot.rain_probability_pct is not None
            else None,
            "rainfall_mm": float(context.weather_snapshot.rainfall_mm)
            if context.weather_snapshot and context.weather_snapshot.rainfall_mm is not None
            else None,
            "wind_kmh": float(context.weather_snapshot.wind_kmh)
            if context.weather_snapshot and context.weather_snapshot.wind_kmh is not None
            else None,
            "weather_text": context.weather_snapshot.weather_text if context.weather_snapshot else None,
            "severe_flag": context.weather_snapshot.severe_flag if context.weather_snapshot else False,
        }
        home_interstate = (
            context.home_team.state_code is not None
            and context.venue.state_code is not None
            and context.home_team.state_code != context.venue.state_code
        )
        away_interstate = (
            context.away_team.state_code is not None
            and context.venue.state_code is not None
            and context.away_team.state_code != context.venue.state_code
        )
        home_ground_edge = (
            context.home_team.state_code is not None
            and context.venue.state_code is not None
            and context.home_team.state_code == context.venue.state_code
            and context.away_team.state_code != context.venue.state_code
        )
        return {
            "home_ground_edge": home_ground_edge,
            "travel_context": {
                "home_interstate": home_interstate,
                "away_interstate": away_interstate,
            },
            "forecast": forecast,
        }

    def _market_features(self, context: LoadedMatchContext) -> dict[str, Any]:
        home_prob = (
            float(context.odds_snapshot.home_implied_probability)
            if context.odds_snapshot and context.odds_snapshot.home_implied_probability is not None
            else None
        )
        away_prob = (
            float(context.odds_snapshot.away_implied_probability)
            if context.odds_snapshot and context.odds_snapshot.away_implied_probability is not None
            else None
        )
        benchmark_home_probability = (
            float(context.benchmark_prediction.home_win_probability)
            if context.benchmark_prediction and context.benchmark_prediction.home_win_probability is not None
            else None
        )
        benchmark_predicted_margin = (
            float(context.benchmark_prediction.predicted_margin)
            if context.benchmark_prediction and context.benchmark_prediction.predicted_margin is not None
            else None
        )
        return {
            "home_implied_probability": home_prob,
            "away_implied_probability": away_prob,
            "bookmaker_count": context.odds_snapshot.bookmaker_count if context.odds_snapshot else 0,
            "benchmark_home_probability": benchmark_home_probability,
            "benchmark_predicted_margin": benchmark_predicted_margin,
        }

    def _team_stat_edges(
        self,
        session: Session,
        *,
        home_team_id,
        away_team_id,
        before: datetime,
    ) -> list[dict[str, Any]]:
        candidate_keys = [
            "inside_50",
            "clearances",
            "contested_possessions",
            "marks_inside_50",
            "tackles_inside_50",
        ]
        stats_rows = session.scalars(
            select(TeamMatchStat)
            .join(Match, TeamMatchStat.match_id == Match.id)
            .where(
                Match.scheduled_at < before,
                TeamMatchStat.team_id.in_([home_team_id, away_team_id]),
            )
            .order_by(Match.scheduled_at.desc())
            .limit(20)
        ).all()

        by_team: dict[str, dict[str, list[float]]] = {
            str(home_team_id): {key: [] for key in candidate_keys},
            str(away_team_id): {key: [] for key in candidate_keys},
        }
        for row in stats_rows:
            team_bucket = by_team.get(str(row.team_id))
            if team_bucket is None:
                continue
            for key in candidate_keys:
                value = _extract_numeric_stat(row.stats, [key, key.replace("_", "")])
                if value is not None:
                    team_bucket[key].append(value)

        edges: list[dict[str, Any]] = []
        for key in candidate_keys:
            home_values = by_team[str(home_team_id)][key]
            away_values = by_team[str(away_team_id)][key]
            if not home_values or not away_values:
                continue
            delta = mean(home_values[:5]) - mean(away_values[:5])
            leans_to = "home" if delta > 0 else "away"
            strength = min(abs(delta) / 20.0, 1.0)
            edges.append(
                {
                    "label": key,
                    "leans_to": leans_to,
                    "strength": round(strength, 3),
                    "evidence": f"recent rolling delta {delta:.2f}",
                }
            )
        return edges

    def _lineup_players(
        self,
        session: Session,
        lineup_snapshot: LineupSnapshot | None,
    ) -> list[LineupSnapshotPlayer]:
        if lineup_snapshot is None:
            return []
        return session.scalars(
            select(LineupSnapshotPlayer).where(
                LineupSnapshotPlayer.lineup_snapshot_id == lineup_snapshot.id,
                LineupSnapshotPlayer.is_selected.is_(True),
            )
        ).all()

    def _previous_lineup_players(
        self,
        session: Session,
        *,
        team_id,
        before: datetime,
        exclude_snapshot_id,
    ) -> list[LineupSnapshotPlayer]:
        previous_snapshot = session.scalar(
            select(LineupSnapshot)
            .join(Match, LineupSnapshot.match_id == Match.id)
            .where(
                LineupSnapshot.team_id == team_id,
                Match.scheduled_at < before,
                LineupSnapshot.id != exclude_snapshot_id if exclude_snapshot_id else True,
            )
            .order_by(Match.scheduled_at.desc(), LineupSnapshot.fetched_at.desc())
        )
        return self._lineup_players(session, previous_snapshot)

    def _count_named_changes(
        self,
        current_players: list[LineupSnapshotPlayer],
        previous_players: list[LineupSnapshotPlayer],
    ) -> int:
        current_ids = {player.player_id or player.source_player_name for player in current_players}
        previous_ids = {player.player_id or player.source_player_name for player in previous_players}
        if not previous_ids:
            return 0
        return len(current_ids.symmetric_difference(previous_ids))

    def _lineup_strength(
        self,
        session: Session,
        players: list[LineupSnapshotPlayer],
    ) -> float:
        player_ids = [player.player_id for player in players if player.player_id is not None]
        if not player_ids:
            return float(len(players))
        stats = session.scalars(
            select(PlayerMatchStat)
            .where(PlayerMatchStat.player_id.in_(player_ids))
            .order_by(PlayerMatchStat.created_at.desc())
            .limit(max(len(player_ids) * 5, 1))
        ).all()
        ratings_by_player: dict[str, list[float]] = {}
        for stat in stats:
            ratings_by_player.setdefault(str(stat.player_id), []).append(
                _extract_player_rating(stat.stats)
            )
        ratings: list[float] = []
        for player_id in player_ids:
            samples = ratings_by_player.get(str(player_id))
            if not samples:
                ratings.append(1.0)
                continue
            ratings.append(mean(samples[:5]))
        return round(sum(ratings), 2)

    def _team_injuries(self, session: Session, context: LoadedMatchContext, team_id) -> dict[str, Any]:
        if context.injury_snapshot is None:
            return {"injury_count": 0, "key_absences": []}
        entries = session.scalars(
            select(InjurySnapshotEntry).where(
                InjurySnapshotEntry.injury_snapshot_id == context.injury_snapshot.id,
                InjurySnapshotEntry.team_id == team_id,
            )
        ).all()
        active_statuses = {"available", "fit", "named"}
        key_absences = []
        injury_count = 0
        for entry in entries:
            status = entry.status_label.strip().lower()
            if status in active_statuses:
                continue
            injury_count += 1
            key_absences.append(
                {
                    "player_name": entry.source_player_name,
                    "status_label": entry.status_label,
                    "injury_note": entry.injury_note,
                    "uncertainty_flag": entry.uncertainty_flag,
                    "team_id": str(team_id),
                }
            )
        return {"injury_count": injury_count, "key_absences": key_absences[:5]}

    def _uncertainties(
        self,
        context: LoadedMatchContext,
        selection_features: dict[str, Any],
        market_features: dict[str, Any],
    ) -> list[str]:
        items: list[str] = []
        if context.home_lineup is None or context.away_lineup is None:
            items.append("missing official lineup snapshot for one side")
        if context.injury_snapshot is None:
            items.append("missing injury snapshot at lock time")
        if context.weather_snapshot is None:
            items.append("missing weather snapshot at lock time")
        if market_features["bookmaker_count"] < 3:
            items.append("thin bookmaker sample at lock time")
        if any(absence.get("uncertainty_flag") for absence in selection_features["key_absences"]):
            items.append("uncertain player availability remains in injury list")
        return items
