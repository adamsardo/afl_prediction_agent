from __future__ import annotations

from datetime import datetime, timezone

from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.sources.common.http import HttpFetchClient
from afl_prediction_agent.sources.common.models import (
    FetchEnvelope,
    NormalizedBenchmarkPrediction,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SquiggleConnector:
    source_name = "squiggle"
    base_url = "https://api.squiggle.com.au/"

    def __init__(self, http_client: HttpFetchClient | None = None, *, benchmark_source: str | None = None) -> None:
        settings = get_settings()
        self.http_client = http_client or HttpFetchClient()
        self.benchmark_source = benchmark_source or settings.squiggle_benchmark_source

    def fetch_predictions(
        self,
        *,
        season_year: int,
        round_number: int,
    ) -> tuple[FetchEnvelope, list[NormalizedBenchmarkPrediction]]:
        sources_response = self.http_client.get(
            self.base_url,
            params={"q": "sources"},
            expect_json=True,
        )
        tips_response = self.http_client.get(
            self.base_url,
            params={"q": "tips", "year": season_year, "round": round_number},
            expect_json=True,
        )
        games_response = self.http_client.get(
            self.base_url,
            params={"q": "games", "year": season_year, "round": round_number},
            expect_json=True,
        )
        source_rows = sources_response.json_data.get("sources", []) if isinstance(sources_response.json_data, dict) else []
        preferred_source = self._preferred_source(source_rows)
        games_by_id = {
            game["id"]: game
            for game in (games_response.json_data.get("games", []) if isinstance(games_response.json_data, dict) else [])
        }
        predictions: list[NormalizedBenchmarkPrediction] = []
        for tip in tips_response.json_data.get("tips", []):
            if preferred_source and tip.get("sourceid") != preferred_source["id"]:
                continue
            game = games_by_id.get(tip.get("gameid"), {})
            home_prob = tip.get("prob")
            margin = tip.get("margin")
            predictions.append(
                NormalizedBenchmarkPrediction(
                    home_team_name=game.get("hteam") or tip.get("hteam"),
                    away_team_name=game.get("ateam") or tip.get("ateam"),
                    season_year=season_year,
                    round_number=round_number,
                    source_name=preferred_source["name"] if preferred_source else str(tip.get("sourceid")),
                    predicted_winner_name=(game.get("hteam") if margin and margin > 0 else game.get("ateam"))
                    if margin is not None
                    else tip.get("tip"),
                    home_win_probability=float(home_prob) if home_prob is not None else None,
                    away_win_probability=(1 - float(home_prob)) if home_prob is not None else None,
                    predicted_margin=float(margin) if margin is not None else None,
                    match_code=str(game.get("id")) if game.get("id") is not None else None,
                )
            )
        envelope = FetchEnvelope(
            source_name=self.source_name,
            fetched_at=_utcnow(),
            request_meta={"season_year": season_year, "round_number": round_number},
            response_meta={
                "preferred_source": preferred_source["name"] if preferred_source else None,
                "prediction_count": len(predictions),
            },
            raw_payload={
                "sources": source_rows,
                "tips": tips_response.json_data.get("tips", []),
                "games": games_response.json_data.get("games", []),
            },
        )
        return envelope, predictions

    def _preferred_source(self, sources: list[dict]) -> dict | None:
        if not sources:
            return None
        requested = (self.benchmark_source or "").lower()
        if requested:
            for source in sources:
                if str(source.get("name", "")).lower() == requested:
                    return source
        for source in sources:
            if str(source.get("name", "")).lower() == "aggregate":
                return source
        return sources[0]
