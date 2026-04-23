from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.sources.common.http import HttpFetchClient
from afl_prediction_agent.sources.common.models import (
    FetchEnvelope,
    NormalizedOddsBook,
    NormalizedOddsSnapshot,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TheOddsApiConnector:
    source_name = "the_odds_api"
    base_url = "https://api.the-odds-api.com/v4/sports/aussierules_afl"

    def __init__(
        self,
        http_client: HttpFetchClient | None = None,
        *,
        api_key: str | None = None,
        bookmaker_path: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.http_client = http_client or HttpFetchClient()
        self.api_key = api_key or settings.odds_api_key
        self.bookmaker_path = bookmaker_path or settings.workspace_root / "data" / "mappings" / "odds_bookmakers_au.json"
        self.allowed_bookmakers = (
            [item.strip() for item in settings.odds_au_bookmakers.split(",") if item.strip()]
            if settings.odds_au_bookmakers
            else json.loads(self.bookmaker_path.read_text(encoding="utf-8"))
        )

    def fetch_head_to_head(
        self,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FetchEnvelope, list[NormalizedOddsSnapshot]]:
        if not self.api_key:
            raise ValueError("AFL_AGENT_ODDS_API_KEY is required for The Odds API")
        params: dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": "au",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": ",".join(self.allowed_bookmakers),
        }
        path = "odds"
        if as_of is not None:
            path = "odds-history"
            params["date"] = as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        response = self.http_client.get(
            f"{self.base_url}/{path}",
            params=params,
            expect_json=True,
        )
        payload = response.json_data
        events = payload.get("data", payload) if isinstance(payload, dict) else payload
        snapshots = [self._event_to_snapshot(event) for event in events]
        envelope = FetchEnvelope(
            source_name=self.source_name,
            fetched_at=_utcnow(),
            request_meta={"as_of": as_of.isoformat() if as_of else None},
            response_meta={
                "status_code": response.status_code,
                "event_count": len(snapshots),
                "remaining_requests": response.headers.get("x-requests-remaining"),
                "used_requests": response.headers.get("x-requests-used"),
            },
            raw_payload={"events": events},
        )
        return envelope, snapshots

    def _event_to_snapshot(self, event: dict[str, Any]) -> NormalizedOddsSnapshot:
        commence_time = event.get("commence_time")
        commence_at = (
            datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            if isinstance(commence_time, str)
            else None
        )
        books: list[NormalizedOddsBook] = []
        for bookmaker in event.get("bookmakers", []):
            if bookmaker.get("key") not in self.allowed_bookmakers:
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {outcome["name"]: outcome["price"] for outcome in market.get("outcomes", [])}
                home_name = event.get("home_team")
                away_name = event.get("away_team")
                if home_name not in prices or away_name not in prices:
                    continue
                books.append(
                    NormalizedOddsBook(
                        bookmaker_key=bookmaker["key"],
                        market_key=market["key"],
                        home_price=float(prices[home_name]),
                        away_price=float(prices[away_name]),
                    )
                )
        return NormalizedOddsSnapshot(
            home_team_name=event["home_team"],
            away_team_name=event["away_team"],
            commence_time=commence_at,
            books=books,
        )
