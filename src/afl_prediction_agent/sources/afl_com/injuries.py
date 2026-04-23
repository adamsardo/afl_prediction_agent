from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from afl_prediction_agent.sources.common.http import BROWSER_HEADERS, HttpFetchClient
from afl_prediction_agent.sources.common.models import (
    FetchEnvelope,
    NormalizedInjuryEntry,
    NormalizedInjurySnapshot,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


TEAM_ORDER = [
    "Adelaide Crows",
    "Brisbane Lions",
    "Carlton",
    "Collingwood",
    "Essendon",
    "Fremantle",
    "Geelong",
    "Gold Coast Suns",
    "GWS Giants",
    "Hawthorn",
    "Melbourne",
    "North Melbourne",
    "Port Adelaide",
    "Richmond",
    "St Kilda",
    "Sydney Swans",
    "West Coast Eagles",
    "Western Bulldogs",
]

TEAM_SLUGS = {
    "adelaide": "Adelaide Crows",
    "brisbane": "Brisbane Lions",
    "carlton": "Carlton",
    "collingwood": "Collingwood",
    "essendon": "Essendon",
    "fremantle": "Fremantle",
    "geelong": "Geelong",
    "gold-coast": "Gold Coast Suns",
    "goldcoast": "Gold Coast Suns",
    "gws": "GWS Giants",
    "giants": "GWS Giants",
    "hawthorn": "Hawthorn",
    "melbourne": "Melbourne",
    "north-melbourne": "North Melbourne",
    "northmelbourne": "North Melbourne",
    "port-adelaide": "Port Adelaide",
    "portadelaide": "Port Adelaide",
    "richmond": "Richmond",
    "st-kilda": "St Kilda",
    "stkilda": "St Kilda",
    "sydney": "Sydney Swans",
    "west-coast": "West Coast Eagles",
    "westcoast": "West Coast Eagles",
    "western-bulldogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
}


class AFLComInjuryConnector:
    source_name = "afl_com"
    url = "https://www.afl.com.au/news/injury-news"

    def __init__(self, http_client: HttpFetchClient | None = None) -> None:
        self.http_client = http_client or HttpFetchClient()

    def fetch_injuries(self) -> tuple[FetchEnvelope, NormalizedInjurySnapshot]:
        response = self.http_client.get(self.url, headers=BROWSER_HEADERS)
        snapshot = self.parse_html(response.text)
        envelope = FetchEnvelope(
            source_name=self.source_name,
            fetched_at=_utcnow(),
            request_meta={"url": self.url},
            response_meta={"status_code": response.status_code, "entry_count": len(snapshot.entries)},
            raw_payload={"html": response.text},
        )
        return envelope, snapshot

    def parse_html(self, html: str) -> NormalizedInjurySnapshot:
        soup = BeautifulSoup(html, "html.parser")
        anchor = soup.find(
            lambda tag: tag.name in {"h2", "h3"} and "FULL INJURY LIST" in tag.get_text(" ", strip=True)
        )
        if anchor is None:
            raise ValueError("Could not find AFL.com injury list anchor")
        published_at = self._extract_published_at(soup)
        tables = anchor.find_all_next("table", limit=18)
        entries: list[NormalizedInjuryEntry] = []
        for index, table in enumerate(tables):
            team_name = self._team_name_for_table(html, table, index)
            body_rows = table.select("tbody tr")
            if not body_rows:
                continue
            for row in body_rows:
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
                if len(cells) < 3:
                    continue
                player_name, injury_note, estimated_return = cells[:3]
                entries.append(
                    NormalizedInjuryEntry(
                        team_name=team_name,
                        source_player_name=player_name,
                        status_label=estimated_return,
                        injury_note=injury_note,
                        estimated_return_text=estimated_return,
                        uncertainty_flag=str(estimated_return).lower() in {"test", "tbc", "concussion protocols"},
                    )
                )
        return NormalizedInjurySnapshot(entries=entries, published_at=published_at)

    def _extract_published_at(self, soup: BeautifulSoup) -> datetime | None:
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            text = script.get_text(strip=True)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("headline") == "AFL Injury List":
                value = payload.get("dateModified") or payload.get("datePublished")
                if isinstance(value, str):
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return None

    def _team_name_for_table(self, html: str, table, index: int) -> str:
        table_html = str(table)
        table_index = html.find(table_html[:120])
        if table_index != -1:
            window = html[max(0, table_index - 4000) : table_index]
            matches = re.findall(r"/photo-resources/[^\"']+/([^/\"']+?)\\.jpg", window)
            for slug in reversed(matches):
                normalized = slug.lower().replace("_", "-")
                for fragment, team_name in TEAM_SLUGS.items():
                    if fragment in normalized:
                        return team_name
        return TEAM_ORDER[min(index, len(TEAM_ORDER) - 1)]

