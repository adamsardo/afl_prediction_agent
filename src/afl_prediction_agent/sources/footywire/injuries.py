from __future__ import annotations

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


class FootyWireInjuryConnector:
    source_name = "footywire"
    url = "https://www.footywire.com/afl/footy/injury_list"

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
        entries: list[NormalizedInjuryEntry] = []
        seen_teams: set[str] = set()
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue
            heading_text = rows[0].get_text(" ", strip=True)
            if "Players)" not in heading_text:
                continue
            team_name = heading_text.split("(")[0].strip()
            if team_name in seen_teams:
                continue
            seen_teams.add(team_name)
            for row in rows[2:]:
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
                if len(cells) < 3:
                    continue
                player_name, injury_note, returning = cells[:3]
                entries.append(
                    NormalizedInjuryEntry(
                        team_name=team_name,
                        source_player_name=player_name,
                        status_label=returning,
                        injury_note=injury_note,
                        estimated_return_text=returning,
                        uncertainty_flag=str(returning).lower() in {"test", "tbc", "indefinite"},
                    )
                )
        if not entries:
            raise ValueError("Could not parse FootyWire injury list")
        return NormalizedInjurySnapshot(entries=entries, published_at=None)
