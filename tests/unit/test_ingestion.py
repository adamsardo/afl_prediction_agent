from __future__ import annotations

from datetime import datetime, timezone
import uuid

from afl_prediction_agent.ingestion.services import OddsBookInput, SnapshotIngestionService


def test_odds_snapshot_normalises_overround_and_medians(session) -> None:
    service = SnapshotIngestionService(session)
    snapshot = service.store_odds_snapshot(
        match_id=uuid.uuid4(),
        source_name="odds_api",
        fetched_at=datetime.now(timezone.utc),
        payload={},
        books=[
            OddsBookInput("a", "h2h", 1.8, 2.0),
            OddsBookInput("b", "h2h", 1.75, 2.05),
            OddsBookInput("c", "h2h", 1.82, 1.98),
        ],
    )
    assert snapshot.bookmaker_count == 3
    assert float(snapshot.home_implied_probability) > 0.5
    assert round(float(snapshot.home_median_price), 2) == 1.8
