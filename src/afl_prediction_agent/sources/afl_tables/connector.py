from __future__ import annotations

from afl_prediction_agent.sources.afl.connector import FitzRoyBridge, FitzRoyConnector


class AflTablesConnector(FitzRoyConnector):
    source_name = "afl_tables"

    def __init__(self, bridge: FitzRoyBridge | None = None) -> None:
        super().__init__(bridge=bridge, source="afltables")
