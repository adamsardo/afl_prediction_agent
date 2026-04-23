from __future__ import annotations

import json
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from ftplib import FTP
from io import BytesIO
from pathlib import Path
from typing import Any

from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.sources.common.models import FetchEnvelope, NormalizedWeatherSnapshot


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


STATE_PREFIX = {
    "VIC": "IDV",
    "NSW": "IDN",
    "ACT": "IDN",
    "QLD": "IDQ",
    "SA": "IDS",
    "WA": "IDW",
    "TAS": "IDT",
    "NT": "IDD",
}


@dataclass(slots=True)
class BomVenueMapping:
    venue_name: str
    aliases: list[str]
    city: str
    state_code: str
    station_id: str | None = None
    forecast_product_id: str | None = None
    forecast_location_name: str | None = None
    forecast_district_name: str | None = None


class BomFtpClient:
    def __init__(self, host: str = "ftp.bom.gov.au", base_path: str = "/anon/gen/fwo") -> None:
        self.host = host
        self.base_path = base_path

    def read_binary(self, file_name: str) -> bytes:
        ftp = FTP(self.host, timeout=30)
        ftp.login()
        buffer = BytesIO()
        ftp.retrbinary(f"RETR {self.base_path}/{file_name}", buffer.write)
        ftp.quit()
        return buffer.getvalue()

    def read_text(self, file_name: str) -> str:
        ftp = FTP(self.host, timeout=30)
        ftp.login()
        lines: list[str] = []
        ftp.retrlines(f"RETR {self.base_path}/{file_name}", lines.append)
        ftp.quit()
        return "\n".join(lines)


class BomWeatherConnector:
    source_name = "bom"

    def __init__(
        self,
        ftp_client: BomFtpClient | None = None,
        *,
        mapping_path: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.ftp_client = ftp_client or BomFtpClient()
        self.mapping_path = mapping_path or settings.workspace_root / "data" / "mappings" / "venue_bom_mapping.json"
        self._mappings = self._load_mappings()

    def fetch_weather_for_venue(
        self,
        *,
        venue_name: str,
        scheduled_at: datetime,
    ) -> tuple[FetchEnvelope, NormalizedWeatherSnapshot]:
        mapping = self.mapping_for_venue(venue_name)
        observation = self._fetch_observation(mapping) if mapping.station_id else {}
        forecast = self._fetch_forecast(mapping, scheduled_at) if mapping.forecast_product_id else {}
        snapshot = NormalizedWeatherSnapshot(
            venue_name=mapping.venue_name,
            forecast_for=scheduled_at,
            temperature_c=self._float_or_none(observation.get("air_temp")),
            rain_probability_pct=self._extract_probability(forecast.get("probability_of_precipitation")),
            rainfall_mm=None,
            wind_kmh=self._float_or_none(observation.get("wind_spd_kmh")),
            weather_text=forecast.get("precis") or observation.get("weather"),
            severe_flag=self._infer_severe_flag(forecast.get("precis")),
        )
        envelope = FetchEnvelope(
            source_name=self.source_name,
            fetched_at=_utcnow(),
            request_meta={
                "venue_name": venue_name,
                "station_id": mapping.station_id,
                "forecast_product_id": mapping.forecast_product_id,
            },
            response_meta={
                "observation_available": bool(observation),
                "forecast_available": bool(forecast),
            },
            raw_payload={"observation": observation, "forecast": forecast, "mapping": asdict(mapping)},
        )
        return envelope, snapshot

    def mapping_for_venue(self, venue_name: str) -> BomVenueMapping:
        lowered = venue_name.strip().lower()
        for mapping in self._mappings:
            if lowered == mapping.venue_name.lower() or lowered in {alias.lower() for alias in mapping.aliases}:
                return mapping
        raise ValueError(f"No BOM mapping configured for venue {venue_name}")

    def _fetch_observation(self, mapping: BomVenueMapping) -> dict[str, Any]:
        prefix = STATE_PREFIX[mapping.state_code]
        archive_name = f"{prefix}60910.tgz"
        payload = self.ftp_client.read_binary(archive_name)
        with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
            target_suffix = f"{prefix}60910.{mapping.station_id}.json"
            member_name = next(name for name in archive.getnames() if name.endswith(target_suffix))
            raw = archive.extractfile(member_name)
            assert raw is not None
            observation_payload = json.loads(raw.read().decode("utf-8"))
        rows = observation_payload.get("observations", {}).get("data", [])
        return rows[0] if rows else {}

    def _fetch_forecast(self, mapping: BomVenueMapping, scheduled_at: datetime) -> dict[str, Any]:
        xml_text = self.ftp_client.read_text(f"{mapping.forecast_product_id}.xml")
        root = ET.fromstring(xml_text)
        area = self._select_forecast_area(root, mapping)
        if area is None:
            return {}
        target = scheduled_at if scheduled_at.tzinfo else scheduled_at.replace(tzinfo=timezone.utc)
        chosen: dict[str, Any] = {}
        for period in area.findall("forecast-period"):
            start = datetime.fromisoformat(period.attrib["start-time-local"])
            end = datetime.fromisoformat(period.attrib["end-time-local"])
            if start <= target.astimezone(start.tzinfo) < end:
                chosen = self._period_payload(period)
                break
        if not chosen:
            periods = area.findall("forecast-period")
            if periods:
                chosen = self._period_payload(periods[0])
        return chosen

    def _select_forecast_area(self, root: ET.Element, mapping: BomVenueMapping) -> ET.Element | None:
        locations = root.findall(".//forecast/area")
        if mapping.forecast_location_name:
            for area in locations:
                if (
                    area.attrib.get("type") == "location"
                    and area.attrib.get("description", "").lower() == mapping.forecast_location_name.lower()
                ):
                    return area
        if mapping.forecast_district_name:
            for area in locations:
                if (
                    area.attrib.get("type") in {"public-district", "district", "metropolitan"}
                    and area.attrib.get("description", "").lower() == mapping.forecast_district_name.lower()
                ):
                    return area
        return None

    def _period_payload(self, period: ET.Element) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for child in period:
            key = child.attrib.get("type")
            if not key:
                continue
            payload[key] = (child.text or "").strip()
        return payload

    def _load_mappings(self) -> list[BomVenueMapping]:
        rows = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        return [
            BomVenueMapping(
                venue_name=row["venue_name"],
                aliases=row.get("aliases", []),
                city=row["city"],
                state_code=row["state_code"],
                station_id=row.get("station_id"),
                forecast_product_id=row.get("forecast_product_id"),
                forecast_location_name=row.get("forecast_location_name"),
                forecast_district_name=row.get("forecast_district_name"),
            )
            for row in rows
        ]

    def _extract_probability(self, value: str | None) -> float | None:
        if value is None:
            return None
        digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
        return float(digits) if digits else None

    def _float_or_none(self, value) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    def _infer_severe_flag(self, precis: str | None) -> bool:
        if precis is None:
            return False
        lowered = precis.lower()
        return any(token in lowered for token in ("storm", "thunder", "gale", "hail", "severe"))
