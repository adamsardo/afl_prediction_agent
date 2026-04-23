from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.ingestion.services import FixtureIngestionService
from afl_prediction_agent.storage.models import ExternalIdMapping, Player, Team, Venue


@dataclass(slots=True)
class MappingResult:
    canonical_id: Any | None
    mapping_status: str
    matched_value: str | None = None


@lru_cache(maxsize=1)
def _alias_payload() -> dict[str, Any]:
    settings = get_settings()
    base = settings.workspace_root / "data" / "mappings"
    payload: dict[str, Any] = {}
    for name in ("team_aliases.json", "player_aliases.json", "venue_bom_mapping.json"):
        path = base / name
        if path.exists():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    return payload


class CanonicalMappingService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.fixture_service = FixtureIngestionService(session)

    def resolve_team(
        self,
        *,
        source_name: str,
        external_id: str | None = None,
        name: str,
        short_name: str | None = None,
        team_code: str | None = None,
        state_code: str | None = None,
        create_missing: bool = False,
    ) -> MappingResult:
        if external_id:
            existing = self._mapped_entity(Team, source_name, "team", external_id)
            if existing is not None:
                return MappingResult(existing.id, "mapped", existing.name)

        alias_map = {
            alias.lower(): canonical
            for canonical, aliases in _alias_payload().get("team_aliases.json", {}).items()
            for alias in aliases
        }
        canonical_name = alias_map.get(name.strip().lower(), name)
        statement = select(Team).where((Team.name == canonical_name) | (Team.short_name == canonical_name))
        if team_code:
            statement = statement.where((Team.name == canonical_name) | (Team.short_name == canonical_name) | (Team.team_code == team_code))
        team = self.session.scalar(statement)
        if team is None and create_missing:
            inferred_code = team_code or self._derive_team_code(canonical_name)
            team = self.fixture_service.upsert_team(
                team_code=inferred_code,
                name=canonical_name,
                short_name=short_name or canonical_name,
                state_code=state_code,
            )
        if team is None:
            return MappingResult(None, "unresolved")

        if external_id:
            self._upsert_mapping(
                source_name=source_name,
                entity_type="team",
                external_id=external_id,
                canonical_id=team.id,
                confidence_score=1.0,
                mapping_status="mapped",
                notes=f"Resolved by canonical team name: {canonical_name}",
            )
        return MappingResult(team.id, "mapped", team.name)

    def resolve_venue(
        self,
        *,
        source_name: str,
        external_id: str | None = None,
        name: str,
        city: str | None = None,
        state_code: str | None = None,
        timezone_name: str = "Australia/Melbourne",
        create_missing: bool = False,
    ) -> MappingResult:
        if external_id:
            existing = self._mapped_entity(Venue, source_name, "venue", external_id)
            if existing is not None:
                return MappingResult(existing.id, "mapped", existing.name)

        alias_rows = _alias_payload().get("venue_bom_mapping.json", [])
        alias_index = {}
        for row in alias_rows:
            for alias in row.get("aliases", []):
                alias_index[alias.lower()] = row
        alias_row = alias_index.get(name.strip().lower())
        canonical_name = alias_row["venue_name"] if alias_row else name
        statement = select(Venue).where(Venue.name == canonical_name)
        if external_id:
            statement = statement.where((Venue.name == canonical_name) | (Venue.venue_code == external_id))
        venue = self.session.scalar(statement)
        if venue is None and create_missing:
            venue = self.fixture_service.upsert_venue(
                venue_code=external_id,
                name=canonical_name,
                city=city or (alias_row.get("city") if alias_row else None),
                state_code=state_code or (alias_row.get("state_code") if alias_row else None),
                timezone_name=timezone_name,
                bom_location_code=alias_row.get("forecast_location_name") if alias_row else None,
                bom_station_id=alias_row.get("station_id") if alias_row else None,
            )
        if venue is None:
            return MappingResult(None, "unresolved")

        if external_id:
            self._upsert_mapping(
                source_name=source_name,
                entity_type="venue",
                external_id=external_id,
                canonical_id=venue.id,
                confidence_score=1.0,
                mapping_status="mapped",
                notes=f"Resolved by canonical venue name: {canonical_name}",
            )
        return MappingResult(venue.id, "mapped", venue.name)

    def resolve_player(
        self,
        *,
        source_name: str,
        external_id: str | None = None,
        full_name: str,
        current_team_id=None,
        create_missing: bool = False,
    ) -> MappingResult:
        if external_id:
            existing = self._mapped_entity(Player, source_name, "player", external_id)
            if existing is not None:
                return MappingResult(existing.id, "mapped", existing.full_name)

        alias_map = {
            alias.lower(): canonical
            for canonical, aliases in _alias_payload().get("player_aliases.json", {}).items()
            for alias in aliases
        }
        canonical_name = alias_map.get(full_name.strip().lower(), full_name)
        statement = select(Player).where(Player.full_name == canonical_name)
        if current_team_id is not None:
            preferred = self.session.scalar(
                statement.where(Player.current_team_id == current_team_id)
            )
        else:
            preferred = None
        player = preferred or self.session.scalar(statement)
        if player is None and create_missing:
            player = self.fixture_service.upsert_player(
                full_name=canonical_name,
                player_code=external_id,
                current_team_id=current_team_id,
            )
        if player is None:
            return MappingResult(None, "unresolved")

        if external_id:
            self._upsert_mapping(
                source_name=source_name,
                entity_type="player",
                external_id=external_id,
                canonical_id=player.id,
                confidence_score=1.0,
                mapping_status="mapped",
                notes=f"Resolved by canonical player name: {canonical_name}",
            )
        return MappingResult(player.id, "mapped", player.full_name)

    def _mapped_entity(self, model, source_name: str, entity_type: str, external_id: str):
        mapping = self.session.scalar(
            select(ExternalIdMapping).where(
                ExternalIdMapping.source_name == source_name,
                ExternalIdMapping.entity_type == entity_type,
                ExternalIdMapping.external_id == external_id,
            )
        )
        if mapping is None:
            return None
        return self.session.get(model, mapping.canonical_id)

    def _upsert_mapping(
        self,
        *,
        source_name: str,
        entity_type: str,
        external_id: str,
        canonical_id,
        confidence_score: float,
        mapping_status: str,
        notes: str,
    ) -> None:
        mapping = self.session.scalar(
            select(ExternalIdMapping).where(
                ExternalIdMapping.source_name == source_name,
                ExternalIdMapping.entity_type == entity_type,
                ExternalIdMapping.external_id == external_id,
            )
        )
        if mapping is None:
            mapping = ExternalIdMapping(
                source_name=source_name,
                entity_type=entity_type,
                external_id=external_id,
                canonical_id=canonical_id,
                confidence_score=confidence_score,
                mapping_status=mapping_status,
                notes=notes,
            )
            self.session.add(mapping)
        else:
            mapping.canonical_id = canonical_id
            mapping.confidence_score = confidence_score
            mapping.mapping_status = mapping_status
            mapping.notes = notes
        self.session.flush()

    def _derive_team_code(self, name: str) -> str:
        words = [chunk for chunk in name.replace("-", " ").split() if chunk]
        if len(words) == 1:
            return words[0][:3].upper()
        return "".join(word[0] for word in words[:3]).upper()
