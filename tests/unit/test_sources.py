from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from afl_prediction_agent.sources.afl.connector import FitzRoyBridge, FitzRoyConnector
from afl_prediction_agent.sources.afl_com.injuries import AFLComInjuryConnector
from afl_prediction_agent.sources.bom.connector import BomWeatherConnector
from afl_prediction_agent.sources.footywire.injuries import FootyWireInjuryConnector
from afl_prediction_agent.sources.odds.the_odds_api import TheOddsApiConnector
from afl_prediction_agent.sources.squiggle.api import SquiggleConnector


def test_fitzroy_bridge_serializes_command(monkeypatch, tmp_path) -> None:
    bridge_script = tmp_path / "fitzroy_bridge.R"
    bridge_script.write_text("# noop", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr("afl_prediction_agent.sources.afl.connector.subprocess.run", fake_run)
    bridge = FitzRoyBridge(rscript_bin="Rscript", timeout_seconds=9, bridge_script=bridge_script)

    response = bridge.fetch("fixtures", season_year=2026, round_number=7, source="AFL")

    assert response.rows == []
    assert captured["timeout"] == 9
    command = captured["command"]
    assert command[0] == "Rscript"
    assert command[2] == "fixtures"
    assert json.loads(command[3]) == {
        "season": 2026,
        "round_number": 7,
        "source": "AFL",
        "comp": "AFLM",
    }


def test_fitzroy_bridge_surfaces_subprocess_errors(monkeypatch, tmp_path) -> None:
    bridge_script = tmp_path / "fitzroy_bridge.R"
    bridge_script.write_text("# noop", encoding="utf-8")

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="fitzRoy exploded")

    monkeypatch.setattr("afl_prediction_agent.sources.afl.connector.subprocess.run", fake_run)
    bridge = FitzRoyBridge(bridge_script=bridge_script)

    with pytest.raises(RuntimeError, match="fitzRoy exploded"):
        bridge.fetch("fixtures", season_year=2026, round_number=7)


def test_fitzroy_fixture_mapper_supports_current_official_fields() -> None:
    match = FitzRoyConnector()._fixture_row_to_match(
        {
            "round.year": "2024",
            "round.roundNumber": 1,
            "round.name": "Round 1",
            "match.utcStartTime": "2024-03-14T08:30:00",
            "match.homeTeam.name": "Carlton",
            "match.homeTeam.nickname": "Blues",
            "match.homeTeam.abbr": "CARL",
            "match.homeTeam.teamId": "CD_T30",
            "match.awayTeam.name": "Richmond",
            "match.awayTeam.nickname": "Tigers",
            "match.awayTeam.abbr": "RICH",
            "match.awayTeam.teamId": "CD_T120",
            "venue.name": "MCG",
            "venue.address": "Melbourne",
            "venue.state": "VIC",
            "venue.venueId": "CD_V40",
            "venue.timeZone": "Australia/Melbourne",
            "match.matchId": "CD_M20240140101",
            "match.status": "CONCLUDED",
            "homeTeamScore.matchScore.totalScore": 86,
            "awayTeamScore.matchScore.totalScore": 81,
        }
    )

    assert match.season_year == 2024
    assert match.round_number == 1
    assert match.home_team.name == "Carlton"
    assert match.away_team.name == "Richmond"
    assert match.home_score == 86
    assert match.away_score == 81
    assert match.match_code == "CD_M20240140101"


def test_fitzroy_fixture_mapper_supports_afltables_fields() -> None:
    match = FitzRoyConnector(source="afltables")._fixture_row_to_match(
        {
            "Season": 2024,
            "Round.Number": 1,
            "Round": "R1",
            "Date": "2024-03-07",
            "Home.Team": "Sydney",
            "Away.Team": "Melbourne",
            "Venue": "S.C.G.",
            "Game": 16407,
            "Home.Points": 86,
            "Away.Points": 64,
        }
    )

    assert match.season_year == 2024
    assert match.round_number == 1
    assert match.home_team.name == "Sydney"
    assert match.away_team.name == "Melbourne"
    assert match.home_score == 86
    assert match.away_score == 64
    assert match.status == "CONCLUDED"


def test_fitzroy_lineup_mapper_supports_current_official_fields() -> None:
    row = {
        "providerId": "CD_M20240140101",
        "teamName": "Carlton",
        "teamNickname": "Blues",
        "teamAbbr": "CARL",
        "teamId": "CD_T30",
        "teamType": "home",
        "player.playerId": "CD_I997316",
        "player.playerName.givenName": "Lewis",
        "player.playerName.surname": "Young",
        "position": "BPL",
    }

    class StubBridge:
        def fetch(self, *args, **kwargs):
            return SimpleNamespace(
                envelope=SimpleNamespace(response_meta={}),
                rows=[row],
            )

    _, snapshots = FitzRoyConnector(bridge=StubBridge()).fetch_lineups(season_year=2024, round_number=1)

    assert len(snapshots) == 1
    assert snapshots[0].team.name == "Carlton"
    assert snapshots[0].home_or_away == "home"
    assert snapshots[0].players[0].source_player_name == "Lewis Young"
    assert snapshots[0].players[0].source_player_id == "CD_I997316"


def test_fitzroy_team_stats_mapper_supports_archive_team_field() -> None:
    row = {
        "Game": 16407,
        "Team": "Adelaide",
        "Home.Team": "Sydney",
        "Away.Team": "Melbourne",
        "1%_for": 1043,
    }

    class StubBridge:
        def fetch(self, *args, **kwargs):
            return SimpleNamespace(
                envelope=SimpleNamespace(response_meta={}),
                rows=[row],
            )

    _, normalized = FitzRoyConnector(bridge=StubBridge(), source="afltables").fetch_team_stats(
        season_year=2024,
        round_number=1,
    )

    assert normalized[0].match_code == "16407"
    assert normalized[0].team_name == "Adelaide"
    assert normalized[0].home_team_name == "Sydney"
    assert normalized[0].away_team_name == "Melbourne"
    assert normalized[0].stats["1%_for"] == 1043


def test_fitzroy_player_stats_mapper_supports_current_official_fields() -> None:
    row = {
        "providerId": "CD_M20240140101",
        "team.name": "Carlton",
        "player.givenName": "Blake",
        "player.surname": "Acres",
        "player.playerId": "CD_I296294",
        "disposals": 17,
    }

    class StubBridge:
        def fetch(self, *args, **kwargs):
            return SimpleNamespace(
                envelope=SimpleNamespace(response_meta={}),
                rows=[row],
            )

    _, normalized = FitzRoyConnector(bridge=StubBridge()).fetch_player_stats(season_year=2024, round_number=1)

    assert normalized[0].match_code == "CD_M20240140101"
    assert normalized[0].team_name == "Carlton"
    assert normalized[0].player_name == "Blake Acres"
    assert normalized[0].source_player_id == "CD_I296294"
    assert normalized[0].stats["disposals"] == 17


def test_fitzroy_team_stats_fallback_derives_from_player_stats() -> None:
    player_rows = [
        {
            "providerId": "CD_M20240140101",
            "home.team.name": "Carlton",
            "away.team.name": "Richmond",
            "team.name": "Carlton",
            "inside50s": 4,
            "clearances.totalClearances": 2,
            "contestedPossessions": 5,
            "marksInside50": 1,
            "tacklesInside50": 1,
            "disposals": 17,
            "kicks": 11,
            "handballs": 6,
            "goals": 0,
        },
        {
            "providerId": "CD_M20240140101",
            "home.team.name": "Carlton",
            "away.team.name": "Richmond",
            "team.name": "Carlton",
            "inside50s": 3,
            "clearances.totalClearances": 1,
            "contestedPossessions": 4,
            "marksInside50": 0,
            "tacklesInside50": 2,
            "disposals": 12,
            "kicks": 8,
            "handballs": 4,
            "goals": 1,
        },
    ]

    class StubBridge:
        def fetch(self, dataset, **kwargs):
            if dataset == "team_stats":
                raise RuntimeError("unsupported")
            return SimpleNamespace(
                envelope=SimpleNamespace(response_meta={}, raw_payload={"rows": player_rows}),
                rows=player_rows,
            )

    envelope, normalized = FitzRoyConnector(bridge=StubBridge()).fetch_team_stats(season_year=2024, round_number=1)

    assert envelope.response_meta["derived_from_player_stats"] is True
    assert len(normalized) == 1
    assert normalized[0].team_name == "Carlton"
    assert normalized[0].home_team_name == "Carlton"
    assert normalized[0].away_team_name == "Richmond"
    assert normalized[0].stats["inside_50"] == 7.0
    assert normalized[0].stats["clearances"] == 3.0
    assert normalized[0].stats["goals"] == 1.0


def test_afl_com_parser_reads_tables_in_order() -> None:
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {"headline":"AFL Injury List","dateModified":"2026-04-23T12:00:00Z"}
        </script>
      </head>
      <body>
        <h2>FULL INJURY LIST</h2>
        <p>Check out the injury updates from all 18 clubs.</p>
        <table>
          <tbody>
            <tr><td>Player One</td><td>Hamstring</td><td>Test</td></tr>
            <tr><td>Player Two</td><td>Knee</td><td>3-4 weeks</td></tr>
          </tbody>
        </table>
        <table>
          <tbody>
            <tr><td>Player Three</td><td>Ankle</td><td>TBC</td></tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    snapshot = AFLComInjuryConnector().parse_html(html)

    assert snapshot.published_at == datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    assert snapshot.entries[0].team_name == "Adelaide Crows"
    assert snapshot.entries[0].status_label == "Test"
    assert snapshot.entries[2].team_name == "Brisbane Lions"


def test_footywire_parser_reads_team_tables() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>Adelaide Crows (2 Players)</th></tr>
        <tr><th>Player</th><th>Injury</th><th>Returning</th></tr>
        <tr><td>Player One</td><td>Hamstring</td><td>Test</td></tr>
        <tr><td>Player Two</td><td>Knee</td><td>3-4 weeks</td></tr>
      </table>
    </body></html>
    """
    snapshot = FootyWireInjuryConnector().parse_html(html)

    assert len(snapshot.entries) == 2
    assert snapshot.entries[0].team_name == "Adelaide Crows"
    assert snapshot.entries[0].uncertainty_flag is True


class FakeBomFtpClient:
    def __init__(self, observation_payload: bytes, forecast_payload: str) -> None:
        self.observation_payload = observation_payload
        self.forecast_payload = forecast_payload

    def read_binary(self, file_name: str) -> bytes:
        assert file_name == "IDV60910.tgz"
        return self.observation_payload

    def read_text(self, file_name: str) -> str:
        assert file_name == "IDV10753.xml"
        return self.forecast_payload


def test_bom_connector_combines_observation_and_forecast(tmp_path) -> None:
    mapping_path = tmp_path / "venue_bom_mapping.json"
    mapping_path.write_text(
        json.dumps(
            [
                {
                    "venue_name": "MCG",
                    "aliases": ["MCG"],
                    "city": "Melbourne",
                    "state_code": "VIC",
                    "station_id": "95936",
                    "forecast_product_id": "IDV10753",
                    "forecast_location_name": "Melbourne",
                }
            ]
        ),
        encoding="utf-8",
    )
    observation_json = json.dumps(
        {
            "observations": {
                "data": [
                    {
                        "air_temp": 16.6,
                        "wind_spd_kmh": 7,
                        "weather": "-",
                    }
                ]
            }
        }
    ).encode("utf-8")
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        info = tarfile.TarInfo("IDV60910.95936.json")
        info.size = len(observation_json)
        archive.addfile(info, io.BytesIO(observation_json))
    forecast_xml = """
    <product><forecast>
      <area type="location" description="Melbourne">
        <forecast-period start-time-local="2026-04-24T00:00:00+10:00" end-time-local="2026-04-25T00:00:00+10:00">
          <text type="precis">Showers.</text>
          <text type="probability_of_precipitation">40%</text>
        </forecast-period>
      </area>
    </forecast></product>
    """
    connector = BomWeatherConnector(
        ftp_client=FakeBomFtpClient(archive_bytes.getvalue(), forecast_xml),
        mapping_path=mapping_path,
    )

    envelope, snapshot = connector.fetch_weather_for_venue(
        venue_name="MCG",
        scheduled_at=datetime(2026, 4, 24, 19, 50, tzinfo=timezone.utc),
    )

    assert envelope.source_name == "bom"
    assert snapshot.temperature_c == 16.6
    assert snapshot.wind_kmh == 7.0
    assert snapshot.rain_probability_pct == 40.0
    assert snapshot.weather_text == "Showers."


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, *, headers=None, params=None, expect_json=False):
        self.calls.append({"url": url, "params": params, "headers": headers, "expect_json": expect_json})
        response = self.responses.pop(0)
        return response


def test_odds_connector_filters_to_allowlisted_bookmakers(tmp_path) -> None:
    bookmaker_path = tmp_path / "odds_bookmakers_au.json"
    bookmaker_path.write_text(json.dumps(["sportsbet", "tab"]), encoding="utf-8")
    http_client = FakeHttpClient(
        [
            SimpleNamespace(
                status_code=200,
                headers={"x-requests-remaining": "99", "x-requests-used": "1"},
                text="",
                json_data=[
                    {
                        "home_team": "Carlton",
                        "away_team": "Geelong",
                        "commence_time": "2026-04-24T09:50:00Z",
                        "bookmakers": [
                            {
                                "key": "sportsbet",
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "outcomes": [
                                            {"name": "Carlton", "price": 1.8},
                                            {"name": "Geelong", "price": 2.0},
                                        ],
                                    }
                                ],
                            },
                            {
                                "key": "not_allowed",
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "outcomes": [
                                            {"name": "Carlton", "price": 1.7},
                                            {"name": "Geelong", "price": 2.1},
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            )
        ]
    )
    connector = TheOddsApiConnector(http_client=http_client, api_key="test", bookmaker_path=bookmaker_path)

    envelope, snapshots = connector.fetch_head_to_head()

    assert envelope.response_meta["event_count"] == 1
    assert len(snapshots) == 1
    assert [book.bookmaker_key for book in snapshots[0].books] == ["sportsbet"]


def test_squiggle_connector_prefers_aggregate_source() -> None:
    http_client = FakeHttpClient(
        [
            SimpleNamespace(status_code=200, headers={}, text="", json_data={"sources": [{"id": 1, "name": "SomeModel"}, {"id": 2, "name": "aggregate"}]}),
            SimpleNamespace(status_code=200, headers={}, text="", json_data={"tips": [{"gameid": 10, "sourceid": 2, "prob": 0.61, "margin": 14.0, "tip": "Carlton"}]}),
            SimpleNamespace(status_code=200, headers={}, text="", json_data={"games": [{"id": 10, "hteam": "Carlton", "ateam": "Geelong"}]}),
        ]
    )
    connector = SquiggleConnector(http_client=http_client)

    envelope, predictions = connector.fetch_predictions(season_year=2026, round_number=7)

    assert envelope.response_meta["preferred_source"] == "aggregate"
    assert len(predictions) == 1
    assert predictions[0].home_win_probability == 0.61
    assert predictions[0].away_win_probability == 0.39
    assert http_client.calls[0]["headers"]["User-Agent"].startswith("AFL Prediction Agent - ")


def test_squiggle_default_user_agent_uses_contact_email() -> None:
    assert SquiggleConnector._default_user_agent("AFL Prediction Agent", "bot@example.com") == (
        "AFL Prediction Agent - bot@example.com"
    )


def test_squiggle_connector_coerces_string_numeric_fields() -> None:
    http_client = FakeHttpClient(
        [
            SimpleNamespace(status_code=200, headers={}, text="", json_data={"sources": [{"id": 2, "name": "aggregate"}]}),
            SimpleNamespace(status_code=200, headers={}, text="", json_data={"tips": [{"gameid": 10, "sourceid": 2, "prob": "0.61", "margin": "14"}]}),
            SimpleNamespace(status_code=200, headers={}, text="", json_data={"games": [{"id": 10, "hteam": "Carlton", "ateam": "Geelong"}]}),
        ]
    )
    connector = SquiggleConnector(http_client=http_client)

    _, predictions = connector.fetch_predictions(season_year=2026, round_number=7)

    assert predictions[0].predicted_margin == 14.0
    assert predictions[0].predicted_winner_name == "Carlton"
