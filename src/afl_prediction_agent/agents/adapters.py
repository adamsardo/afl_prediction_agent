from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from afl_prediction_agent.agents.codex_app_server import get_codex_app_server_client
from afl_prediction_agent.contracts import (
    AnalystResponse,
    CaseAgentResponse,
    FinalDecisionResponse,
    MatchDossierContract,
)


def to_openai_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict_schema = copy.deepcopy(schema)
    _normalize_openai_schema_node(strict_schema)
    return strict_schema


def _normalize_openai_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_openai_schema_node(item)
        return
    if not isinstance(node, dict):
        return

    node.pop("default", None)
    node.pop("examples", None)

    if "$defs" in node and isinstance(node["$defs"], dict):
        for child in node["$defs"].values():
            _normalize_openai_schema_node(child)

    if "definitions" in node and isinstance(node["definitions"], dict):
        for child in node["definitions"].values():
            _normalize_openai_schema_node(child)

    if "properties" in node and isinstance(node["properties"], dict):
        for child in node["properties"].values():
            _normalize_openai_schema_node(child)
        node["required"] = list(node["properties"].keys())
        node.setdefault("additionalProperties", False)

    if "items" in node:
        _normalize_openai_schema_node(node["items"])

    if "anyOf" in node:
        _normalize_openai_schema_node(node["anyOf"])

    if "allOf" in node:
        _normalize_openai_schema_node(node["allOf"])

    if "oneOf" in node:
        _normalize_openai_schema_node(node["oneOf"])

    if "prefixItems" in node:
        _normalize_openai_schema_node(node["prefixItems"])

    if "not" in node:
        _normalize_openai_schema_node(node["not"])

    if "if" in node:
        _normalize_openai_schema_node(node["if"])
    if "then" in node:
        _normalize_openai_schema_node(node["then"])
    if "else" in node:
        _normalize_openai_schema_node(node["else"])


@dataclass(slots=True)
class StructuredAdapterResult:
    output_json: dict[str, Any]
    tokens_input: int | None
    tokens_output: int | None
    provider_meta: dict[str, Any]


class LLMAdapter:
    provider_name: str

    def run_structured(
        self,
        *,
        step_name: str,
        prompt: str,
        input_json: dict[str, Any],
        model_name: str,
        temperature: float | None,
        reasoning_effort: str | None,
        output_schema: dict[str, Any],
    ) -> StructuredAdapterResult:
        raise NotImplementedError


class HeuristicAdapter(LLMAdapter):
    provider_name = "heuristic"

    def run_structured(
        self,
        *,
        step_name: str,
        prompt: str,
        input_json: dict[str, Any],
        model_name: str,
        temperature: float | None,
        reasoning_effort: str | None,
        output_schema: dict[str, Any],
    ) -> StructuredAdapterResult:
        if step_name in {
            "form_analyst_v1",
            "selection_analyst_v1",
            "venue_weather_analyst_v1",
            "market_analyst_v1",
        }:
            output = self._run_analyst(step_name, input_json)
        elif step_name in {"home_case_v1", "away_case_v1"}:
            output = self._run_case(step_name, input_json)
        elif step_name in {"final_decision_v1", "correction_pass_v1"}:
            output = self._run_final(input_json)
        else:
            raise ValueError(f"Unsupported heuristic step: {step_name}")
        rendered = json.dumps(output, sort_keys=True)
        return StructuredAdapterResult(
            output_json=output,
            tokens_input=max(len(prompt) // 4, 1),
            tokens_output=max(len(rendered) // 4, 1),
            provider_meta={},
        )

    def _run_analyst(self, step_name: str, input_json: dict[str, Any]) -> dict[str, Any]:
        dossier = MatchDossierContract.model_validate(input_json["dossier"])
        if step_name == "form_analyst_v1":
            home_win_rate = dossier.form.home_recent_form.get("win_rate", 0.5)
            away_win_rate = dossier.form.away_recent_form.get("win_rate", 0.5)
            leans_to = "home" if home_win_rate >= away_win_rate else "away"
            signals = []
            for edge in dossier.form.team_stat_edges[:3]:
                signals.append(
                    {
                        "label": edge["label"],
                        "leans_to": edge["leans_to"],
                        "strength": edge["strength"],
                        "evidence": edge["evidence"],
                    }
                )
            if not signals:
                signals.append(
                    {
                        "label": "recent win rate",
                        "leans_to": leans_to,
                        "strength": round(abs(home_win_rate - away_win_rate), 3),
                        "evidence": f"home {home_win_rate:.2f} vs away {away_win_rate:.2f}",
                    }
                )
            output = AnalystResponse(
                summary=(
                    f"Recent form leans {leans_to} with baseline form gap "
                    f"{abs(home_win_rate - away_win_rate):.2f}."
                ),
                signals=signals,
                risks=dossier.uncertainties[:2],
                unknowns=dossier.uncertainties[2:],
            )
            return output.model_dump(mode="json")

        if step_name == "selection_analyst_v1":
            home_strength = dossier.selection.home_lineup_strength
            away_strength = dossier.selection.away_lineup_strength
            leans_to = "home" if home_strength >= away_strength else "away"
            output = AnalystResponse(
                summary=(
                    f"Selection profile leans {leans_to}; named changes are "
                    f"{dossier.selection.home_named_changes} vs {dossier.selection.away_named_changes}."
                ),
                signals=[
                    {
                        "label": "lineup strength",
                        "leans_to": leans_to,
                        "strength": min(abs(home_strength - away_strength) / 25.0, 1.0),
                        "evidence": f"home {home_strength:.1f} vs away {away_strength:.1f}",
                    }
                ],
                risks=[
                    absence["player_name"]
                    for absence in dossier.selection.key_absences[:3]
                ],
                unknowns=dossier.uncertainties,
            )
            return output.model_dump(mode="json")

        if step_name == "venue_weather_analyst_v1":
            lean = "home" if dossier.venue_weather.home_ground_edge else "neutral"
            output = AnalystResponse(
                summary=(
                    "Venue and weather context is stable."
                    if lean == "neutral"
                    else "Venue context gives the home side a structural edge."
                ),
                signals=[
                    {
                        "label": "venue context",
                        "leans_to": lean,
                        "strength": 0.6 if lean == "home" else 0.25,
                        "evidence": json.dumps(dossier.venue_weather.travel_context),
                    }
                ],
                risks=[dossier.venue_weather.forecast.get("weather_text") or "weather stable"],
                unknowns=[],
            )
            return output.model_dump(mode="json")

        market_home = dossier.market.home_implied_probability or dossier.baseline.home_win_probability
        market_away = dossier.market.away_implied_probability or dossier.baseline.away_win_probability
        lean = "home" if market_home >= market_away else "away"
        disagreement = abs(market_home - dossier.baseline.home_win_probability)
        output = AnalystResponse(
            summary=f"Market consensus leans {lean} with {dossier.market.bookmaker_count} books in sample.",
            signals=[
                {
                    "label": "market implied probability",
                    "leans_to": lean,
                    "strength": min(abs(market_home - 0.5) * 2.0, 1.0),
                    "evidence": f"home {market_home:.3f} vs away {market_away:.3f}",
                }
            ],
            risks=[
                "market disagrees with baseline"
                if disagreement > 0.1
                else "market broadly aligned with baseline"
            ],
            unknowns=[],
        )
        return output.model_dump(mode="json")

    def _run_case(self, step_name: str, input_json: dict[str, Any]) -> dict[str, Any]:
        dossier = MatchDossierContract.model_validate(input_json["dossier"])
        analysts = {
            name: AnalystResponse.model_validate(payload)
            for name, payload in input_json["analysts"].items()
        }
        side = "home" if step_name == "home_case_v1" else "away"
        strongest_points = []
        weak_points = []
        for analyst_name, response in analysts.items():
            for signal in response.signals:
                if signal.leans_to == side:
                    strongest_points.append(
                        {
                            "label": signal.label,
                            "strength": signal.strength,
                            "evidence": f"{analyst_name}: {signal.evidence}",
                        }
                    )
                elif signal.leans_to not in {side, "neutral"}:
                    weak_points.append(f"{analyst_name}: {signal.label}")
        if side == "home":
            strongest_points.append(
                {
                    "label": "baseline probability",
                    "strength": dossier.baseline.home_win_probability,
                    "evidence": f"baseline home win probability {dossier.baseline.home_win_probability:.3f}",
                }
            )
        else:
            strongest_points.append(
                {
                    "label": "baseline probability",
                    "strength": dossier.baseline.away_win_probability,
                    "evidence": f"baseline away win probability {dossier.baseline.away_win_probability:.3f}",
                }
            )
        strongest_points = sorted(
            strongest_points,
            key=lambda item: item["strength"],
            reverse=True,
        )[:4]
        output = CaseAgentResponse(
            side=side,
            case_summary=f"{side.capitalize()} case leans on its clearest supportive signals.",
            strongest_points=strongest_points,
            weak_points=weak_points[:3],
            rebuttals=dossier.uncertainties[:2],
        )
        return output.model_dump(mode="json")

    def _run_final(self, input_json: dict[str, Any]) -> dict[str, Any]:
        dossier = MatchDossierContract.model_validate(input_json["dossier"])
        analysts = {
            name: AnalystResponse.model_validate(payload)
            for name, payload in input_json["analysts"].items()
        }
        cases = {
            name: CaseAgentResponse.model_validate(payload)
            for name, payload in input_json["cases"].items()
        }
        baseline_home = dossier.baseline.home_win_probability
        market_home = dossier.market.home_implied_probability or baseline_home
        squiggle_home = dossier.benchmarks.squiggle.get("home_win_probability") or baseline_home
        analyst_home = 0.0
        analyst_away = 0.0
        for report in analysts.values():
            for signal in report.signals:
                if signal.leans_to == "home":
                    analyst_home += signal.strength
                elif signal.leans_to == "away":
                    analyst_away += signal.strength
        home_case = cases.get("home_case_v1")
        away_case = cases.get("away_case_v1")
        case_home = (
            sum(point.strength for point in home_case.strongest_points)
            if home_case is not None
            else 0.0
        )
        case_away = (
            sum(point.strength for point in away_case.strongest_points)
            if away_case is not None
            else 0.0
        )
        home_probability = (
            baseline_home * 0.5
            + market_home * 0.3
            + squiggle_home * 0.1
            + max(min((analyst_home - analyst_away) * 0.03, 0.08), -0.08)
            + max(min((case_home - case_away) * 0.015, 0.08), -0.08)
        )
        home_probability = min(max(home_probability, 0.02), 0.98)
        away_probability = 1.0 - home_probability
        predicted_winner_team_id = (
            dossier.match.home_team.team_id
            if home_probability >= away_probability
            else dossier.match.away_team.team_id
        )
        baseline_margin = dossier.baseline.predicted_margin
        directional_margin = abs((home_probability - 0.5) * 52.0 + baseline_margin * 0.45)
        predicted_margin = max(directional_margin, 1.0)
        if predicted_winner_team_id == dossier.match.away_team.team_id:
            predicted_margin *= -1
        combined_drivers = [
            *[driver.model_dump(mode="json") for driver in dossier.baseline.top_drivers],
            *[
                {
                    "label": signal.label,
                    "leans_to": signal.leans_to,
                    "strength": signal.strength,
                    "evidence": signal.evidence,
                    "source_component": component_name,
                }
                for component_name, report in analysts.items()
                for signal in report.signals
            ],
        ]
        top_drivers = sorted(
            combined_drivers,
            key=lambda item: item["strength"],
            reverse=True,
        )[:4]
        confidence_score = min(
            90.0,
            45.0 + abs(home_probability - 0.5) * 85.0 + (3 - min(len(dossier.uncertainties), 3)) * 4.0,
        )
        uncertainty_note = (
            dossier.uncertainties[0]
            if dossier.uncertainties
            else "No major pre-lock uncertainty flagged."
        )
        rationale_summary = (
            f"Decision anchored to baseline {baseline_home:.2f}, market {market_home:.2f}, "
            f"and case balance {case_home:.2f} vs {case_away:.2f}."
        )
        output = FinalDecisionResponse(
            predicted_winner_team_id=predicted_winner_team_id,
            home_win_probability=round(home_probability, 6),
            away_win_probability=round(away_probability, 6),
            predicted_margin=round(predicted_margin, 2),
            confidence_score=round(confidence_score, 2),
            top_drivers=top_drivers,
            uncertainty_note=uncertainty_note,
            rationale_summary=rationale_summary,
        )
        return output.model_dump(mode="json")


class CodexAppServerAdapter(LLMAdapter):
    provider_name = "codex_app_server"

    def run_structured(
        self,
        *,
        step_name: str,
        prompt: str,
        input_json: dict[str, Any],
        model_name: str,
        temperature: float | None,
        reasoning_effort: str | None,
        output_schema: dict[str, Any],
    ) -> StructuredAdapterResult:
        client = get_codex_app_server_client()
        turn_result = client.run_turn(
            step_name=step_name,
            prompt=prompt,
            input_json=input_json,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
        )
        try:
            output_json = json.loads(turn_result.output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Codex app-server returned invalid JSON for {step_name}: {turn_result.output_text!r}"
            ) from exc
        return StructuredAdapterResult(
            output_json=output_json,
            tokens_input=turn_result.tokens_input,
            tokens_output=turn_result.tokens_output,
            provider_meta=turn_result.provider_meta,
        )


def build_adapter(provider_name: str) -> LLMAdapter:
    if provider_name == "heuristic":
        return HeuristicAdapter()
    if provider_name == "codex_app_server":
        return CodexAppServerAdapter()
    raise ValueError(f"Unsupported provider: {provider_name}")
