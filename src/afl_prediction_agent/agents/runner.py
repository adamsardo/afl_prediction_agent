from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from afl_prediction_agent.agents.adapters import LLMAdapter, build_adapter
from afl_prediction_agent.contracts import (
    AnalystResponse,
    CaseAgentResponse,
    FinalDecisionResponse,
    MatchDossierContract,
    ModelSettings,
    RunConfigFile,
)
from afl_prediction_agent.core.db.base import utcnow
from afl_prediction_agent.storage.models import AgentStep, PromptTemplate, ValidationLog


ANALYST_STEP_NAMES = [
    "form_analyst_v1",
    "selection_analyst_v1",
    "venue_weather_analyst_v1",
    "market_analyst_v1",
]
CASE_STEP_NAMES = ["home_case_v1", "away_case_v1"]
FINAL_STEP_NAME = "final_decision_v1"
CORRECTION_STEP_NAME = "correction_pass_v1"


@dataclass(slots=True)
class MatchAgentRunResult:
    analysts: dict[str, dict[str, Any]]
    cases: dict[str, dict[str, Any]]
    final_response: FinalDecisionResponse | None
    final_step_id: Any | None
    correction_pass_count: int
    failed_steps: list[dict[str, Any]]


class AgentPipelineRunner:
    def __init__(self, session: Session, prompt_set_version: str) -> None:
        self.session = session
        self.prompt_set_version = prompt_set_version

    def run_for_match(
        self,
        *,
        round_run_id,
        match_id,
        dossier: MatchDossierContract,
        config: RunConfigFile,
    ) -> MatchAgentRunResult:
        analysts: dict[str, dict[str, Any]] = {}
        cases: dict[str, dict[str, Any]] = {}
        failed_steps: list[dict[str, Any]] = []
        analyst_adapter = build_adapter(config.analyst_model.provider)
        case_adapter = build_adapter(config.case_model.provider)
        final_adapter = build_adapter(config.final_model.provider)

        for step_name in ANALYST_STEP_NAMES:
            try:
                result = self._execute_step(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    step_name=step_name,
                    input_json={"dossier": dossier.model_dump(mode="json")},
                    settings=config.analyst_model,
                    adapter=analyst_adapter,
                    schema_type=AnalystResponse,
                )
                analysts[step_name] = result
                self._record_validation(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    component_name=step_name,
                    validation_status="passed",
                )
            except Exception as exc:
                failed_steps.append({"step_name": step_name, "error": str(exc)})
                self._record_validation(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    component_name=step_name,
                    validation_status="failed",
                    errors=[{"message": str(exc)}],
                )

        for step_name in CASE_STEP_NAMES:
            try:
                result = self._execute_step(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    step_name=step_name,
                    input_json={
                        "dossier": dossier.model_dump(mode="json"),
                        "analysts": analysts,
                    },
                    settings=config.case_model,
                    adapter=case_adapter,
                    schema_type=CaseAgentResponse,
                )
                cases[step_name] = result
                self._record_validation(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    component_name=step_name,
                    validation_status="passed",
                )
            except Exception as exc:
                failed_steps.append({"step_name": step_name, "error": str(exc)})
                self._record_validation(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    component_name=step_name,
                    validation_status="failed",
                    errors=[{"message": str(exc)}],
                )

        correction_pass_count = 0
        final_response: FinalDecisionResponse | None = None
        final_step_id = None
        try:
            final_json, final_step_id = self._execute_step_with_metadata(
                round_run_id=round_run_id,
                match_id=match_id,
                step_name=FINAL_STEP_NAME,
                input_json={
                    "dossier": dossier.model_dump(mode="json"),
                    "analysts": analysts,
                    "cases": cases,
                },
                settings=config.final_model,
                adapter=final_adapter,
                schema_type=FinalDecisionResponse,
            )
            final_response = FinalDecisionResponse.model_validate(final_json)
        except Exception as exc:
            correction_pass_count = 1
            failed_steps.append({"step_name": FINAL_STEP_NAME, "error": str(exc)})
            self._record_validation(
                round_run_id=round_run_id,
                match_id=match_id,
                component_name=FINAL_STEP_NAME,
                validation_status="failed",
                errors=[{"message": str(exc)}],
            )
            try:
                final_json, final_step_id = self._execute_step_with_metadata(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    step_name=CORRECTION_STEP_NAME,
                    input_json={
                        "dossier": dossier.model_dump(mode="json"),
                        "analysts": analysts,
                        "cases": cases,
                        "failed_steps": failed_steps,
                        "validation_error": str(exc),
                    },
                    settings=config.final_model,
                    adapter=final_adapter,
                    schema_type=FinalDecisionResponse,
                )
                final_response = FinalDecisionResponse.model_validate(final_json)
            except Exception as correction_exc:
                failed_steps.append({"step_name": CORRECTION_STEP_NAME, "error": str(correction_exc)})
                self._record_validation(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    component_name=CORRECTION_STEP_NAME,
                    validation_status="failed",
                    errors=[{"message": str(correction_exc)}],
                )

        if final_response is not None:
            self._record_validation(
                round_run_id=round_run_id,
                match_id=match_id,
                component_name=FINAL_STEP_NAME,
                validation_status="passed",
            )
        return MatchAgentRunResult(
            analysts=analysts,
            cases=cases,
            final_response=final_response,
            final_step_id=final_step_id,
            correction_pass_count=correction_pass_count,
            failed_steps=failed_steps,
        )

    def _execute_step(
        self,
        *,
        round_run_id,
        match_id,
        step_name: str,
        input_json: dict[str, Any],
        settings: ModelSettings,
        adapter: LLMAdapter,
        schema_type,
    ) -> dict[str, Any]:
        output_json, _ = self._execute_step_with_metadata(
            round_run_id=round_run_id,
            match_id=match_id,
            step_name=step_name,
            input_json=input_json,
            settings=settings,
            adapter=adapter,
            schema_type=schema_type,
        )
        return output_json

    def _execute_step_with_metadata(
        self,
        *,
        round_run_id,
        match_id,
        step_name: str,
        input_json: dict[str, Any],
        settings: ModelSettings,
        adapter: LLMAdapter,
        schema_type,
    ) -> tuple[dict[str, Any], Any]:
        prompt_template = self._get_prompt_template(step_name)
        rendered_prompt = self._render_prompt(prompt_template.template_text, input_json)
        step = AgentStep(
            round_run_id=round_run_id,
            match_id=match_id,
            step_name=step_name,
            prompt_template_id=prompt_template.id,
            rendered_prompt=rendered_prompt,
            model_provider=settings.provider,
            model_name=settings.model,
            temperature=Decimal(str(settings.temperature)) if settings.temperature is not None else None,
            reasoning_effort=settings.reasoning_effort,
            input_json=input_json,
            provider_meta={},
            status="running",
            started_at=utcnow(),
        )
        self.session.add(step)
        self.session.flush()
        try:
            adapter_result = adapter.run_structured(
                step_name=step_name,
                prompt=rendered_prompt,
                input_json=input_json,
                model_name=settings.model,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                output_schema=schema_type.model_json_schema(),
            )
            validated = schema_type.model_validate(adapter_result.output_json)
            step.output_json = validated.model_dump(mode="json")
            step.tokens_input = adapter_result.tokens_input
            step.tokens_output = adapter_result.tokens_output
            step.provider_meta = adapter_result.provider_meta
            step.status = "completed"
            step.completed_at = utcnow()
            self.session.flush()
            return step.output_json, step.id
        except Exception as exc:
            step.status = "failed"
            step.error_message = str(exc)
            step.completed_at = utcnow()
            self.session.flush()
            raise

    def _get_prompt_template(self, step_name: str) -> PromptTemplate:
        template = self.session.scalar(
            select(PromptTemplate).where(
                PromptTemplate.prompt_set_version == self.prompt_set_version,
                PromptTemplate.step_name == step_name,
                PromptTemplate.is_active.is_(True),
            )
        )
        if template is None:
            raise ValueError(f"Prompt template not found for step {step_name}")
        return template

    def _render_prompt(self, template_text: str, input_json: dict[str, Any]) -> str:
        import json

        serialized = json.dumps(input_json, indent=2, sort_keys=True, default=str)
        try:
            return template_text.format(
                input_json=serialized,
                dossier_json=json.dumps(input_json.get("dossier"), indent=2, sort_keys=True, default=str),
            )
        except KeyError:
            return f"{template_text}\n\nINPUT\n{serialized}"

    def _record_validation(
        self,
        *,
        round_run_id,
        match_id,
        component_name: str,
        validation_status: str,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        self.session.add(
            ValidationLog(
                round_run_id=round_run_id,
                match_id=match_id,
                component_name=component_name,
                validation_status=validation_status,
                errors=errors,
            )
        )
