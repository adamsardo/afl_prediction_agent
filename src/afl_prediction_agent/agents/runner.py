from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from afl_prediction_agent.agents.adapters import build_adapter, to_openai_output_schema
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


@dataclass(slots=True)
class PreparedStep:
    step_name: str
    step_row: AgentStep
    rendered_prompt: str
    input_json: dict[str, Any]
    settings: ModelSettings
    schema_type: Any


@dataclass(slots=True)
class StepExecutionResult:
    output_json: dict[str, Any]
    tokens_input: int | None
    tokens_output: int | None
    provider_meta: dict[str, Any]


class AgentPipelineRunner:
    def __init__(
        self,
        session: Session,
        prompt_set_version: str,
        *,
        progress_callback: Callable[[str], None] | None = None,
        max_parallel_workers: int = 4,
    ) -> None:
        self.session = session
        self.prompt_set_version = prompt_set_version
        self.progress_callback = progress_callback
        self.executor = ThreadPoolExecutor(
            max_workers=max_parallel_workers,
            thread_name_prefix="agent-step",
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def run_for_match(
        self,
        *,
        round_run_id,
        match_id,
        dossier: MatchDossierContract,
        config: RunConfigFile,
        match_label: str | None = None,
    ) -> MatchAgentRunResult:
        analysts: dict[str, dict[str, Any]] = {}
        cases: dict[str, dict[str, Any]] = {}
        failed_steps: list[dict[str, Any]] = []
        dossier_json = dossier.model_dump(mode="json")

        self._emit_progress(match_label, "starting analyst wave")
        analyst_results, analyst_failures = self._execute_parallel_batch(
            round_run_id=round_run_id,
            match_id=match_id,
            step_names=ANALYST_STEP_NAMES,
            input_json_factory=lambda _step_name: {"dossier": dossier_json},
            settings=config.analyst_model,
            schema_type=AnalystResponse,
            match_label=match_label,
        )
        analysts.update(analyst_results)
        failed_steps.extend(analyst_failures)

        self._emit_progress(match_label, "starting case wave")
        case_results, case_failures = self._execute_parallel_batch(
            round_run_id=round_run_id,
            match_id=match_id,
            step_names=CASE_STEP_NAMES,
            input_json_factory=lambda _step_name: {
                "dossier": dossier_json,
                "analysts": analysts,
            },
            settings=config.case_model,
            schema_type=CaseAgentResponse,
            match_label=match_label,
        )
        cases.update(case_results)
        failed_steps.extend(case_failures)

        correction_pass_count = 0
        final_response: FinalDecisionResponse | None = None
        final_step_id = None
        try:
            self._emit_progress(match_label, f"starting {FINAL_STEP_NAME}")
            final_json, final_step_id = self._execute_step_with_metadata(
                round_run_id=round_run_id,
                match_id=match_id,
                step_name=FINAL_STEP_NAME,
                input_json={
                    "dossier": dossier_json,
                    "analysts": analysts,
                    "cases": cases,
                },
                settings=config.final_model,
                schema_type=FinalDecisionResponse,
                match_label=match_label,
            )
            final_response = FinalDecisionResponse.model_validate(final_json)
        except Exception as exc:
            correction_pass_count = 1
            failed_steps.append({"step_name": FINAL_STEP_NAME, "error": str(exc)})
            try:
                self._emit_progress(match_label, f"starting {CORRECTION_STEP_NAME}")
                final_json, final_step_id = self._execute_step_with_metadata(
                    round_run_id=round_run_id,
                    match_id=match_id,
                    step_name=CORRECTION_STEP_NAME,
                    input_json={
                        "dossier": dossier_json,
                        "analysts": analysts,
                        "cases": cases,
                        "failed_steps": failed_steps,
                        "validation_error": str(exc),
                    },
                    settings=config.final_model,
                    schema_type=FinalDecisionResponse,
                    match_label=match_label,
                )
                final_response = FinalDecisionResponse.model_validate(final_json)
            except Exception as correction_exc:
                failed_steps.append(
                    {"step_name": CORRECTION_STEP_NAME, "error": str(correction_exc)}
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

    def _execute_parallel_batch(
        self,
        *,
        round_run_id,
        match_id,
        step_names: list[str],
        input_json_factory: Callable[[str], dict[str, Any]],
        settings: ModelSettings,
        schema_type,
        match_label: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        prepared_steps = [
            self._prepare_step(
                round_run_id=round_run_id,
                match_id=match_id,
                step_name=step_name,
                input_json=input_json_factory(step_name),
                settings=settings,
                schema_type=schema_type,
            )
            for step_name in step_names
        ]
        futures: dict[Future[StepExecutionResult], PreparedStep] = {
            self.executor.submit(self._run_step_worker, prepared_step): prepared_step
            for prepared_step in prepared_steps
        }
        results: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        for future in as_completed(futures):
            prepared_step = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                self._apply_step_failure(prepared_step, exc, match_label=match_label)
                failures.append({"step_name": prepared_step.step_name, "error": str(exc)})
            else:
                results[prepared_step.step_name] = self._apply_step_success(
                    prepared_step,
                    result,
                    match_label=match_label,
                )
        return results, failures

    def _execute_step_with_metadata(
        self,
        *,
        round_run_id,
        match_id,
        step_name: str,
        input_json: dict[str, Any],
        settings: ModelSettings,
        schema_type,
        match_label: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        prepared_step = self._prepare_step(
            round_run_id=round_run_id,
            match_id=match_id,
            step_name=step_name,
            input_json=input_json,
            settings=settings,
            schema_type=schema_type,
        )
        try:
            result = self._run_step_worker(prepared_step)
            output_json = self._apply_step_success(
                prepared_step,
                result,
                match_label=match_label,
            )
            return output_json, prepared_step.step_row.id
        except Exception as exc:
            self._apply_step_failure(prepared_step, exc, match_label=match_label)
            raise

    def _prepare_step(
        self,
        *,
        round_run_id,
        match_id,
        step_name: str,
        input_json: dict[str, Any],
        settings: ModelSettings,
        schema_type,
    ) -> PreparedStep:
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
        return PreparedStep(
            step_name=step_name,
            step_row=step,
            rendered_prompt=rendered_prompt,
            input_json=input_json,
            settings=settings,
            schema_type=schema_type,
        )

    @staticmethod
    def _run_step_worker(prepared_step: PreparedStep) -> StepExecutionResult:
        adapter = build_adapter(prepared_step.settings.provider)
        adapter_result = adapter.run_structured(
            step_name=prepared_step.step_name,
            prompt=prepared_step.rendered_prompt,
            input_json=prepared_step.input_json,
            model_name=prepared_step.settings.model,
            temperature=prepared_step.settings.temperature,
            reasoning_effort=prepared_step.settings.reasoning_effort,
            output_schema=to_openai_output_schema(prepared_step.schema_type.model_json_schema()),
        )
        validated = prepared_step.schema_type.model_validate(adapter_result.output_json)
        return StepExecutionResult(
            output_json=validated.model_dump(mode="json"),
            tokens_input=adapter_result.tokens_input,
            tokens_output=adapter_result.tokens_output,
            provider_meta=adapter_result.provider_meta,
        )

    def _apply_step_success(
        self,
        prepared_step: PreparedStep,
        result: StepExecutionResult,
        *,
        match_label: str | None = None,
    ) -> dict[str, Any]:
        prepared_step.step_row.output_json = result.output_json
        prepared_step.step_row.tokens_input = result.tokens_input
        prepared_step.step_row.tokens_output = result.tokens_output
        prepared_step.step_row.provider_meta = result.provider_meta
        prepared_step.step_row.status = "completed"
        prepared_step.step_row.completed_at = utcnow()
        self._record_validation(
            round_run_id=prepared_step.step_row.round_run_id,
            match_id=prepared_step.step_row.match_id,
            component_name=prepared_step.step_name,
            validation_status="passed",
        )
        self.session.flush()
        self._emit_progress(match_label, f"completed {prepared_step.step_name}")
        return prepared_step.step_row.output_json

    def _apply_step_failure(
        self,
        prepared_step: PreparedStep,
        exc: Exception,
        *,
        match_label: str | None = None,
    ) -> None:
        prepared_step.step_row.status = "failed"
        prepared_step.step_row.error_message = str(exc)
        prepared_step.step_row.completed_at = utcnow()
        self._record_validation(
            round_run_id=prepared_step.step_row.round_run_id,
            match_id=prepared_step.step_row.match_id,
            component_name=prepared_step.step_name,
            validation_status="failed",
            errors=[{"message": str(exc)}],
        )
        self.session.flush()
        self._emit_progress(match_label, f"failed {prepared_step.step_name}: {exc}")

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
                dossier_json=json.dumps(
                    input_json.get("dossier"),
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
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

    def _emit_progress(self, match_label: str | None, message: str) -> None:
        if self.progress_callback is None:
            return
        prefix = f"{match_label}: " if match_label else ""
        self.progress_callback(f"{prefix}{message}")
