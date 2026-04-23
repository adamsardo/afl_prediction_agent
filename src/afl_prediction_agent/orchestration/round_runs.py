from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from afl_prediction_agent.agents.codex_app_server import get_codex_app_server_client
from afl_prediction_agent.agents.runner import AgentPipelineRunner
from afl_prediction_agent.configuration import ensure_run_config_seeded
from afl_prediction_agent.contracts import (
    MatchRunDetailResponse,
    RoundRunSummaryResponse,
    RunConfigFile,
    RunDetailResponse,
)
from afl_prediction_agent.core.db.base import utcnow
from afl_prediction_agent.dossiers.builder import DossierBuilder
from afl_prediction_agent.features.builder import FeatureBuilder
from afl_prediction_agent.models.baseline import DeterministicBaselineService
from afl_prediction_agent.sources.service import RoundSourceSyncService
from afl_prediction_agent.storage.context import LoadedMatchContext, load_match_context
from afl_prediction_agent.storage.models import (
    AgentStep,
    AuditEvent,
    BaselineModelRun,
    BaselinePrediction,
    FeatureSet,
    FinalAgentVerdict,
    Match,
    MatchDossier,
    MatchEvaluation,
    Round,
    RoundRun,
    RunConfig,
    SeasonEvaluationSummary,
    ValidationLog,
)
from afl_prediction_agent.validation.rules import validate_final_response


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(value, 6)))


def _uuid(value):
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _safe_float(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(slots=True)
class TrackPrediction:
    track_name: str
    match_id: uuid.UUID
    predicted_winner_team_id: uuid.UUID | None
    home_win_probability: float | None
    predicted_margin: float | None
    confidence_score: float | None
    winner_correct: bool | None
    margin_error: float | None
    brier: float | None
    log_loss: float | None


class RoundRunService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def run_round(
        self,
        *,
        round_id,
        config_name: str,
        lock_timestamp: datetime | None = None,
        notes: str | None = None,
        fetch_sources: bool = False,
    ) -> RoundRun:
        round_id = _uuid(round_id)
        round_obj = self.session.get(Round, round_id)
        if round_obj is None:
            raise ValueError(f"Round {round_id} not found")
        run_config = ensure_run_config_seeded(self.session, config_name)
        config = RunConfigFile.model_validate(run_config.config)
        lock_timestamp = lock_timestamp or utcnow()
        round_run = RoundRun(
            season_id=round_obj.season_id,
            round_id=round_obj.id,
            run_config_id=run_config.id,
            lock_timestamp=lock_timestamp,
            status="running",
            notes=notes,
        )
        self.session.add(round_run)
        self.session.flush()

        try:
            self._run_provider_preflight(round_run=round_run, config=config)
            if fetch_sources:
                self._prefetch_round_sources(round_run=round_run)

            winner_model_run = BaselineModelRun(
                round_run_id=round_run.id,
                model_type="winner",
                model_version=config.winner_model_version,
                feature_version=config.feature_version,
                training_window="2012_onward_weighted_recent_5",
                config={"config_name": config.config_name},
            )
            margin_model_run = BaselineModelRun(
                round_run_id=round_run.id,
                model_type="margin",
                model_version=config.margin_model_version,
                feature_version=config.feature_version,
                training_window="2012_onward_weighted_recent_5",
                config={"config_name": config.config_name},
            )
            self.session.add_all([winner_model_run, margin_model_run])
            self.session.flush()

            feature_builder = FeatureBuilder(config.feature_version)
            baseline_service = DeterministicBaselineService(
                winner_model_version=config.winner_model_version,
                margin_model_version=config.margin_model_version,
            )
            dossier_builder = DossierBuilder()
            agent_runner = AgentPipelineRunner(self.session, config.prompt_set_version)

            total_matches = self.session.scalar(
                select(func.count()).select_from(Match).where(Match.round_id == round_run.round_id)
            ) or 0
            eligible_matches = self._eligible_matches(round_run=round_run)
            if not eligible_matches:
                raise ValueError("No eligible matches were available for this round run")

            run_had_warnings = len(eligible_matches) != total_matches
            for match, context in eligible_matches:
                try:
                    feature_result = feature_builder.build_for_match(self.session, context)
                    feature_set = FeatureSet(
                        round_run_id=round_run.id,
                        match_id=match.id,
                        feature_version=config.feature_version,
                        input_hash=feature_result.input_hash,
                        features=feature_result.features,
                    )
                    self.session.add(feature_set)
                    self.session.flush()

                    baseline_result = baseline_service.predict(feature_result.features)
                    predicted_winner_team_id = (
                        match.home_team_id
                        if baseline_result.home_win_probability >= baseline_result.away_win_probability
                        else match.away_team_id
                    )
                    baseline_prediction = BaselinePrediction(
                        round_run_id=round_run.id,
                        match_id=match.id,
                        winner_model_run_id=winner_model_run.id,
                        margin_model_run_id=margin_model_run.id,
                        predicted_winner_team_id=predicted_winner_team_id,
                        home_win_probability=_decimal(baseline_result.home_win_probability),
                        away_win_probability=_decimal(baseline_result.away_win_probability),
                        predicted_margin=Decimal(str(baseline_result.predicted_margin)),
                        confidence_reference=_decimal(baseline_result.confidence_reference),
                        top_drivers=baseline_result.top_drivers,
                    )
                    self.session.add(baseline_prediction)
                    self.session.flush()

                    dossier, dossier_hash = dossier_builder.build(
                        context=context,
                        feature_result=feature_result,
                        baseline_result=baseline_result,
                        feature_set_id=feature_set.id,
                        baseline_prediction_id=baseline_prediction.id,
                        benchmark_prediction_id=context.benchmark_prediction.id
                        if context.benchmark_prediction
                        else None,
                    )
                    dossier_row = MatchDossier(
                        round_run_id=round_run.id,
                        match_id=match.id,
                        dossier_version=dossier_builder.dossier_version,
                        input_hash=dossier_hash,
                        dossier=dossier.model_dump(mode="json"),
                    )
                    self.session.add(dossier_row)
                    self.session.flush()

                    agent_result = agent_runner.run_for_match(
                        round_run_id=round_run.id,
                        match_id=match.id,
                        dossier=dossier,
                        config=config,
                    )
                    if agent_result.final_response is None or agent_result.final_step_id is None:
                        run_had_warnings = True
                        self._mark_final_verdict_unavailable(
                            round_run_id=round_run.id,
                            match_id=match.id,
                            reason="final_agent_failed",
                            payload={"failed_steps": agent_result.failed_steps},
                        )
                        continue

                    validation = validate_final_response(dossier, agent_result.final_response)
                    self.session.add(
                        ValidationLog(
                            round_run_id=round_run.id,
                            match_id=match.id,
                            component_name="final_verdict_consistency",
                            validation_status=validation.status,
                            errors=validation.errors + validation.warnings,
                        )
                    )
                    if validation.status != "passed":
                        run_had_warnings = True
                        self._mark_final_verdict_unavailable(
                            round_run_id=round_run.id,
                            match_id=match.id,
                            reason="final_verdict_consistency_failed",
                            payload={
                                "errors": validation.errors,
                                "warnings": validation.warnings,
                            },
                        )
                        continue

                    self.session.add(
                        FinalAgentVerdict(
                            round_run_id=round_run.id,
                            match_id=match.id,
                            final_agent_step_id=agent_result.final_step_id,
                            predicted_winner_team_id=agent_result.final_response.predicted_winner_team_id,
                            home_win_probability=_decimal(agent_result.final_response.home_win_probability),
                            away_win_probability=_decimal(agent_result.final_response.away_win_probability),
                            predicted_margin=Decimal(str(agent_result.final_response.predicted_margin)),
                            confidence_score=Decimal(str(agent_result.final_response.confidence_score)),
                            top_drivers=[
                                driver.model_dump(mode="json")
                                for driver in agent_result.final_response.top_drivers
                            ],
                            uncertainty_note=agent_result.final_response.uncertainty_note,
                            rationale_summary=agent_result.final_response.rationale_summary,
                            validation_status=validation.status,
                            correction_pass_count=agent_result.correction_pass_count,
                        )
                    )
                    if agent_result.failed_steps:
                        run_had_warnings = True
                except Exception as exc:
                    run_had_warnings = True
                    self.session.add(
                        ValidationLog(
                            round_run_id=round_run.id,
                            match_id=match.id,
                            component_name="match_pipeline",
                            validation_status="failed",
                            errors=[{"message": str(exc)}],
                        )
                    )
                    self._create_audit_event(
                        round_run_id=round_run.id,
                        match_id=match.id,
                        event_type="match_pipeline_failed",
                        payload={"error": str(exc)},
                    )
            round_run.status = "completed_with_warnings" if run_had_warnings else "completed"
            round_run.completed_at = utcnow()
            self.session.flush()
            return round_run
        except Exception:
            round_run.status = "failed"
            round_run.completed_at = utcnow()
            self.session.flush()
            raise

    def _prefetch_round_sources(self, *, round_run: RoundRun) -> None:
        sync_service = RoundSourceSyncService(self.session)
        sync_service.snapshot_round(round_id=round_run.round_id, round_run_id=round_run.id)

    def _eligible_matches(self, *, round_run: RoundRun) -> list[tuple[Match, LoadedMatchContext]]:
        matches = self.session.scalars(
            select(Match)
            .where(Match.round_id == round_run.round_id)
            .order_by(Match.scheduled_at.asc())
        ).all()
        eligible: list[tuple[Match, LoadedMatchContext]] = []
        for match in matches:
            context = load_match_context(
                self.session,
                match=match,
                lock_timestamp=round_run.lock_timestamp,
                round_run_id=round_run.id,
            )
            if context.home_lineup is None or context.away_lineup is None:
                missing_side = []
                if context.home_lineup is None:
                    missing_side.append("home")
                if context.away_lineup is None:
                    missing_side.append("away")
                self._create_audit_event(
                    round_run_id=round_run.id,
                    match_id=match.id,
                    event_type="match_excluded",
                    payload={
                        "reason": "missing_official_lineup",
                        "missing_sides": missing_side,
                    },
                )
                continue
            eligible.append((match, context))
        return eligible

    def _mark_final_verdict_unavailable(
        self,
        *,
        round_run_id,
        match_id,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._create_audit_event(
            round_run_id=round_run_id,
            match_id=match_id,
            event_type="final_agent_verdict_unavailable",
            payload={"reason": reason, **(payload or {})},
        )

    def _create_audit_event(
        self,
        *,
        round_run_id,
        match_id,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.session.add(
            AuditEvent(
                round_run_id=round_run_id,
                match_id=match_id,
                event_type=event_type,
                payload=payload,
            )
        )

    def list_round_runs(self, round_id) -> list[RoundRunSummaryResponse]:
        runs = self.session.scalars(
            select(RoundRun).where(RoundRun.round_id == _uuid(round_id)).order_by(RoundRun.created_at.desc())
        ).all()
        return [
            RoundRunSummaryResponse(
                run_id=row.id,
                round_id=row.round_id,
                status=row.status,
                created_at=row.created_at,
                completed_at=row.completed_at,
            )
            for row in runs
        ]

    def get_run_detail(self, run_id) -> RunDetailResponse:
        run_id = _uuid(run_id)
        run = self.session.get(RoundRun, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        match_count = self.session.scalar(
            select(func.count()).select_from(Match).where(Match.round_id == run.round_id)
        ) or 0
        skipped_match_count = self.session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(
                AuditEvent.round_run_id == run.id,
                AuditEvent.event_type == "match_excluded",
            )
        ) or 0
        baseline_only_match_count = self.session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(
                AuditEvent.round_run_id == run.id,
                AuditEvent.event_type == "final_agent_verdict_unavailable",
            )
        ) or 0
        processed_match_count = self.session.scalar(
            select(func.count())
            .select_from(BaselinePrediction)
            .where(BaselinePrediction.round_run_id == run.id)
        ) or 0
        verdict_count = self.session.scalar(
            select(func.count())
            .select_from(FinalAgentVerdict)
            .where(FinalAgentVerdict.round_run_id == run.id)
        ) or 0
        return RunDetailResponse(
            run_id=run.id,
            round_id=run.round_id,
            season_id=run.season_id,
            status=run.status,
            lock_timestamp=run.lock_timestamp,
            match_count=match_count,
            eligible_match_count=max(match_count - skipped_match_count, 0),
            skipped_match_count=skipped_match_count,
            processed_match_count=processed_match_count,
            baseline_only_match_count=baseline_only_match_count,
            verdict_count=verdict_count,
            created_at=run.created_at,
            completed_at=run.completed_at,
        )

    def get_match_run_detail(self, run_id, match_id) -> MatchRunDetailResponse:
        run_id = _uuid(run_id)
        match_id = _uuid(match_id)
        run = self.session.get(RoundRun, run_id)
        match = self.session.get(Match, match_id)
        if run is None or match is None:
            raise ValueError("Run or match not found")
        run_config = self.session.get(RunConfig, run.run_config_id)
        context = load_match_context(
            self.session,
            match=match,
            lock_timestamp=run.lock_timestamp,
            round_run_id=run.id,
        )
        dossier = self.session.scalar(
            select(MatchDossier).where(
                MatchDossier.round_run_id == run_id,
                MatchDossier.match_id == match_id,
            )
        )
        feature_set = self.session.scalar(
            select(FeatureSet).where(
                FeatureSet.round_run_id == run_id,
                FeatureSet.match_id == match_id,
            )
        )
        baseline = self.session.scalar(
            select(BaselinePrediction).where(
                BaselinePrediction.round_run_id == run_id,
                BaselinePrediction.match_id == match_id,
            )
        )
        verdict = self.session.scalar(
            select(FinalAgentVerdict).where(
                FinalAgentVerdict.round_run_id == run_id,
                FinalAgentVerdict.match_id == match_id,
            )
        )
        steps = self.session.scalars(
            select(AgentStep)
            .where(AgentStep.round_run_id == run_id, AgentStep.match_id == match_id)
            .order_by(AgentStep.started_at.asc())
        ).all()
        skip_event = self.session.scalar(
            select(AuditEvent).where(
                AuditEvent.round_run_id == run_id,
                AuditEvent.match_id == match_id,
                AuditEvent.event_type == "match_excluded",
            )
        )
        final_unavailable = self.session.scalar(
            select(AuditEvent).where(
                AuditEvent.round_run_id == run_id,
                AuditEvent.match_id == match_id,
                AuditEvent.event_type == "final_agent_verdict_unavailable",
            )
        )
        analyst_summaries = {
            step.step_name: (step.output_json or {}).get("summary", "")
            for step in steps
            if step.step_name.endswith("analyst_v1") and step.output_json
        }
        case_summaries = {
            step.step_name: (step.output_json or {}).get("case_summary", "")
            for step in steps
            if step.step_name.endswith("case_v1") and step.output_json
        }
        final_steps = [
            step
            for step in steps
            if step.step_name in {"final_decision_v1", "correction_pass_v1"} and step.status == "completed"
        ]
        match_status = "pending"
        skip_reason = None
        if skip_event is not None:
            match_status = "skipped"
            skip_reason = skip_event.payload.get("reason")
        elif verdict is not None:
            match_status = "completed"
        elif final_unavailable is not None and baseline is not None:
            match_status = "baseline_only"
            skip_reason = final_unavailable.payload.get("reason")
        elif baseline is not None:
            match_status = "failed"
        return MatchRunDetailResponse(
            run_id=run.id,
            match_id=match_id,
            season_year=context.season.season_year,
            round_number=context.round.round_number,
            home_team_name=context.home_team.name,
            away_team_name=context.away_team.name,
            venue_name=context.venue.name,
            prediction_lock_timestamp=run.lock_timestamp,
            match_status=match_status,
            skip_reason=skip_reason,
            analyst_summaries=analyst_summaries,
            case_summaries=case_summaries,
            bookmaker_snapshot={
                "home_median_price": _safe_float(context.odds_snapshot.home_median_price) if context.odds_snapshot else None,
                "away_median_price": _safe_float(context.odds_snapshot.away_median_price) if context.odds_snapshot else None,
                "home_implied_probability": _safe_float(context.odds_snapshot.home_implied_probability) if context.odds_snapshot else None,
                "away_implied_probability": _safe_float(context.odds_snapshot.away_implied_probability) if context.odds_snapshot else None,
                "bookmaker_count": context.odds_snapshot.bookmaker_count if context.odds_snapshot else 0,
            }
            if context.odds_snapshot
            else None,
            squiggle_snapshot={
                "source_name": context.benchmark_prediction.source_name,
                "predicted_winner_team_id": str(context.benchmark_prediction.predicted_winner_team_id)
                if context.benchmark_prediction.predicted_winner_team_id
                else None,
                "home_win_probability": _safe_float(context.benchmark_prediction.home_win_probability),
                "away_win_probability": _safe_float(context.benchmark_prediction.away_win_probability),
                "predicted_margin": _safe_float(context.benchmark_prediction.predicted_margin),
            }
            if context.benchmark_prediction
            else None,
            final_decision_model_version=final_steps[-1].model_name if final_steps else None,
            prompt_version_set=run_config.prompt_set_version if run_config else None,
            feature_version=feature_set.feature_version if feature_set else None,
            dossier=dossier.dossier if dossier else None,
            baseline_prediction={
                "predicted_winner_team_id": str(baseline.predicted_winner_team_id),
                "home_win_probability": float(baseline.home_win_probability),
                "away_win_probability": float(baseline.away_win_probability),
                "predicted_margin": float(baseline.predicted_margin),
                "top_drivers": baseline.top_drivers,
            }
            if baseline
            else None,
            final_verdict={
                "predicted_winner_team_id": str(verdict.predicted_winner_team_id),
                "home_win_probability": float(verdict.home_win_probability),
                "away_win_probability": float(verdict.away_win_probability),
                "predicted_margin": float(verdict.predicted_margin),
                "confidence_score": float(verdict.confidence_score),
                "top_drivers": verdict.top_drivers,
                "uncertainty_note": verdict.uncertainty_note,
                "rationale_summary": verdict.rationale_summary,
            }
            if verdict
            else None,
            agent_steps=[
                {
                    "step_name": step.step_name,
                    "model_provider": step.model_provider,
                    "model_name": step.model_name,
                    "temperature": float(step.temperature) if step.temperature is not None else None,
                    "reasoning_effort": step.reasoning_effort,
                    "status": step.status,
                    "input_json": step.input_json,
                    "output_json": step.output_json,
                    "provider_meta": step.provider_meta,
                    "error_message": step.error_message,
                }
                for step in steps
            ],
        )

    def _run_provider_preflight(self, *, round_run: RoundRun, config: RunConfigFile) -> None:
        providers = {
            config.analyst_model.provider,
            config.case_model.provider,
            config.final_model.provider,
        }
        if "codex_app_server" not in providers:
            return

        client = get_codex_app_server_client()
        try:
            snapshot = client.preflight_auth()
            payload = {
                "provider": "codex_app_server",
                "auth_mode": snapshot.auth_mode,
                "email": snapshot.email,
                "account_plan_type": snapshot.account_plan_type,
                "effective_plan_type": snapshot.effective_plan_type,
                "requires_openai_auth": snapshot.requires_openai_auth,
                "supported_plan": snapshot.supported_plan,
                "rate_limits": snapshot.rate_limits,
            }
            self.session.add(
                AuditEvent(
                    round_run_id=round_run.id,
                    match_id=None,
                    event_type="codex_app_server_preflight",
                    payload=payload,
                )
            )
            self.session.flush()
        except Exception as exc:
            self.session.add(
                AuditEvent(
                    round_run_id=round_run.id,
                    match_id=None,
                    event_type="codex_app_server_preflight_failed",
                    payload={
                        "provider": "codex_app_server",
                        "error": str(exc),
                    },
                )
            )
            self.session.flush()
            raise


class EvaluationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def evaluate_run(self, run_id) -> list[MatchEvaluation]:
        run_id = _uuid(run_id)
        run = self.session.get(RoundRun, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        existing = self.session.scalars(
            select(MatchEvaluation).where(MatchEvaluation.round_run_id == run_id)
        ).all()
        if existing:
            self._store_season_summary(run, existing)
            return existing

        matches = self.session.scalars(
            select(Match)
            .where(
                Match.round_id == run.round_id,
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
            .order_by(Match.scheduled_at.asc())
        ).all()
        evaluations: list[MatchEvaluation] = []
        for match in matches:
            verdict = self.session.scalar(
                select(FinalAgentVerdict).where(
                    FinalAgentVerdict.round_run_id == run_id,
                    FinalAgentVerdict.match_id == match.id,
                )
            )
            baseline = self.session.scalar(
                select(BaselinePrediction).where(
                    BaselinePrediction.round_run_id == run_id,
                    BaselinePrediction.match_id == match.id,
                )
            )
            if verdict is None or baseline is None or match.winning_team_id is None or match.actual_margin is None:
                continue
            actual_home_win = 1.0 if match.winning_team_id == match.home_team_id else 0.0
            agent_home = float(verdict.home_win_probability)
            baseline_home = float(baseline.home_win_probability)
            evaluation = MatchEvaluation(
                round_run_id=run_id,
                match_id=match.id,
                actual_winner_team_id=match.winning_team_id,
                actual_margin=Decimal(str(match.actual_margin)),
                agent_winner_correct=verdict.predicted_winner_team_id == match.winning_team_id,
                baseline_winner_correct=baseline.predicted_winner_team_id == match.winning_team_id,
                agent_margin_error=Decimal(str(abs(float(verdict.predicted_margin) - match.actual_margin))),
                baseline_margin_error=Decimal(str(abs(float(baseline.predicted_margin) - match.actual_margin))),
                agent_brier=Decimal(str((agent_home - actual_home_win) ** 2)),
                baseline_brier=Decimal(str((baseline_home - actual_home_win) ** 2)),
                agent_log_loss=Decimal(str(self._log_loss(agent_home, actual_home_win))),
                baseline_log_loss=Decimal(str(self._log_loss(baseline_home, actual_home_win))),
            )
            self.session.add(evaluation)
            evaluations.append(evaluation)
        self.session.flush()
        self._store_season_summary(run, evaluations)
        return evaluations

    def _store_season_summary(self, run: RoundRun, evaluations: list[MatchEvaluation]) -> None:
        matches = self.session.scalars(
            select(Match)
            .where(
                Match.round_id == run.round_id,
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
            .order_by(Match.scheduled_at.asc())
        ).all()
        track_predictions: dict[str, list[TrackPrediction]] = {}
        agent_confidence_rows: list[dict[str, Any]] = []
        for match in matches:
            actual_home_win = 1.0 if match.winning_team_id == match.home_team_id else 0.0
            context = load_match_context(
                self.session,
                match=match,
                lock_timestamp=run.lock_timestamp,
                round_run_id=run.id,
            )
            feature_set = self.session.scalar(
                select(FeatureSet).where(
                    FeatureSet.round_run_id == run.id,
                    FeatureSet.match_id == match.id,
                )
            )
            verdict = self.session.scalar(
                select(FinalAgentVerdict).where(
                    FinalAgentVerdict.round_run_id == run.id,
                    FinalAgentVerdict.match_id == match.id,
                )
            )
            baseline = self.session.scalar(
                select(BaselinePrediction).where(
                    BaselinePrediction.round_run_id == run.id,
                    BaselinePrediction.match_id == match.id,
                )
            )
            for track in self._build_track_predictions(
                match=match,
                actual_home_win=actual_home_win,
                context=context,
                feature_set=feature_set,
                baseline=baseline,
                verdict=verdict,
            ):
                track_predictions.setdefault(track.track_name, []).append(track)
            if verdict is not None:
                agent_confidence_rows.append(
                    {
                        "confidence_score": float(verdict.confidence_score),
                        "winner_correct": verdict.predicted_winner_team_id == match.winning_team_id,
                    }
                )

        if not track_predictions:
            return

        summary = {
            "run_id": str(run.id),
            "round_id": str(run.round_id),
            "match_count": len(matches),
            "tracks": {
                name: self._track_summary(rows)
                for name, rows in track_predictions.items()
            },
            "calibration_buckets": {
                name: self._calibration_buckets(rows)
                for name, rows in track_predictions.items()
                if any(row.home_win_probability is not None for row in rows)
            },
            "agent_confidence_bands": self._confidence_bands(agent_confidence_rows),
            "agreement_analysis": self._agreement_analysis(track_predictions),
        }
        self.session.add(
            SeasonEvaluationSummary(
                season_id=run.season_id,
                run_config_id=run.run_config_id,
                summary_type=f"run_evaluation::{run.id}",
                summary=summary,
            )
        )
        self.session.flush()

    def _build_track_predictions(
        self,
        *,
        match: Match,
        actual_home_win: float,
        context: LoadedMatchContext,
        feature_set: FeatureSet | None,
        baseline: BaselinePrediction | None,
        verdict: FinalAgentVerdict | None,
    ) -> list[TrackPrediction]:
        tracks: list[TrackPrediction] = []
        if verdict is not None:
            tracks.append(
                self._track_prediction(
                    track_name="agent",
                    match=match,
                    actual_home_win=actual_home_win,
                    predicted_winner_team_id=verdict.predicted_winner_team_id,
                    home_win_probability=float(verdict.home_win_probability),
                    predicted_margin=float(verdict.predicted_margin),
                    confidence_score=float(verdict.confidence_score),
                )
            )
        if baseline is not None:
            tracks.append(
                self._track_prediction(
                    track_name="deterministic_baseline",
                    match=match,
                    actual_home_win=actual_home_win,
                    predicted_winner_team_id=baseline.predicted_winner_team_id,
                    home_win_probability=float(baseline.home_win_probability),
                    predicted_margin=float(baseline.predicted_margin),
                    confidence_score=float(baseline.confidence_reference) * 100.0 if baseline.confidence_reference is not None else None,
                )
            )
        if context.odds_snapshot and context.odds_snapshot.home_implied_probability is not None:
            market_home = float(context.odds_snapshot.home_implied_probability)
            market_away = float(context.odds_snapshot.away_implied_probability or (1.0 - market_home))
            tracks.append(
                self._track_prediction(
                    track_name="bookmaker_favourite",
                    match=match,
                    actual_home_win=actual_home_win,
                    predicted_winner_team_id=match.home_team_id if market_home >= market_away else match.away_team_id,
                    home_win_probability=market_home,
                    predicted_margin=None,
                    confidence_score=max(market_home, market_away) * 100.0,
                )
            )
        if context.benchmark_prediction and context.benchmark_prediction.home_win_probability is not None:
            squiggle_home = float(context.benchmark_prediction.home_win_probability)
            squiggle_away = float(context.benchmark_prediction.away_win_probability or (1.0 - squiggle_home))
            tracks.append(
                self._track_prediction(
                    track_name="squiggle",
                    match=match,
                    actual_home_win=actual_home_win,
                    predicted_winner_team_id=(
                        context.benchmark_prediction.predicted_winner_team_id
                        or (match.home_team_id if squiggle_home >= squiggle_away else match.away_team_id)
                    ),
                    home_win_probability=squiggle_home,
                    predicted_margin=_safe_float(context.benchmark_prediction.predicted_margin),
                    confidence_score=max(squiggle_home, squiggle_away) * 100.0,
                )
            )
        tracks.append(
            self._track_prediction(
                track_name="naive_home",
                match=match,
                actual_home_win=actual_home_win,
                predicted_winner_team_id=match.home_team_id,
                home_win_probability=0.5,
                predicted_margin=None,
                confidence_score=50.0,
            )
        )
        if feature_set is not None:
            home_recent = float(feature_set.features.get("home_recent_win_rate", 0.5))
            away_recent = float(feature_set.features.get("away_recent_win_rate", 0.5))
            diff = max(min((home_recent - away_recent) * 0.35, 0.2), -0.2)
            recent_form_home = min(max(0.5 + diff, 0.01), 0.99)
            tracks.append(
                self._track_prediction(
                    track_name="recent_form",
                    match=match,
                    actual_home_win=actual_home_win,
                    predicted_winner_team_id=match.home_team_id if recent_form_home >= 0.5 else match.away_team_id,
                    home_win_probability=recent_form_home,
                    predicted_margin=None,
                    confidence_score=max(recent_form_home, 1.0 - recent_form_home) * 100.0,
                )
            )
        return tracks

    def _track_prediction(
        self,
        *,
        track_name: str,
        match: Match,
        actual_home_win: float,
        predicted_winner_team_id,
        home_win_probability: float | None,
        predicted_margin: float | None,
        confidence_score: float | None,
    ) -> TrackPrediction:
        winner_correct = None
        brier = None
        log_loss = None
        margin_error = None
        if predicted_winner_team_id is not None and match.winning_team_id is not None:
            winner_correct = predicted_winner_team_id == match.winning_team_id
        if home_win_probability is not None:
            brier = (home_win_probability - actual_home_win) ** 2
            log_loss = self._log_loss(home_win_probability, actual_home_win)
        if predicted_margin is not None and match.actual_margin is not None:
            margin_error = abs(predicted_margin - match.actual_margin)
        return TrackPrediction(
            track_name=track_name,
            match_id=match.id,
            predicted_winner_team_id=predicted_winner_team_id,
            home_win_probability=home_win_probability,
            predicted_margin=predicted_margin,
            confidence_score=confidence_score,
            winner_correct=winner_correct,
            margin_error=margin_error,
            brier=brier,
            log_loss=log_loss,
        )

    def _track_summary(self, rows: list[TrackPrediction]) -> dict[str, Any]:
        winner_rows = [row for row in rows if row.winner_correct is not None]
        margin_rows = [row.margin_error for row in rows if row.margin_error is not None]
        brier_rows = [row.brier for row in rows if row.brier is not None]
        log_loss_rows = [row.log_loss for row in rows if row.log_loss is not None]
        return {
            "match_count": len(rows),
            "winner_accuracy": (
                sum(1 for row in winner_rows if row.winner_correct) / len(winner_rows)
                if winner_rows
                else None
            ),
            "mean_absolute_error": sum(margin_rows) / len(margin_rows) if margin_rows else None,
            "root_mean_squared_error": (
                math.sqrt(sum(error ** 2 for error in margin_rows) / len(margin_rows))
                if margin_rows
                else None
            ),
            "mean_brier": sum(brier_rows) / len(brier_rows) if brier_rows else None,
            "mean_log_loss": sum(log_loss_rows) / len(log_loss_rows) if log_loss_rows else None,
        }

    def _calibration_buckets(self, rows: list[TrackPrediction]) -> list[dict[str, Any]]:
        buckets: dict[tuple[float, float], list[TrackPrediction]] = {}
        for row in rows:
            if row.home_win_probability is None or row.winner_correct is None:
                continue
            confidence = max(row.home_win_probability, 1.0 - row.home_win_probability)
            lower = math.floor(confidence * 10) / 10
            upper = min(lower + 0.1, 1.0)
            buckets.setdefault((lower, upper), []).append(row)
        output: list[dict[str, Any]] = []
        for (lower, upper), bucket_rows in sorted(buckets.items()):
            output.append(
                {
                    "bucket": f"{lower:.1f}-{upper:.1f}",
                    "count": len(bucket_rows),
                    "avg_confidence": sum(
                        max(row.home_win_probability or 0.0, 1.0 - (row.home_win_probability or 0.0))
                        for row in bucket_rows
                    )
                    / len(bucket_rows),
                    "empirical_accuracy": sum(1 for row in bucket_rows if row.winner_correct) / len(bucket_rows),
                }
            )
        return output

    def _confidence_bands(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            score = row["confidence_score"]
            lower = int(score // 10 * 10)
            upper = min(lower + 10, 100)
            key = f"{lower}-{upper}"
            buckets.setdefault(key, []).append(row)
        output: list[dict[str, Any]] = []
        for bucket, bucket_rows in sorted(
            buckets.items(),
            key=lambda item: int(item[0].split("-")[0]),
        ):
            output.append(
                {
                    "band": bucket,
                    "count": len(bucket_rows),
                    "winner_accuracy": sum(1 for row in bucket_rows if row["winner_correct"]) / len(bucket_rows),
                }
            )
        return output

    def _agreement_analysis(self, track_predictions: dict[str, list[TrackPrediction]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        agent_rows = {str(row.match_id): row for row in track_predictions.get("agent", [])}
        for comparison_name in ["deterministic_baseline", "bookmaker_favourite", "squiggle"]:
            comparison_rows = {
                str(row.match_id): row for row in track_predictions.get(comparison_name, [])
            }
            shared_match_ids = sorted(set(agent_rows).intersection(comparison_rows))
            if not shared_match_ids:
                continue
            agreements = [
                agent_rows[match_id].predicted_winner_team_id
                == comparison_rows[match_id].predicted_winner_team_id
                for match_id in shared_match_ids
            ]
            agree_rows = [
                agent_rows[match_id]
                for match_id, agreed in zip(shared_match_ids, agreements, strict=True)
                if agreed and agent_rows[match_id].winner_correct is not None
            ]
            disagree_rows = [
                agent_rows[match_id]
                for match_id, agreed in zip(shared_match_ids, agreements, strict=True)
                if not agreed and agent_rows[match_id].winner_correct is not None
            ]
            output[comparison_name] = {
                "match_count": len(shared_match_ids),
                "agreement_rate": sum(1 for agreed in agreements if agreed) / len(shared_match_ids),
                "accuracy_when_agree": (
                    sum(1 for row in agree_rows if row.winner_correct) / len(agree_rows)
                    if agree_rows
                    else None
                ),
                "accuracy_when_disagree": (
                    sum(1 for row in disagree_rows if row.winner_correct) / len(disagree_rows)
                    if disagree_rows
                    else None
                ),
            }
        return output

    def _log_loss(self, probability: float, actual_home_win: float) -> float:
        clipped = min(max(probability, 1e-6), 1 - 1e-6)
        return -(actual_home_win * math.log(clipped) + (1 - actual_home_win) * math.log(1 - clipped))

    def get_season_summaries(self, season_id) -> list[dict[str, Any]]:
        season_id = _uuid(season_id)
        rows = self.session.scalars(
            select(SeasonEvaluationSummary).where(SeasonEvaluationSummary.season_id == season_id)
        ).all()
        return [
            {
                "id": str(row.id),
                "season_id": str(row.season_id),
                "run_config_id": str(row.run_config_id),
                "summary_type": row.summary_type,
                "summary": row.summary,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]


class ReplayService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.round_service = RoundRunService(session)
        self.evaluation_service = EvaluationService(session)

    def replay_round(
        self,
        *,
        round_id,
        config_name: str,
        lock_timestamp: datetime | None = None,
    ) -> RoundRun:
        round_id = _uuid(round_id)
        inferred_lock, approximate, strategy = self._infer_lock_timestamp(round_id, lock_timestamp)
        notes = (
            f"historical_replay strategy={strategy} approximate={str(approximate).lower()}"
        )
        run = self.round_service.run_round(
            round_id=round_id,
            config_name=config_name,
            lock_timestamp=inferred_lock,
            notes=notes,
            fetch_sources=False,
        )
        self.session.add(
            AuditEvent(
                round_run_id=run.id,
                match_id=None,
                event_type="historical_replay",
                payload={
                    "approximate": approximate,
                    "lock_strategy": strategy,
                    "lock_timestamp": inferred_lock.isoformat(),
                },
            )
        )
        self.session.flush()
        return run

    def replay_season(
        self,
        *,
        season_id,
        config_name: str,
    ) -> list[RoundRun]:
        season_id = _uuid(season_id)
        rounds = self.session.scalars(
            select(Round)
            .where(Round.season_id == season_id)
            .order_by(Round.round_number.asc(), Round.starts_at.asc().nulls_last())
        ).all()
        runs: list[RoundRun] = []
        for round_obj in rounds:
            run = self.replay_round(round_id=round_obj.id, config_name=config_name)
            self.evaluation_service.evaluate_run(run.id)
            runs.append(run)
        return runs

    def _infer_lock_timestamp(
        self,
        round_id,
        explicit_lock_timestamp: datetime | None,
    ) -> tuple[datetime, bool, str]:
        if explicit_lock_timestamp is not None:
            return explicit_lock_timestamp, False, "explicit"
        matches = self.session.scalars(
            select(Match)
            .where(Match.round_id == round_id)
            .order_by(Match.scheduled_at.asc())
        ).all()
        if not matches:
            raise ValueError(f"Round {round_id} has no matches to replay")
        earliest_match = _as_utc_datetime(matches[0].scheduled_at)
        candidate_snapshot_times: list[datetime] = []
        for match in matches:
            context = load_match_context(
                self.session,
                match=match,
                lock_timestamp=earliest_match - timedelta(minutes=1),
                round_run_id=None,
            )
            if context.home_lineup is not None:
                candidate_snapshot_times.append(_as_utc_datetime(context.home_lineup.fetched_at))
            if context.away_lineup is not None:
                candidate_snapshot_times.append(_as_utc_datetime(context.away_lineup.fetched_at))
        if candidate_snapshot_times:
            inferred = min(
                max(candidate_snapshot_times),
                earliest_match - timedelta(minutes=1),
            )
            approximate = len(candidate_snapshot_times) < len(matches) * 2
            return inferred, approximate, "latest_available_lineup_snapshot"
        return earliest_match - timedelta(hours=24), True, "scheduled_minus_24h"
