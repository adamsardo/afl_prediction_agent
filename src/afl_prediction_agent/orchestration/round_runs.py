from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
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
from afl_prediction_agent.storage.context import load_match_context
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

        matches = self.session.scalars(
            select(Match)
            .where(Match.round_id == round_id)
            .order_by(Match.scheduled_at.asc())
        ).all()
        try:
            for match in matches:
                context = load_match_context(
                    self.session,
                    match=match,
                    lock_timestamp=round_run.lock_timestamp,
                    round_run_id=round_run.id,
                )
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
                    raise ValueError(f"Final verdict validation failed for match {match.id}")

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
            round_run.status = "completed"
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
        matches = self.session.scalars(
            select(Match).where(Match.round_id == round_run.round_id).order_by(Match.scheduled_at.asc())
        ).all()
        missing_lineups: list[str] = []
        for match in matches:
            context = load_match_context(
                self.session,
                match=match,
                lock_timestamp=round_run.lock_timestamp,
                round_run_id=round_run.id,
            )
            if context.home_lineup is None or context.away_lineup is None:
                missing_lineups.append(str(match.id))
        if missing_lineups:
            raise ValueError(f"Official lineups missing for matches: {', '.join(missing_lineups)}")

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
        )
        verdict_count = self.session.scalar(
            select(func.count()).select_from(FinalAgentVerdict).where(FinalAgentVerdict.round_run_id == run.id)
        )
        return RunDetailResponse(
            run_id=run.id,
            round_id=run.round_id,
            season_id=run.season_id,
            status=run.status,
            lock_timestamp=run.lock_timestamp,
            match_count=match_count or 0,
            verdict_count=verdict_count or 0,
            created_at=run.created_at,
            completed_at=run.completed_at,
        )

    def get_match_run_detail(self, run_id, match_id) -> MatchRunDetailResponse:
        dossier = self.session.scalar(
            select(MatchDossier).where(
                MatchDossier.round_run_id == _uuid(run_id),
                MatchDossier.match_id == _uuid(match_id),
            )
        )
        baseline = self.session.scalar(
            select(BaselinePrediction).where(
                BaselinePrediction.round_run_id == _uuid(run_id),
                BaselinePrediction.match_id == _uuid(match_id),
            )
        )
        verdict = self.session.scalar(
            select(FinalAgentVerdict).where(
                FinalAgentVerdict.round_run_id == _uuid(run_id),
                FinalAgentVerdict.match_id == _uuid(match_id),
            )
        )
        steps = self.session.scalars(
            select(AgentStep)
            .where(AgentStep.round_run_id == _uuid(run_id), AgentStep.match_id == _uuid(match_id))
            .order_by(AgentStep.started_at.asc())
        ).all()
        return MatchRunDetailResponse(
            match_id=_uuid(match_id),
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
            return existing

        matches = self.session.scalars(
            select(Match).where(Match.round_id == run.round_id, Match.home_score.is_not(None), Match.away_score.is_not(None))
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
        self._store_season_summary(run.season_id, run.run_config_id, evaluations)
        return evaluations

    def _store_season_summary(self, season_id, run_config_id, evaluations: list[MatchEvaluation]) -> None:
        if not evaluations:
            return
        summary = {
            "match_count": len(evaluations),
            "agent_winner_accuracy": sum(1 for row in evaluations if row.agent_winner_correct) / len(evaluations),
            "baseline_winner_accuracy": sum(1 for row in evaluations if row.baseline_winner_correct) / len(evaluations),
            "agent_mean_brier": sum(float(row.agent_brier or 0) for row in evaluations) / len(evaluations),
            "baseline_mean_brier": sum(float(row.baseline_brier or 0) for row in evaluations) / len(evaluations),
            "agent_mean_margin_error": sum(float(row.agent_margin_error) for row in evaluations) / len(evaluations),
            "baseline_mean_margin_error": sum(float(row.baseline_margin_error) for row in evaluations) / len(evaluations),
        }
        self.session.add(
            SeasonEvaluationSummary(
                season_id=season_id,
                run_config_id=run_config_id,
                summary_type="season_to_date",
                summary=summary,
            )
        )
        self.session.flush()

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
