from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from afl_prediction_agent.contracts import RunRoundRequest
from afl_prediction_agent.core.db.session import get_session
from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.orchestration.round_runs import EvaluationService, RoundRunService


settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/rounds/{round_id}/runs")
def list_round_runs(round_id: str, session: Session = Depends(get_session)):
    service = RoundRunService(session)
    try:
        return service.list_round_runs(round_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}")
def get_run(run_id: str, session: Session = Depends(get_session)):
    service = RoundRunService(session)
    try:
        return service.get_run_detail(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/matches/{match_id}")
def get_match_run(run_id: str, match_id: str, session: Session = Depends(get_session)):
    service = RoundRunService(session)
    try:
        return service.get_match_run_detail(run_id, match_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/rounds/{round_id}/run")
def run_round(round_id: str, payload: RunRoundRequest, session: Session = Depends(get_session)):
    service = RoundRunService(session)
    try:
        run = service.run_round(
            round_id=round_id,
            config_name=payload.config_name,
            lock_timestamp=payload.lock_timestamp,
            notes=payload.notes,
        )
        session.commit()
        return service.get_run_detail(run.id)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/runs/{run_id}/evaluate")
def evaluate_run(run_id: str, session: Session = Depends(get_session)):
    service = EvaluationService(session)
    try:
        rows = service.evaluate_run(run_id)
        session.commit()
        return {"run_id": run_id, "match_evaluations": len(rows)}
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/seasons/{season_id}/summary")
def get_season_summary(season_id: str, session: Session = Depends(get_session)):
    service = EvaluationService(session)
    return service.get_season_summaries(season_id)
