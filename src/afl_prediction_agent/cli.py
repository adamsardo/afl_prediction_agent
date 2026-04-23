from __future__ import annotations

import typer

from afl_prediction_agent.agents.codex_app_server import get_codex_app_server_client
from afl_prediction_agent.configuration import ensure_prompt_set_seeded, ensure_run_config_seeded
from afl_prediction_agent.core.db.base import Base
from afl_prediction_agent.core.db.session import SessionLocal, engine
from afl_prediction_agent.orchestration.round_runs import EvaluationService, ReplayService, RoundRunService
from afl_prediction_agent.sources.service import RoundSourceSyncService


app = typer.Typer(help="AFL Prediction Agent CLI")
auth_app = typer.Typer(help="Authentication helpers.")
codex_auth_app = typer.Typer(help="Codex app-server auth helpers.")
ingest_app = typer.Typer(help="Source ingestion helpers.")
auth_app.add_typer(codex_auth_app, name="codex")
app.add_typer(auth_app, name="auth")
app.add_typer(ingest_app, name="ingest")


@app.command("init-db")
def init_db() -> None:
    """Create tables directly from metadata for local/dev usage."""
    Base.metadata.create_all(bind=engine)
    typer.echo("Database tables created.")


@app.command("seed-config")
def seed_config(config_name: str = "v1_agentic_default") -> None:
    with SessionLocal() as session:
        ensure_run_config_seeded(session, config_name)
        session.commit()
    typer.echo(f"Seeded run config and prompts for {config_name}.")


@app.command("seed-prompts")
def seed_prompts(prompt_set_version: str = "prompt_set_v1") -> None:
    with SessionLocal() as session:
        ensure_prompt_set_seeded(session, prompt_set_version)
        session.commit()
    typer.echo(f"Seeded prompt templates for {prompt_set_version}.")


@app.command("run-round")
def run_round(
    round_id: str,
    config_name: str = "v1_agentic_default",
    notes: str | None = None,
    fetch_sources: bool = typer.Option(
        True,
        "--fetch-sources/--no-fetch-sources",
        help="Capture source snapshots before running the round pipeline.",
    ),
) -> None:
    with SessionLocal() as session:
        service = RoundRunService(session)
        run = service.run_round(
            round_id=round_id,
            config_name=config_name,
            notes=notes,
            fetch_sources=fetch_sources,
            progress_callback=typer.echo,
        )
        session.commit()
        typer.echo(f"Completed run {run.id} for round {round_id} with status={run.status}.")


@app.command("replay-round")
def replay_round(
    round_id: str,
    config_name: str = "v1_agentic_default",
    lock_timestamp: str | None = None,
) -> None:
    with SessionLocal() as session:
        service = ReplayService(session)
        parsed_lock_timestamp = None
        if lock_timestamp:
            from datetime import datetime

            parsed_lock_timestamp = datetime.fromisoformat(lock_timestamp.replace("Z", "+00:00"))
        run = service.replay_round(
            round_id=round_id,
            config_name=config_name,
            lock_timestamp=parsed_lock_timestamp,
        )
        session.commit()
        typer.echo(f"Completed replay run {run.id} for round {round_id}.")


@app.command("replay-season")
def replay_season(
    season_id: str,
    config_name: str = "v1_agentic_default",
) -> None:
    with SessionLocal() as session:
        service = ReplayService(session)
        runs = service.replay_season(season_id=season_id, config_name=config_name)
        session.commit()
        typer.echo(f"Completed {len(runs)} replay runs for season {season_id}.")


@app.command("evaluate-run")
def evaluate_run(run_id: str) -> None:
    with SessionLocal() as session:
        service = EvaluationService(session)
        rows = service.evaluate_run(run_id)
        session.commit()
        typer.echo(f"Stored {len(rows)} match evaluations for run {run_id}.")


@app.command("list-runs")
def list_runs(round_id: str) -> None:
    with SessionLocal() as session:
        service = RoundRunService(session)
        runs = service.list_round_runs(round_id)
        for run in runs:
            typer.echo(
                f"{run.run_id} status={run.status} created_at={run.created_at.isoformat()}"
            )


@codex_auth_app.command("status")
def codex_status() -> None:
    client = get_codex_app_server_client()
    account = client.read_account(refresh_token=False)
    rate_limits = client.read_rate_limits()
    account_payload = account.get("account") or {}
    effective_plan = (
        (rate_limits.get("rateLimits") or {}).get("planType")
        or account_payload.get("planType")
    )
    typer.echo(f"auth_mode={account_payload.get('type') or 'none'}")
    typer.echo(f"email={account_payload.get('email') or 'unknown'}")
    typer.echo(f"account_plan_type={account_payload.get('planType') or 'unknown'}")
    typer.echo(f"effective_plan_type={effective_plan or 'unknown'}")


@codex_auth_app.command("login")
def codex_login(
    device_code: bool = typer.Option(
        False,
        "--device-code",
        help="Use the ChatGPT device-code login flow.",
    )
) -> None:
    if not device_code:
        raise typer.BadParameter("Only --device-code is implemented for Codex login.")
    client = get_codex_app_server_client()
    login = client.start_device_code_login()
    typer.echo(f"verification_url={login['verification_url']}")
    typer.echo(f"user_code={login['user_code']}")
    auth_snapshot = client.wait_for_device_code_login(login_id=login["login_id"])
    typer.echo(f"auth_mode={auth_snapshot.auth_mode}")
    typer.echo(f"effective_plan_type={auth_snapshot.effective_plan_type}")


@codex_auth_app.command("logout")
def codex_logout() -> None:
    client = get_codex_app_server_client()
    client.logout()
    typer.echo("Logged out of Codex app-server ChatGPT auth.")


@ingest_app.command("fixtures")
def ingest_fixtures(
    season_year: int,
    round_number: int | None = None,
    use_archive_fallback: bool = False,
) -> None:
    with SessionLocal() as session:
        summary = RoundSourceSyncService(session).ingest_fixtures(
            season_year=season_year,
            round_number=round_number,
            use_archive_fallback=use_archive_fallback,
        )
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("results")
def ingest_results(
    season_year: int,
    round_number: int | None = None,
    use_archive_fallback: bool = False,
) -> None:
    with SessionLocal() as session:
        summary = RoundSourceSyncService(session).ingest_results(
            season_year=season_year,
            round_number=round_number,
            use_archive_fallback=use_archive_fallback,
        )
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("lineups")
def ingest_lineups(round_id: str) -> None:
    with SessionLocal() as session:
        summary = RoundSourceSyncService(session).ingest_lineups(round_id=round_id)
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("stats")
def ingest_stats(
    season_year: int,
    round_number: int | None = None,
    source_track: str = typer.Option("official", help="official or archive"),
) -> None:
    with SessionLocal() as session:
        summary = RoundSourceSyncService(session).ingest_stats(
            season_year=season_year,
            round_number=round_number,
            source_track=source_track,
        )
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("injuries")
def ingest_injuries() -> None:
    with SessionLocal() as session:
        summaries = RoundSourceSyncService(session).ingest_injuries()
        session.commit()
        for summary in summaries:
            typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("weather")
def ingest_weather(round_id: str) -> None:
    with SessionLocal() as session:
        summary = RoundSourceSyncService(session).ingest_weather(round_id=round_id)
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("odds")
def ingest_odds(round_id: str, as_of: str | None = None) -> None:
    with SessionLocal() as session:
        parsed_as_of = None
        if as_of:
            from datetime import datetime

            parsed_as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        summary = RoundSourceSyncService(session).ingest_odds(round_id=round_id, as_of=parsed_as_of)
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")
        if summary.errors:
            for error in summary.errors:
                typer.echo(f"error={error}")


@ingest_app.command("benchmarks")
def ingest_benchmarks(round_id: str, run_id: str) -> None:
    with SessionLocal() as session:
        summary = RoundSourceSyncService(session).ingest_benchmarks(
            round_id=round_id,
            round_run_id=run_id,
        )
        session.commit()
        typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("snapshot-round")
def snapshot_round(round_id: str, run_id: str | None = None) -> None:
    with SessionLocal() as session:
        summaries = RoundSourceSyncService(session).snapshot_round(
            round_id=round_id,
            round_run_id=run_id,
        )
        session.commit()
        for summary in summaries:
            typer.echo(f"{summary.source_name}: created={summary.created} skipped={summary.skipped}")


@ingest_app.command("review-unresolved")
def review_unresolved() -> None:
    with SessionLocal() as session:
        review = RoundSourceSyncService(session).review_unresolved()
        for section, rows in review.items():
            typer.echo(f"[{section}]")
            if not rows:
                typer.echo("none")
                continue
            for row in rows:
                typer.echo(str(row))
