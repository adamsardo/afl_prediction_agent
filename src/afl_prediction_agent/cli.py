from __future__ import annotations

import typer

from afl_prediction_agent.agents.codex_app_server import get_codex_app_server_client
from afl_prediction_agent.configuration import ensure_prompt_set_seeded, ensure_run_config_seeded
from afl_prediction_agent.core.db.base import Base
from afl_prediction_agent.core.db.session import SessionLocal, engine
from afl_prediction_agent.orchestration.round_runs import EvaluationService, RoundRunService


app = typer.Typer(help="AFL Prediction Agent CLI")
auth_app = typer.Typer(help="Authentication helpers.")
codex_auth_app = typer.Typer(help="Codex app-server auth helpers.")
auth_app.add_typer(codex_auth_app, name="codex")
app.add_typer(auth_app, name="auth")


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
) -> None:
    with SessionLocal() as session:
        service = RoundRunService(session)
        run = service.run_round(round_id=round_id, config_name=config_name, notes=notes)
        session.commit()
        typer.echo(f"Completed run {run.id} for round {round_id}.")


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
