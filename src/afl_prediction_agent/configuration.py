from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from afl_prediction_agent.contracts import RunConfigFile
from afl_prediction_agent.core.settings import get_settings
from afl_prediction_agent.storage.models import PromptTemplate, RunConfig


PROMPT_SCHEMA_VERSION = {
    "form_analyst_v1": "analyst_response_v1",
    "selection_analyst_v1": "analyst_response_v1",
    "venue_weather_analyst_v1": "analyst_response_v1",
    "market_analyst_v1": "analyst_response_v1",
    "home_case_v1": "case_response_v1",
    "away_case_v1": "case_response_v1",
    "final_decision_v1": "final_decision_v1",
    "correction_pass_v1": "final_decision_v1",
}


def load_run_config_file(config_name: str) -> RunConfigFile:
    settings = get_settings()
    path = settings.run_config_dir / f"{config_name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunConfigFile.model_validate(data)


def load_prompt_templates(prompt_set_version: str) -> dict[str, str]:
    settings = get_settings()
    prompt_dir = settings.prompts_dir / prompt_set_version
    templates: dict[str, str] = {}
    for path in sorted(prompt_dir.glob("*.txt")):
        templates[path.stem] = path.read_text(encoding="utf-8")
    return templates


def ensure_run_config_seeded(session: Session, config_name: str) -> RunConfig:
    existing = session.scalar(select(RunConfig).where(RunConfig.config_name == config_name))
    if existing is not None:
        return existing

    config_file = load_run_config_file(config_name)
    run_config = RunConfig(
        config_name=config_file.config_name,
        feature_version=config_file.feature_version,
        winner_model_version=config_file.winner_model_version,
        margin_model_version=config_file.margin_model_version,
        prompt_set_version=config_file.prompt_set_version,
        final_model_provider=config_file.final_model.provider,
        final_model_name=config_file.final_model.model,
        default_temperature=config_file.final_model.temperature,
        default_reasoning_effort=config_file.final_model.reasoning_effort,
        config=config_file.model_dump(mode="json"),
    )
    session.add(run_config)
    session.flush()
    ensure_prompt_set_seeded(session, config_file.prompt_set_version)
    return run_config


def ensure_prompt_set_seeded(session: Session, prompt_set_version: str) -> None:
    templates = load_prompt_templates(prompt_set_version)
    existing_steps = {
        step_name
        for step_name in session.scalars(
            select(PromptTemplate.step_name).where(
                PromptTemplate.prompt_set_version == prompt_set_version,
            )
        )
    }
    for step_name, template_text in templates.items():
        if step_name in existing_steps:
            continue
        session.add(
            PromptTemplate(
                prompt_set_version=prompt_set_version,
                step_name=step_name,
                template_text=template_text,
                response_schema_version=PROMPT_SCHEMA_VERSION.get(step_name, "json_v1"),
                is_active=True,
            )
        )
        existing_steps.add(step_name)
