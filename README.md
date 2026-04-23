# AFL Prediction Agent

Spec-driven implementation of the AFL prediction system described in:

1. `afl_prediction_agent_v_1_spec.md`
2. `afl_prediction_agent_implementation_plan_and_schema.md`

This repo now includes:

1. SQLAlchemy ORM models for the v1 schema
2. Alembic migration scaffolding
3. Pydantic dossier and agent output contracts
4. Deterministic feature and baseline services
5. A provider-abstracted agent runner with a deterministic heuristic adapter
6. A round orchestration service
7. FastAPI inspection endpoints
8. Typer CLI entrypoints

## Quick start

1. Install the package:

```bash
python3 -m pip install -e ".[dev]"
```

2. Set a database URL:

```bash
export AFL_AGENT_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/afl_agent"
```

3. Run migrations:

```bash
alembic upgrade head
```

4. Seed prompts and run config:

```bash
afl-agent seed-config
```

5. Start the API:

```bash
uvicorn afl_prediction_agent.api.app:app --reload
```

## Current scope

The codebase implements the data spine, schema, typed contracts, run orchestration, evaluation flow, and a deterministic first-pass agent provider. Source-specific AFL/BOM/Odds scrapers are left as integration points and can write into the ingestion services already provided here.
