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
9. Source-specific connectors for official AFL via `fitzRoy`, AFL Tables, AFL.com injuries, FootyWire, BOM FTP feeds, The Odds API, and Squiggle

## Quick start

1. Install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

2. Start local Postgres on Docker:

```bash
docker compose up -d postgres
```

3. Use the checked-in local env file, or export the same database URL manually:

```bash
export AFL_AGENT_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/afl_agent"
```

4. Run migrations:

```bash
alembic upgrade head
```

5. Seed prompts and run configs:

```bash
afl-agent seed-config --config-name v1_agentic_default
afl-agent seed-config --config-name v1_agentic_codex_gpt54
```

6. Start the API:

```bash
uvicorn afl_prediction_agent.api.app:app --reload
```

7. Optional Codex auth status check:

```bash
afl-agent auth codex status
```

8. Optional source config:

```bash
export AFL_AGENT_FITZROY_RSCRIPT_BIN=Rscript
export AFL_AGENT_ODDS_API_KEY=your_the_odds_api_key
```

9. Ingest official data and capture a round snapshot:

```bash
afl-agent ingest fixtures 2026 --round-number 7
afl-agent ingest stats 2026 --round-number 7
afl-agent ingest snapshot-round <round_id>
afl-agent run-round <round_id> --config-name v1_agentic_codex_gpt54
```

10. Review unresolved source mappings:

```bash
afl-agent ingest review-unresolved
```

## Current scope

The codebase implements the data spine, schema, typed contracts, source connector layer, run orchestration, evaluation flow, and both heuristic and Codex-backed agent providers.

The source layer uses:

1. `fitzRoy` via a local `Rscript` bridge for official AFL data and AFL Tables archive data
2. AFL.com as the primary injury source, with FootyWire as a secondary cross-check
3. BOM anonymous FTP / Weather Data Services products only for weather
4. The Odds API for bookmaker head-to-head snapshots
5. Squiggle for benchmark predictions

Current practical requirements:

1. R plus the `fitzRoy` and `jsonlite` packages for official AFL and AFL Tables ingestion
2. `AFL_AGENT_ODDS_API_KEY` if you want odds snapshots
3. curated venue mappings under [data/mappings/venue_bom_mapping.json](/Users/adamsardo/Developer/afl_prediction_agent/data/mappings/venue_bom_mapping.json)
