# AFL Prediction Agent

Round-based H&A match prediction system for AFL men's. Produces **one locked verdict per match, per round**, from a deterministic data spine fronted by a multi-agent reasoning panel. Finals, AFLW, and in-game updates are out of scope.

Every prediction is reproducible: all sources are stored as timestamped snapshots, and every agent step (prompt, rendered input, model, reasoning effort, output, token usage, validation events) is persisted.

## What you get per match

| Output | Shape |
| --- | --- |
| Predicted winner | `team_id` |
| Win probabilities | `home_prob`, `away_prob` (sum to 1) |
| Predicted margin | Float, sign aligns with winner |
| Confidence | `0–100` |
| Top drivers | Ranked list with `leans_to`, `strength`, `evidence`, `source_component` |
| Analyst + case summaries | 4 analysts (form, selection, venue/weather, market) + home and away cases |
| Rationale | Final agent's own one-paragraph summary + uncertainty note |
| Reference tracks | Deterministic baseline (winner + margin), bookmaker median H2H, Squiggle tip/margin |

Reference tracks live alongside the agent verdict — they never override it, they're there for calibration and disagreement analysis.

## Pipeline

```mermaid
flowchart LR
    subgraph Sources["External sources"]
        FITZ["fitzRoy (Rscript bridge)<br/>fixtures · results · lineups · team+player stats"]
        ATAB["AFL Tables<br/>archive fallback"]
        AFLCOM["AFL.com<br/>injuries (primary)"]
        FW["FootyWire<br/>injuries (cross-check)"]
        BOM["BOM FTP<br/>weather forecast"]
        ODDS["The Odds API<br/>AU bookmaker H2H"]
        SQ["Squiggle<br/>benchmark tip + margin"]
    end

    subgraph Deterministic["Deterministic layers (all hashed + versioned)"]
        direction TB
        ING["Ingestion + entity resolution<br/>(team / player / venue / match)"]
        SNAP[("Snapshot storage<br/>lineup · injury · weather · odds · benchmark")]
        FEAT["Feature builder v1"]
        BASE["Baseline models<br/>LR winner + Ridge margin<br/>(heuristic fallback)"]
        DOSS["Match dossier builder"]
    end

    subgraph Agents["Agent panel"]
        ANA["4 analyst agents"]
        CASE["home + away case agents"]
        FIN["Final decision agent"]
        CORR["1 bounded correction pass<br/>(on validation fail only)"]
    end

    VAL["Validator<br/>probs sum = 1 · winner matches prob · margin sign · 0≤conf≤100"]
    STORE[("Postgres audit store<br/>prompts · rendered inputs · outputs · tokens · verdicts")]

    FITZ & ATAB & AFLCOM & FW & BOM & ODDS & SQ --> ING
    ING --> SNAP --> FEAT --> BASE --> DOSS
    DOSS --> ANA --> CASE --> FIN
    FIN -- pass --> VAL
    VAL -- fail --> CORR --> VAL
    VAL -- pass --> STORE
    VAL -- fail twice --> STORE
```

Agents only read the structured dossier and same-run outputs. No scraping, browsing, tool-calling, or cross-round memory in the reasoning step.

## Agent panel

```mermaid
graph LR
    D[(Match dossier)]

    D --> FA[form_analyst_v1]
    D --> SA[selection_analyst_v1]
    D --> VWA[venue_weather_analyst_v1]
    D --> MA[market_analyst_v1]

    FA & SA & VWA & MA --> HC[home_case_v1]
    FA & SA & VWA & MA --> AC[away_case_v1]

    FA & SA & VWA & MA --> FD[final_decision_v1]
    HC & AC --> FD

    FD --> V{{Validator}}
    V -- fail --> CP[correction_pass_v1]
    CP --> V
    V -- pass --> OUT[[FinalAgentVerdict]]
```

Each step has its own prompt in `config/prompts/prompt_set_v1/`, its own Pydantic response contract in `contracts.py`, and its own row in `agent_steps`.

### Providers

Configured per-step in `config/run_configs/<name>.json`:

| Provider | Use case | Notes |
| --- | --- | --- |
| `heuristic` | Local dev, CI, deterministic replay | Pure Python, no external calls. Same schema as LLM output. |
| `codex_app_server` | `gpt-5.4` via Codex app-server | Device-code auth. `reasoning_effort ∈ {none, low, medium, high, xhigh}`; `temperature` must be `null` once `reasoning_effort` is active on `gpt-5.4`. |

Two checked-in configs: `v1_agentic_default` (all-heuristic) and `v1_agentic_codex_gpt54` (all-`gpt-5.4-xhigh`).

## Sources + precedence

| Source | Role | Degrades to |
| --- | --- | --- |
| fitzRoy (official) | Fixtures, results, named lineups, team + player stats from 2012 | AFL Tables when `--use-archive-fallback` and the official pull returns 0 matches |
| AFL Tables | Deep history + fixture/result fallback | Not used for lineups or injuries |
| AFL.com injuries | **Primary** injury source at lock time | "missing injury snapshot" on dossier `uncertainties` |
| FootyWire injuries | Cross-check; disagreements recorded as `injury_source_disagreement` audit events | Silent if missing |
| BOM FTP | Weather forecast per venue (temperature, rain prob, rainfall, wind, severe flag) | "missing weather snapshot" on dossier `uncertainties`; neutral weather features |
| The Odds API | AU bookmaker H2H snapshots, median implied probability after overround normalisation | "thin bookmaker sample" uncertainty when `bookmaker_count < 3`; market analyst signal disabled if no prices at all |
| Squiggle | Benchmark tip + margin | Dossier continues without the benchmark block |

Precedence: **official AFL > non-official on fixtures/results/lineups**; **AFL.com > FootyWire on injuries**.

Mappings that can't auto-resolve are quarantined, not silently accepted:

```bash
afl-agent ingest review-unresolved
```

## Feature set (v1)

Produced deterministically by `features/builder.py` and versioned as `features_v1`.

| Group | Examples |
| --- | --- |
| Form | rolling 5-match win rate, avg margin, points for/against, home/away split, venue-specific win rate, rest days, ladder position, recent-vs-overall form momentum |
| Team stat edges | rolling deltas for inside-50, clearances, contested possessions, marks inside 50, tackles inside 50 |
| Selection | named changes vs last match, continuity score, lineup strength (aggregated player contribution rating over last 5 games), selected experience, missing-player penalty, injury count, key absences |
| Venue + weather | home ground edge (same-state rule), interstate travel burden, wet-weather flag (≥50% rain prob or ≥2mm), windy flag (≥25 km/h) |
| Market | median home/away implied probability, bookmaker count, market confidence (`|p − 0.5| × 2`) |
| Benchmark | Squiggle home win probability, predicted margin |

## Baseline models

`models/baseline.py`: `winner_lr_v1` (scikit-learn `LogisticRegression`) and `margin_ridge_v1` (`Ridge`), artifacts in `data/artifacts/`. When no artifact is present, a calibrated heuristic takes over — market logit anchor + form/split/lineup/continuity/experience/replacement/injury/rest/ladder/venue/weather/Squiggle edges — so the dossier is never missing baseline numbers.

## Validation

The final decision response is rejected before storage if any of these fail:

- `home_win_probability + away_win_probability == 1.0` (float tol)
- `predicted_winner_team_id` matches higher probability
- `predicted_margin` sign aligns with winner, non-zero
- `0 ≤ confidence_score ≤ 100`

On rejection, the runner invokes **one** `correction_pass_v1` with the original dossier + analysts + cases + validation error. A second failure is recorded as `final_agent_verdict_unavailable` and the match falls back to the deterministic baseline for reporting.

Soft warnings (logged, don't block): market disagreement > 0.18, baseline disagreement > 0.2, confidence > 85, margin > 60.

## Storage model

Postgres via SQLAlchemy + Alembic. Every run is audit-complete.

| Category | Tables |
| --- | --- |
| Canonical | `competitions`, `seasons`, `rounds`, `teams`, `players`, `venues`, `matches`, `source_external_ids` |
| Snapshots | `lineup_snapshots` (+players), `injury_snapshots` (+entries), `weather_snapshots`, `odds_snapshots` (+books), `benchmark_predictions`, `source_fetch_logs` |
| Stats + modelling | `team_match_stats`, `player_match_stats`, `feature_sets`, `baseline_model_runs`, `baseline_predictions`, `match_dossiers` |
| Agent runs | `run_configs`, `prompt_templates`, `round_runs`, `agent_steps` (rendered prompt, model, reasoning effort, input, output, tokens, provider meta, status), `final_agent_verdicts`, `validation_logs`, `audit_events` |
| Evaluation | `match_evaluations`, `season_evaluation_summaries` |

Eligibility rule for a round run: if official lineups for either side are missing at lock time, the match is skipped with a `match_excluded` audit event. No late-out reruns in v1.

## Run It End To End

### 1. Create the local env file

```bash
cp .env.example .env
```

Then edit `.env` and set at least:

```bash
AFL_AGENT_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/afl_agent
AFL_AGENT_CONTACT_EMAIL=your-real-email@example.com
```

Optional but recommended:

```bash
AFL_AGENT_ODDS_API_KEY=your_the_odds_api_key
AFL_AGENT_ODDS_AU_BOOKMAKERS=tab,sportsbet,neds,ladbrokes_au,betfair_ex_au,betr_au,pointsbetau,bluebet
```

Only set this if you want to override the default Squiggle header:

```bash
AFL_AGENT_SQUIGGLE_USER_AGENT=AFL Prediction Agent - your-real-email@example.com
```

### 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

### 3. Start Postgres

```bash
docker compose up -d postgres
```

### 4. Install R dependencies for `fitzRoy`

`fitzRoy` is required for official AFL and AFL Tables ingestion:

```bash
Rscript -e "install.packages(c('jsonlite','fitzRoy'), repos='https://cloud.r-project.org')"
```

### 5. Initialize the database and seed configs

```bash
alembic upgrade head
afl-agent seed-config --config-name v1_agentic_default
afl-agent seed-config --config-name v1_agentic_codex_gpt54
```

### 6. If you want `gpt-5.4`, authenticate Codex once

Skip this if you only want the deterministic `heuristic` config.

```bash
afl-agent auth codex login --device-code
afl-agent auth codex status
```

You want `auth_mode=chatgpt` and a supported paid plan in the status output.

### 7. Start the API

```bash
PYTHONPATH=src uvicorn afl_prediction_agent.api.app:app --reload
```

### 8. Ingest football data

For a target season and round:

```bash
afl-agent ingest fixtures 2026 --round-number 7
afl-agent ingest results 2026 --round-number 7
afl-agent ingest stats 2026 --round-number 7 --source-track official
afl-agent ingest stats 2026 --round-number 7 --source-track archive
```

If you do not set `AFL_AGENT_ODDS_AU_BOOKMAKERS`, the app uses the checked-in default allowlist from [`data/mappings/odds_bookmakers_au.json`](data/mappings/odds_bookmakers_au.json). Set the env var only if you want to override that list.

If you want to review any unresolved mappings before running:

```bash
afl-agent ingest review-unresolved
```

### 9. Find the `round_id`

Use Postgres directly:

```bash
psql "postgresql://postgres:postgres@localhost:5432/afl_agent" -c "
select
  r.id,
  s.season_year,
  r.round_number,
  r.round_name
from rounds r
join seasons s on s.id = r.season_id
order by s.season_year desc, r.round_number desc;
"
```

### 10. Capture round snapshots

This pulls lineups, injuries, weather, odds, and benchmarks for that round:

```bash
afl-agent ingest snapshot-round <round_id>
```

### 11. Run the prediction pipeline

Heuristic/local run:

```bash
afl-agent run-round <round_id> --config-name v1_agentic_default
```

`gpt-5.4` run through local Codex app-server auth:

```bash
afl-agent run-round <round_id> --config-name v1_agentic_codex_gpt54
```

`run-round` defaults to `--fetch-sources`, so you can also skip the separate snapshot command and let the run capture the latest source state itself:

```bash
afl-agent run-round <round_id> --config-name v1_agentic_codex_gpt54 --fetch-sources
```

### 12. Inspect the run

List runs for that round:

```bash
afl-agent list-runs <round_id>
```

Inspect over HTTP:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/runs/<run_id>
curl http://127.0.0.1:8000/runs/<run_id>/matches/<match_id>
```

### 13. Evaluate after results land

Once the actual round results are present in `matches`:

```bash
afl-agent evaluate-run <run_id>
```

### 14. Replay historical rounds or seasons

```bash
afl-agent replay-round <round_id> --config-name v1_agentic_default --lock-timestamp 2026-05-01T09:00:00+00:00
afl-agent replay-season <season_id> --config-name v1_agentic_default
```

### 15. Minimal happy path

If you just want the shortest working sequence:

```bash
source .venv/bin/activate
docker compose up -d postgres
alembic upgrade head
afl-agent seed-config --config-name v1_agentic_default
afl-agent ingest fixtures 2026 --round-number 7
afl-agent ingest stats 2026 --round-number 7 --source-track official
afl-agent ingest stats 2026 --round-number 7 --source-track archive
afl-agent run-round <round_id> --config-name v1_agentic_default
```

## Round workflow

```mermaid
sequenceDiagram
    autonumber
    participant You
    participant CLI as afl-agent
    participant DB as Postgres
    participant LLM as Agent panel

    You->>CLI: ingest fixtures <season> [--round-number N]
    You->>CLI: ingest stats <season> [--round-number N]
    You->>CLI: ingest snapshot-round <round_id>
    CLI->>DB: lineup + injury + weather + odds snapshots
    You->>CLI: run-round <round_id> --config-name v1_agentic_codex_gpt54
    CLI->>DB: features → baseline → dossier
    CLI->>LLM: analysts → cases → final (+ correction on fail)
    LLM-->>CLI: FinalDecisionResponse
    CLI->>DB: validate + store verdict + audit
    You->>CLI: evaluate-run <run_id>   # after results land
```

Most common commands:

```bash
afl-agent ingest fixtures 2026 --round-number 7
afl-agent ingest stats 2026 --round-number 7
afl-agent ingest snapshot-round <round_id>     # lineups + injuries + weather + odds (+ benchmarks if run_id given)
afl-agent run-round <round_id> --config-name v1_agentic_codex_gpt54
afl-agent list-runs <round_id>
afl-agent evaluate-run <run_id>                # Brier, log loss, winner accuracy, margin MAE/RMSE vs market, Squiggle, baseline
afl-agent replay-round <round_id> --config-name v1_agentic_default --lock-timestamp 2026-05-01T09:00:00+00:00
afl-agent replay-season <season_id> --config-name v1_agentic_default
```

`run-round` defaults to `--fetch-sources`, which calls `snapshot-round` for you inside the same run.

## HTTP API

`uvicorn afl_prediction_agent.api.app:app --reload`:

| Method + path | Purpose |
| --- | --- |
| `GET /health` | Liveness |
| `GET /rounds/{round_id}/runs` | All runs for a round |
| `GET /runs/{run_id}` | Run summary (match counts, status, timestamps) |
| `GET /runs/{run_id}/matches/{match_id}` | Full match dossier + baseline + agent steps + verdict |
| `POST /rounds/{round_id}/run` | Kick off a run (`RunRoundRequest`) |
| `POST /rounds/{round_id}/replay` | Replay with a fixed `lock_timestamp` |
| `POST /runs/{run_id}/evaluate` | Score against actual results + benchmark tracks |
| `GET /seasons/{season_id}/summary` | Per-config season summaries |

## Requirements

- Python ≥ 3.12, Postgres 16
- **R + `fitzRoy` + `jsonlite`**, reachable via `Rscript` — mandatory for official AFL and AFL Tables ingestion
- `AFL_AGENT_ODDS_API_KEY` from [The Odds API](https://the-odds-api.com/) for odds snapshots
- `AFL_AGENT_CONTACT_EMAIL` or `AFL_AGENT_SQUIGGLE_USER_AGENT` for Squiggle, because the live API requires an identifying `User-Agent` with a contact email
- Curated venue → BOM station mapping at [`data/mappings/venue_bom_mapping.json`](data/mappings/venue_bom_mapping.json) (ships checked-in; extend when new venues appear)
- Codex CLI on `PATH` only if you use `codex_app_server` runs (see `AFL_AGENT_CODEX_BIN`)

See [`.env.example`](.env.example) for the full tunable set (`AFL_AGENT_*`).

## Scope boundaries

In scope: AFL men's H&A, one run per round, predictions locked at official team-announcement time, walk-forward backtests from 2012.

Out of scope for v1: finals, AFLW, live/in-game updates, news or sentiment, player props, totals/lines as primary outputs, automated betting, cross-round memory, self-training from prior agent outputs.

## Repo layout

```
src/afl_prediction_agent/
  agents/          runner + adapters (heuristic, codex_app_server) + codex client
  api/             FastAPI inspection endpoints
  configuration.py run-config + prompt seeding
  contracts.py     Pydantic dossier + agent response + run config contracts
  core/            settings, DB session, base
  dossiers/        DossierBuilder
  features/        FeatureBuilder (features_v1)
  ingestion/       FixtureIngestionService, SnapshotIngestionService, StatsIngestionService
  models/          DeterministicBaselineService + model registry (joblib)
  orchestration/   RoundRunService, ReplayService, EvaluationService
  sources/         afl (fitzRoy), afl_tables, afl_com, footywire, bom, odds, squiggle, common
  storage/         SQLAlchemy models + match context loader
  validation/      final-response rule checks
  cli.py           Typer entrypoint (afl-agent)

config/
  prompts/prompt_set_v1/   one .txt per step
  run_configs/             v1_agentic_default.json, v1_agentic_codex_gpt54.json

data/
  mappings/        team_aliases, player_aliases, venue_bom_mapping, odds_bookmakers_au
  artifacts/       trained baseline model artifacts (.joblib)

migrations/        Alembic
```

## Spec references

- [`docs/afl_prediction_agent_v_1_spec.md`](docs/afl_prediction_agent_v_1_spec.md) — product spec, locked decisions, acceptance criteria
- [`docs/afl_prediction_agent_implementation_plan_and_schema.md`](docs/afl_prediction_agent_implementation_plan_and_schema.md) — full schema + phase plan
- [`docs/Plan_Implement the Full Source Connector Layer.md`](docs/Plan_Implement%20the%20Full%20Source%20Connector%20Layer.md) — connector layer design
