# AFL Prediction Agent Implementation Plan and Schema

## 1. Goal

Turn the approved agentic v1 spec into a buildable system with:

1. a clean data spine
2. reproducible round level snapshots
3. deterministic baseline models
4. structured match dossiers
5. analyst, case, and final decision agents
6. stored run history for replay and evaluation
7. round outputs that can be trusted, inspected, and compared over time

This document covers:

1. recommended implementation approach
2. suggested stack
3. repository structure
4. delivery phases
5. database schema
6. structured object schemas for dossiers and agent outputs
7. validation and evaluation requirements

## 2. Recommended stack

Keep it simple. This is a leisure system, not a bank pretending to be a football nerd.

### Core stack

1. Python 3.12
2. Postgres 16
3. SQLAlchemy + Alembic
4. Pydantic v2 for typed schemas
5. Polars for feature work and data prep
6. scikit learn for baseline models
7. FastAPI for a lightweight internal API
8. Typer for CLI jobs
9. object storage optional for raw payload archives if Postgres JSONB becomes annoying

### Why this stack

1. Python is the easiest fit for scraping, feature engineering, modelling, and LLM orchestration.
2. Postgres gives you strong relational structure plus JSONB for source payloads and run artifacts.
3. Alembic keeps schema changes sane.
4. Pydantic gives typed dossier and agent contract validation.
5. Polars is fast and clean for rolling AFL feature pipelines.
6. scikit learn is enough for v1 baselines.
7. FastAPI gives you a lightweight way to inspect runs and outputs.
8. Typer makes scheduled or manual commands straightforward.

### LLM provider abstraction

Use a thin adapter layer so the system is not married to one provider.

Minimum interface:

1. model provider
2. exact model name
3. temperature
4. response format mode
5. token usage
6. error handling and retries

## 3. System boundaries

### Deterministic components

1. source ingestion
2. entity resolution
3. snapshot storage
4. feature generation
5. baseline winner model
6. baseline margin model
7. dossier construction
8. output validation
9. evaluation and reporting

### Agentic components

1. form analyst agent
2. selection analyst agent
3. venue and weather analyst agent
4. market analyst agent
5. home case agent
6. away case agent
7. final decision agent

### Explicit non goals for v1

1. no cross round memory
2. no self training from prior outputs
3. no autonomous rescraping by agents
4. no live updates after late outs
5. no automated bet execution

## 4. Repository structure

Recommended single repo structure:

```text
afl-agent/
  apps/
    api/
    admin/
  config/
    prompts/
    run_configs/
  data/
    mappings/
    seeds/
  migrations/
  src/
    core/
      db/
      logging/
      settings/
      types/
    sources/
      afl/
      afl_tables/
      footywire/
      bom/
      odds/
      squiggle/
    ingestion/
      fixtures/
      lineups/
      injuries/
      weather/
      odds/
    canonical/
      entities/
      resolvers/
      mappers/
    features/
      builders/
      selectors/
      ratings/
    models/
      winner/
      margin/
      evaluation/
    dossiers/
      builders/
      validators/
    agents/
      adapters/
      prompts/
      runners/
      validators/
    orchestration/
      round_runs/
      replay/
    reporting/
      round/
      season/
      comparisons/
  tests/
    unit/
    integration/
    replay/
  scripts/
    backfill/
    maintenance/
```

## 5. Delivery phases

## Phase 1
### Foundation and canonical data spine

### Scope

1. create database and migrations
2. create core entity tables
3. implement source fetch logging
4. implement team, player, venue, and match identity mapping
5. ingest fixtures and results
6. ingest lineups
7. seed venue to BOM mapping

### Deliverables

1. working Postgres schema
2. CLI commands for fixture and lineup ingestion
3. canonical match ids
4. first end to end stored round with match, team, player, and lineup data

### Done when

1. you can pull a round and see all matches with canonical ids
2. each lineup is stored as a timestamped snapshot
3. unresolved players and venues are surfaced for manual review

## Phase 2
### Snapshots and external context

### Scope

1. injury snapshot ingestion
2. weather snapshot ingestion
3. odds snapshot ingestion
4. Squiggle benchmark ingestion
5. snapshot bundle logic for reproducible match inputs

### Deliverables

1. match level snapshot sets
2. odds median calculation
3. injury status normalisation
4. weather feature ready records

### Done when

1. every eligible match can be tied to one snapshot bundle
2. bundle contains lineups, injuries, weather, and odds
3. every fetch has source, timestamp, and status metadata

## Phase 3
### Features and deterministic baselines

### Scope

1. team form features
2. venue and travel features
3. selection continuity and lineup strength features
4. market features
5. winner baseline model
6. margin baseline model
7. walk forward evaluation harness

### Deliverables

1. feature version 1
2. baseline winner outputs
3. baseline margin outputs
4. backtest reports against market, Squiggle, and naive baselines

### Done when

1. you can run a historical round and get baseline outputs
2. the outputs are versioned and reproducible
3. evaluation tables are populated

## Phase 4
### Dossiers and agent pipeline

### Scope

1. structured dossier builder
2. typed agent input schema
3. analyst agent prompts
4. case agent prompts
5. final decision agent prompt
6. output validator and one correction pass
7. full run metadata storage

### Deliverables

1. per match dossier records
2. analyst outputs
3. home and away case outputs
4. final decision outputs
5. validation logs

### Done when

1. a full round run produces both baseline and agent verdict tracks
2. every agent step is stored with prompt, model, settings, inputs, and outputs
3. final outputs always pass schema validation or fail explicitly

## Phase 5
### Replay, comparisons, and review layer

### Scope

1. replay historical rounds using fixed configs
2. compare agent versus baseline performance
3. compare prompt versions and model versions
4. expose round results via API or a lightweight admin page

### Deliverables

1. run comparison views
2. season summary reports
3. per match output view
4. prompt and model version comparison reports

### Done when

1. you can inspect any prior round run end to end
2. you can compare two configs without guessing what changed
3. you can export round predictions cleanly for personal use

## 6. Core runtime flow

### Pre round run

1. create round run record
2. fetch and store latest lineups
3. fetch and store injuries
4. fetch and store weather
5. fetch and store odds
6. build snapshot bundles per match
7. build features per match
8. run baseline winner and margin models
9. build dossier per match
10. run analyst agents
11. run home and away case agents
12. run final decision agent
13. validate outputs
14. store final verdicts and all run artifacts

### Post round flow

1. ingest results
2. calculate winner correctness
3. calculate margin error
4. calculate Brier score and log loss
5. compare against market, Squiggle, and baseline
6. persist evaluation summary

## 7. Schema design principles

1. store canonical football entities relationally
2. store source payloads and agent artifacts in JSONB where the shape is wide or evolving
3. use versioned run configs for reproducibility
4. separate baseline outputs from final agent verdicts
5. keep snapshot history immutable once written
6. never overwrite a prior run result

## 8. Database schema overview

## 8.1 Core football entities

### competitions

Purpose: competition dimension

Columns:

1. id UUID PK
2. code TEXT UNIQUE
3. name TEXT
4. created_at TIMESTAMPTZ

### seasons

Purpose: competition season container

Columns:

1. id UUID PK
2. competition_id UUID FK competitions.id
3. season_year INT
4. created_at TIMESTAMPTZ
5. UNIQUE (competition_id, season_year)

### rounds

Purpose: round container

Columns:

1. id UUID PK
2. season_id UUID FK seasons.id
3. round_number INT
4. round_name TEXT
5. is_finals BOOLEAN DEFAULT FALSE
6. starts_at TIMESTAMPTZ NULL
7. ends_at TIMESTAMPTZ NULL
8. created_at TIMESTAMPTZ
9. UNIQUE (season_id, round_number, is_finals)

### teams

Purpose: canonical team identity

Columns:

1. id UUID PK
2. team_code TEXT UNIQUE
3. name TEXT
4. short_name TEXT
5. slug TEXT UNIQUE
6. state_code TEXT NULL
7. created_at TIMESTAMPTZ

### venues

Purpose: canonical venue identity and weather mapping

Columns:

1. id UUID PK
2. venue_code TEXT UNIQUE NULL
3. name TEXT
4. city TEXT NULL
5. state_code TEXT NULL
6. timezone TEXT DEFAULT 'Australia/Melbourne'
7. bom_location_code TEXT NULL
8. bom_station_id TEXT NULL
9. latitude NUMERIC NULL
10. longitude NUMERIC NULL
11. created_at TIMESTAMPTZ

### players

Purpose: canonical player identity

Columns:

1. id UUID PK
2. player_code TEXT UNIQUE NULL
3. full_name TEXT
4. first_name TEXT NULL
5. last_name TEXT NULL
6. current_team_id UUID FK teams.id NULL
7. active BOOLEAN DEFAULT TRUE
8. created_at TIMESTAMPTZ

### matches

Purpose: canonical match record

Columns:

1. id UUID PK
2. season_id UUID FK seasons.id
3. round_id UUID FK rounds.id
4. match_code TEXT UNIQUE NULL
5. home_team_id UUID FK teams.id
6. away_team_id UUID FK teams.id
7. venue_id UUID FK venues.id
8. scheduled_at TIMESTAMPTZ
9. status TEXT
10. home_score INT NULL
11. away_score INT NULL
12. winning_team_id UUID FK teams.id NULL
13. actual_margin INT NULL
14. created_at TIMESTAMPTZ
15. updated_at TIMESTAMPTZ

Indexes:

1. (round_id)
2. (scheduled_at)
3. (home_team_id, away_team_id, scheduled_at)

## 8.2 Source tracking and mapping

### source_fetch_logs

Purpose: every external fetch gets logged

Columns:

1. id UUID PK
2. source_name TEXT
3. entity_type TEXT
4. entity_key TEXT NULL
5. requested_at TIMESTAMPTZ
6. completed_at TIMESTAMPTZ NULL
7. status TEXT
8. request_meta JSONB DEFAULT '{}'::jsonb
9. response_meta JSONB DEFAULT '{}'::jsonb
10. raw_payload JSONB NULL
11. error_message TEXT NULL

Indexes:

1. (source_name, requested_at DESC)
2. (entity_type, entity_key)

### external_id_mappings

Purpose: map source specific ids to canonical ids

Columns:

1. id UUID PK
2. source_name TEXT
3. entity_type TEXT
4. external_id TEXT
5. canonical_id UUID
6. confidence_score NUMERIC NULL
7. mapping_status TEXT
8. notes TEXT NULL
9. created_at TIMESTAMPTZ
10. updated_at TIMESTAMPTZ
11. UNIQUE (source_name, entity_type, external_id)

## 8.3 Snapshots

### round_runs

Purpose: top level execution record for a round prediction run

Columns:

1. id UUID PK
2. season_id UUID FK seasons.id
3. round_id UUID FK rounds.id
4. run_config_id UUID FK run_configs.id
5. lock_timestamp TIMESTAMPTZ
6. status TEXT
7. notes TEXT NULL
8. created_at TIMESTAMPTZ
9. completed_at TIMESTAMPTZ NULL

Indexes:

1. (round_id, created_at DESC)
2. (run_config_id)

### lineup_snapshots

Purpose: one stored lineup snapshot per match and team

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id NULL
3. match_id UUID FK matches.id
4. team_id UUID FK teams.id
5. source_name TEXT
6. fetched_at TIMESTAMPTZ
7. payload JSONB
8. created_at TIMESTAMPTZ

Indexes:

1. (match_id, team_id, fetched_at DESC)

### lineup_snapshot_players

Purpose: players named in a lineup snapshot

Columns:

1. id UUID PK
2. lineup_snapshot_id UUID FK lineup_snapshots.id
3. player_id UUID FK players.id NULL
4. source_player_name TEXT
5. slot_label TEXT NULL
6. named_role TEXT NULL
7. is_selected BOOLEAN DEFAULT TRUE
8. is_interchange BOOLEAN DEFAULT FALSE
9. is_emergency BOOLEAN DEFAULT FALSE
10. is_sub BOOLEAN DEFAULT FALSE
11. mapping_status TEXT
12. created_at TIMESTAMPTZ

Indexes:

1. (lineup_snapshot_id)
2. (player_id)

### injury_snapshots

Purpose: injury list at run time

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id NULL
3. source_name TEXT
4. fetched_at TIMESTAMPTZ
5. payload JSONB
6. created_at TIMESTAMPTZ

### injury_snapshot_entries

Purpose: injury rows from the injury snapshot

Columns:

1. id UUID PK
2. injury_snapshot_id UUID FK injury_snapshots.id
3. team_id UUID FK teams.id NULL
4. player_id UUID FK players.id NULL
5. source_player_name TEXT
6. status_label TEXT
7. injury_note TEXT NULL
8. estimated_return_text TEXT NULL
9. uncertainty_flag BOOLEAN DEFAULT FALSE
10. mapping_status TEXT
11. created_at TIMESTAMPTZ

Indexes:

1. (injury_snapshot_id)
2. (team_id)
3. (player_id)

### weather_snapshots

Purpose: weather context per match

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id NULL
3. match_id UUID FK matches.id
4. venue_id UUID FK venues.id
5. source_name TEXT
6. fetched_at TIMESTAMPTZ
7. temperature_c NUMERIC NULL
8. rain_probability_pct NUMERIC NULL
9. rainfall_mm NUMERIC NULL
10. wind_kmh NUMERIC NULL
11. weather_text TEXT NULL
12. severe_flag BOOLEAN DEFAULT FALSE
13. payload JSONB
14. created_at TIMESTAMPTZ

Indexes:

1. (match_id, fetched_at DESC)

### odds_snapshots

Purpose: market context per match at lock time

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id NULL
3. match_id UUID FK matches.id
4. source_name TEXT
5. fetched_at TIMESTAMPTZ
6. home_median_price NUMERIC NULL
7. away_median_price NUMERIC NULL
8. home_implied_probability NUMERIC NULL
9. away_implied_probability NUMERIC NULL
10. bookmaker_count INT DEFAULT 0
11. payload JSONB
12. created_at TIMESTAMPTZ

Indexes:

1. (match_id, fetched_at DESC)

### odds_snapshot_books

Purpose: bookmaker level market rows

Columns:

1. id UUID PK
2. odds_snapshot_id UUID FK odds_snapshots.id
3. bookmaker_key TEXT
4. market_key TEXT
5. home_price NUMERIC
6. away_price NUMERIC
7. overround_pct NUMERIC NULL
8. created_at TIMESTAMPTZ

### benchmark_predictions

Purpose: external comparison track such as Squiggle

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. source_name TEXT
5. predicted_winner_team_id UUID FK teams.id NULL
6. home_win_probability NUMERIC NULL
7. away_win_probability NUMERIC NULL
8. predicted_margin NUMERIC NULL
9. payload JSONB
10. created_at TIMESTAMPTZ

## 8.4 Historical football stats

### team_match_stats

Purpose: normalised team stats per match

Columns:

1. id UUID PK
2. match_id UUID FK matches.id
3. team_id UUID FK teams.id
4. source_name TEXT
5. stats JSONB
6. created_at TIMESTAMPTZ
7. UNIQUE (match_id, team_id, source_name)

### player_match_stats

Purpose: normalised player stats per match

Columns:

1. id UUID PK
2. match_id UUID FK matches.id
3. team_id UUID FK teams.id
4. player_id UUID FK players.id
5. source_name TEXT
6. stats JSONB
7. created_at TIMESTAMPTZ
8. UNIQUE (match_id, player_id, source_name)

## 8.5 Feature and baseline model storage

### feature_sets

Purpose: versioned per match feature payload

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. feature_version TEXT
5. input_hash TEXT
6. features JSONB
7. created_at TIMESTAMPTZ

Indexes:

1. (match_id, feature_version)
2. (round_run_id)

### baseline_model_runs

Purpose: versioned deterministic model execution metadata

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. model_type TEXT
4. model_version TEXT
5. feature_version TEXT
6. training_window TEXT
7. config JSONB
8. created_at TIMESTAMPTZ

### baseline_predictions

Purpose: baseline winner and margin outputs per match

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. winner_model_run_id UUID FK baseline_model_runs.id NULL
5. margin_model_run_id UUID FK baseline_model_runs.id NULL
6. predicted_winner_team_id UUID FK teams.id NULL
7. home_win_probability NUMERIC
8. away_win_probability NUMERIC
9. predicted_margin NUMERIC
10. confidence_reference NUMERIC NULL
11. top_drivers JSONB
12. created_at TIMESTAMPTZ

Indexes:

1. (round_run_id, match_id)

## 8.6 Dossiers and agent runs

### run_configs

Purpose: fixed configuration envelope for a run

Columns:

1. id UUID PK
2. config_name TEXT UNIQUE
3. feature_version TEXT
4. winner_model_version TEXT
5. margin_model_version TEXT
6. prompt_set_version TEXT
7. final_model_provider TEXT
8. final_model_name TEXT
9. default_temperature NUMERIC
10. config JSONB
11. created_at TIMESTAMPTZ

### prompt_templates

Purpose: prompt template registry

Columns:

1. id UUID PK
2. prompt_set_version TEXT
3. step_name TEXT
4. template_text TEXT
5. response_schema_version TEXT
6. is_active BOOLEAN DEFAULT TRUE
7. created_at TIMESTAMPTZ
8. UNIQUE (prompt_set_version, step_name)

### match_dossiers

Purpose: structured input package for each match

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. dossier_version TEXT
5. input_hash TEXT
6. dossier JSONB
7. created_at TIMESTAMPTZ
8. UNIQUE (round_run_id, match_id)

### agent_steps

Purpose: one row per agent execution step

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. step_name TEXT
5. prompt_template_id UUID FK prompt_templates.id NULL
6. rendered_prompt TEXT
7. model_provider TEXT
8. model_name TEXT
9. temperature NUMERIC
10. input_json JSONB
11. output_json JSONB NULL
12. status TEXT
13. attempt_number INT DEFAULT 1
14. tokens_input INT NULL
15. tokens_output INT NULL
16. started_at TIMESTAMPTZ
17. completed_at TIMESTAMPTZ NULL
18. error_message TEXT NULL

Indexes:

1. (round_run_id, match_id)
2. (step_name)
3. (status)

### final_agent_verdicts

Purpose: public agent output per match

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. final_agent_step_id UUID FK agent_steps.id
5. predicted_winner_team_id UUID FK teams.id
6. home_win_probability NUMERIC
7. away_win_probability NUMERIC
8. predicted_margin NUMERIC
9. confidence_score NUMERIC
10. top_drivers JSONB
11. uncertainty_note TEXT
12. rationale_summary TEXT
13. validation_status TEXT
14. correction_pass_count INT DEFAULT 0
15. created_at TIMESTAMPTZ

Indexes:

1. (round_run_id, match_id)

### validation_logs

Purpose: schema and consistency checks on outputs

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. component_name TEXT
5. validation_status TEXT
6. errors JSONB NULL
7. created_at TIMESTAMPTZ

### audit_events

Purpose: late outs, source issues, retries, and other run level events

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id NULL
3. match_id UUID FK matches.id NULL
4. event_type TEXT
5. payload JSONB
6. created_at TIMESTAMPTZ

## 8.7 Evaluation tables

### match_evaluations

Purpose: post result scoring per match and track

Columns:

1. id UUID PK
2. round_run_id UUID FK round_runs.id
3. match_id UUID FK matches.id
4. actual_winner_team_id UUID FK teams.id
5. actual_margin NUMERIC
6. agent_winner_correct BOOLEAN
7. baseline_winner_correct BOOLEAN
8. agent_margin_error NUMERIC
9. baseline_margin_error NUMERIC
10. agent_brier NUMERIC NULL
11. baseline_brier NUMERIC NULL
12. agent_log_loss NUMERIC NULL
13. baseline_log_loss NUMERIC NULL
14. created_at TIMESTAMPTZ

### season_evaluation_summaries

Purpose: cached rollups for quick comparison

Columns:

1. id UUID PK
2. season_id UUID FK seasons.id
3. run_config_id UUID FK run_configs.id
4. summary_type TEXT
5. summary JSONB
6. created_at TIMESTAMPTZ

## 9. Key relationships

1. one season has many rounds
2. one round has many matches
3. one round run belongs to one round and one config
4. one round run produces many snapshots, one feature set per match, one dossier per match, many agent steps per match, and one final verdict per match
5. one match later receives one evaluation record per run

## 10. Snapshot bundle concept

For reproducibility, each match dossier should be built from a fixed logical bundle:

1. lineup snapshot ids for both teams
2. injury snapshot id
3. weather snapshot id
4. odds snapshot id
5. benchmark snapshot id if applicable
6. feature set id
7. baseline prediction id

This can live inside the dossier JSON and also be materialised in a helper table later if needed.

## 11. Structured dossier schema

Suggested Pydantic shape:

```json
{
  "match": {
    "match_id": "uuid",
    "season_year": 2026,
    "round_number": 7,
    "scheduled_at": "2026-04-25T19:50:00+10:00",
    "venue": {
      "name": "MCG",
      "city": "Melbourne"
    },
    "home_team": {
      "team_id": "uuid",
      "name": "Carlton"
    },
    "away_team": {
      "team_id": "uuid",
      "name": "Geelong"
    }
  },
  "baseline": {
    "home_win_probability": 0.57,
    "away_win_probability": 0.43,
    "predicted_margin": 8.5,
    "top_drivers": []
  },
  "form": {
    "home_recent_form": {},
    "away_recent_form": {},
    "team_stat_edges": []
  },
  "selection": {
    "home_named_changes": 2,
    "away_named_changes": 4,
    "home_lineup_strength": 73.2,
    "away_lineup_strength": 69.8,
    "key_absences": []
  },
  "venue_weather": {
    "home_ground_edge": true,
    "travel_context": {},
    "forecast": {}
  },
  "market": {
    "home_implied_probability": 0.54,
    "away_implied_probability": 0.46,
    "bookmaker_count": 6
  },
  "benchmarks": {
    "squiggle": {}
  },
  "uncertainties": [
    "late selection uncertainty around key tall"
  ],
  "source_refs": {
    "lineup_snapshot_ids": [],
    "injury_snapshot_id": "uuid",
    "weather_snapshot_id": "uuid",
    "odds_snapshot_id": "uuid"
  }
}
```

## 12. Analyst output schemas

### Analyst response schema

```json
{
  "summary": "string",
  "signals": [
    {
      "label": "inside 50 edge",
      "leans_to": "home",
      "strength": 0.72,
      "evidence": "home side has led this metric in 5 of last 6"
    }
  ],
  "risks": ["string"],
  "unknowns": ["string"]
}
```

### Case agent response schema

```json
{
  "side": "home",
  "case_summary": "string",
  "strongest_points": [
    {
      "label": "lineup continuity",
      "strength": 0.81,
      "evidence": "string"
    }
  ],
  "weak_points": ["string"],
  "rebuttals": ["string"]
}
```

### Final decision agent response schema

```json
{
  "predicted_winner_team_id": "uuid",
  "home_win_probability": 0.61,
  "away_win_probability": 0.39,
  "predicted_margin": 12,
  "confidence_score": 74,
  "top_drivers": [
    {
      "label": "selection strength",
      "leans_to": "home",
      "strength": 0.78,
      "evidence": "string",
      "source_component": "selection_analyst"
    }
  ],
  "uncertainty_note": "string",
  "rationale_summary": "string"
}
```

## 13. Validation rules

### Hard rules

1. home and away probabilities must sum to 1 within a tiny tolerance
2. winner must match the higher probability team
3. margin sign must align with the winner
4. confidence score must be between 0 and 100
5. top drivers array cannot be empty
6. required fields must be present

### Soft rules

1. confidence above 85 should be rare and reviewable
2. very large margins should be flagged for sanity review
3. strong disagreement with market should be logged explicitly
4. strong disagreement with baseline should be logged explicitly

## 14. Prompt set structure

Version prompts as a set, not as random text blobs scattered around your repo like confetti.

Suggested prompt set files:

1. form_analyst_v1.txt
2. selection_analyst_v1.txt
3. venue_weather_analyst_v1.txt
4. market_analyst_v1.txt
5. home_case_v1.txt
6. away_case_v1.txt
7. final_decision_v1.txt
8. correction_pass_v1.txt

Each prompt should declare:

1. role
2. allowed inputs
3. required output schema
4. constraints
5. refusal conditions

## 15. Run configuration schema

Suggested config file shape:

```json
{
  "config_name": "v1_agentic_default",
  "feature_version": "features_v1",
  "winner_model_version": "winner_lr_v1",
  "margin_model_version": "margin_ridge_v1",
  "prompt_set_version": "prompt_set_v1",
  "analyst_model": {
    "provider": "openai",
    "model": "gpt x",
    "temperature": 0.2
  },
  "case_model": {
    "provider": "openai",
    "model": "gpt x",
    "temperature": 0.3
  },
  "final_model": {
    "provider": "openai",
    "model": "gpt x",
    "temperature": 0.2
  }
}
```

## 16. First practical build slice

Do not start by building the whole cathedral. Start with one boring but decisive slice.

### Slice 1

1. fixtures and results ingestion
2. teams, venues, players, matches tables
3. lineup snapshots
4. injury snapshots
5. odds snapshots
6. feature set builder for one round
7. baseline winner model only
8. one basic dossier
9. one final decision agent only, no analyst swarm yet
10. validation + stored verdict

### Why this slice first

1. it proves the data spine
2. it proves snapshotting
3. it proves dossier construction
4. it proves the agent can make a structured verdict
5. it avoids building six agent layers before you know the plumbing works

Then expand to full analyst and case agent structure.

## 17. Suggested first ticket list

1. bootstrap repo, settings, DB session, Alembic
2. create core tables and migrations
3. implement AFL fixture ingestion
4. implement lineup ingestion and mapping review flow
5. implement injury ingestion and mapping review flow
6. implement odds ingestion and median implied probability logic
7. seed venue to BOM map and implement weather ingestion
8. build round run creator and snapshot capture workflow
9. build feature set v1 for winner baseline
10. train and persist winner baseline v1
11. build match dossier schema and builder
12. create final decision prompt and response validator
13. store agent step metadata and final verdict
14. build post result evaluation job
15. build simple round output API endpoint

## 18. API surfaces

### Internal endpoints

1. `GET /rounds/{round_id}/runs`
2. `GET /runs/{run_id}`
3. `GET /runs/{run_id}/matches/{match_id}`
4. `POST /rounds/{round_id}/run`
5. `POST /runs/{run_id}/evaluate`
6. `GET /seasons/{season_id}/summary`

### Why this is enough

This gives you:

1. manual triggering
2. run inspection
3. per match review
4. season level comparison

## 19. Testing strategy

### Unit tests

1. source parsers
2. entity mappers
3. feature builders
4. odds implied probability logic
5. dossier validation
6. agent output validation

### Integration tests

1. full round ingest
2. snapshot bundle creation
3. baseline prediction generation
4. dossier to agent verdict flow
5. post result evaluation

### Replay tests

1. fixed historical round with stored fixtures and snapshots
2. fixed prompt set and fixed agent config
3. stable schema validation and evaluation outputs

## 20. What must be immutable

1. snapshot rows once stored
2. run config for a completed run
3. prompt template text for a stored version
4. final stored verdict for a completed run

If something changes, create a new version. Human beings love editing history when a prediction looks stupid afterwards.

## 21. What can evolve safely

1. feature versions
2. baseline model versions
3. prompt set versions
4. agent structure
5. validation thresholds
6. reporting views

## 22. Final recommendation

Build this in three practical layers:

1. **Data and snapshots first**
2. **Baseline and dossier second**
3. **Full agent swarm third**

That order matters because an agentic forecaster without a proper data spine is just a sports pundit with a JSON addiction.

The schema above is enough to support:

1. reproducible round runs
2. agent and baseline comparison
3. historical replay
4. per match audit trails
5. future expansion into memory or extra markets later without wrecking the foundation

