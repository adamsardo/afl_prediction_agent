# Plan: Implement the Full Source Connector Layer

## Summary
- Build a real `src/afl_prediction_agent/sources/` layer that covers every source named in the spec: official AFL via `fitzRoy`, AFL Tables, AFL.com injuries, FootyWire, BOM, The Odds API, and Squiggle.
- Keep Python as the system runtime and DB owner. Use a narrow `Rscript` bridge only for `fitzRoy`.
- Use the existing schema for fetch logs, mappings, snapshots, stats, and benchmarks. Do not add new core source tables in this pass.
- Lock the open source decisions as:
  - `fitzRoy`: hybrid R bridge
  - odds: The Odds API
  - BOM mapping: curated venue map

## Implementation Changes
- Add `sources/common` with:
  - typed normalized records for fixtures, lineups, injuries, weather, odds, benchmarks, team stats, and player stats
  - a `FetchEnvelope` carrying `source_name`, `fetched_at`, `request_meta`, `response_meta`, and `raw_payload`
  - shared HTTP client behavior: timeout, retries, rate limiting, browser headers where needed
  - mapping helpers that read/write `external_id_mappings` and emit unresolved review reports instead of guessing IDs
  - centralized fetch logging into `source_fetch_logs`

- Add a proper ingestion CLI surface under `afl-agent ingest`:
  - `fixtures`
  - `results`
  - `lineups`
  - `stats`
  - `injuries`
  - `weather`
  - `odds`
  - `benchmarks`
  - `snapshot-round`
  - keep `run-round` able to call the pre-round fetch flow rather than assuming data is already present

- Official AFL / `fitzRoy`:
  - implement a repo-local R bridge invoked by Python with `subprocess`
  - use `fitzRoy` `source="AFL"` for fixtures, results, ladders, lineups, team stats, player stats, and player details where needed for mapping
  - make official fixtures/results/lineups the blocking inputs for current-round runs
  - persist official stats as `source_name="afl_official"`

- AFL Tables:
  - implement a separate `AflTablesConnector`, backed by `fitzRoy` `source="afltables"` for the initial implementation
  - use it for archive depth, historical backfills, and explicit gap-filling only
  - never let AFL Tables overwrite official current fixtures, results, or lineups
  - persist archive stats as `source_name="afl_tables"`

- AFL.com injuries:
  - build a primary HTML connector for the current injury page/article structure
  - extract team, player, status, injury note, estimated return text, and preserve original wording in payload
  - store one immutable injury snapshot per fetch as `source_name="afl_com"`

- FootyWire:
  - build a browser-header-based HTML connector, because generic requests currently return `406`
  - normalize the injury list into the same injury schema
  - store it as a secondary injury snapshot source and compare it against AFL.com
  - never override AFL.com; discrepancies become audit events and uncertainty flags
  - use FootyWire only for injury cross-checks and entity-resolution support, not for current source-of-truth match data

- BOM:
  - do not scrape BOM website HTML
  - use sanctioned BOM Weather Data Services / anonymous FTP-style products only
  - maintain a curated venue mapping seed with venue -> BOM station/location/product codes, lat/lon, timezone, and fallback area mapping
  - fetch observation data for current temperature, wind, and weather text
  - fetch forecast products for rain probability, rainfall, and severe-weather context at the match timeslot
  - if a sanctioned feed does not expose a requested field, store `null` and log a coverage event rather than scraping blocked pages

- Odds / The Odds API:
  - implement a dedicated connector using `sport=aussierules_afl`, `markets=h2h`, `oddsFormat=decimal`, and an explicit AU bookmaker allowlist
  - store bookmaker-level rows in `odds_snapshot_books`
  - compute median prices and normalized implied probabilities in Python before writing `odds_snapshots`
  - support both current odds and historical odds replay when the account plan allows it
  - treat missing API key or exhausted credits as non-blocking source failures with audit logging

- Squiggle:
  - implement direct HTTP ingestion of `games`, `tips`, and `sources`
  - match Squiggle games to canonical matches by season, round, home, away, and date
  - store the configured benchmark model in `benchmark_predictions`
  - archive all returned tip/model payloads; default benchmark source is `aggregate` if present, otherwise the configured source

- Precedence and orchestration:
  - current-round canonical match data always comes from official AFL / `fitzRoy`
  - AFL Tables is fallback/archive only
  - AFL.com is the primary injury source; FootyWire is secondary
  - pre-round snapshot order is: official lineups -> AFL.com injuries -> FootyWire cross-check -> BOM weather -> odds -> Squiggle
  - unresolved players, teams, and venues must surface through review commands, never silent inference

## Public Interfaces / Config
- Add settings/env vars:
  - `AFL_AGENT_FITZROY_RSCRIPT_BIN`
  - `AFL_AGENT_FITZROY_TIMEOUT_SECONDS`
  - `AFL_AGENT_SOURCE_HTTP_TIMEOUT_SECONDS`
  - `AFL_AGENT_SOURCE_RETRY_ATTEMPTS`
  - `AFL_AGENT_ODDS_API_KEY`
  - `AFL_AGENT_ODDS_AU_BOOKMAKERS`
  - `AFL_AGENT_SQUIGGLE_BENCHMARK_SOURCE`
- Add normalized connector output types:
  - `NormalizedFixtureMatch`
  - `NormalizedLineupSnapshot`
  - `NormalizedInjurySnapshot`
  - `NormalizedWeatherSnapshot`
  - `NormalizedOddsSnapshot`
  - `NormalizedBenchmarkPrediction`
  - `NormalizedTeamMatchStats`
  - `NormalizedPlayerMatchStats`
- Add seed/config assets:
  - curated BOM venue mapping file
  - AU bookmaker allowlist
  - common alias seeds for team/player normalization where needed

## Test Plan
- Unit tests:
  - `fitzRoy` bridge command serialization and subprocess failure handling
  - AFL.com injury parser
  - FootyWire parser with browser headers
  - BOM feed parser and venue mapping resolution
  - The Odds API AU bookmaker filtering and odds normalization
  - Squiggle response parsing and benchmark source selection
  - precedence rules: official AFL > AFL Tables, AFL.com > FootyWire
  - unresolved mapping reporting

- Integration tests:
  - official AFL round ingest end-to-end through existing ingestion services
  - historical stats backfill with AFL Tables fallback
  - full pre-round snapshot capture writes lineups, injuries, weather, odds, and benchmark rows
  - `run-round` with fetch enabled succeeds when required sources are present and degrades cleanly when optional sources fail
  - replay flow uses stored historical snapshots without re-fetching

- Manual acceptance:
  - official round data loads from `fitzRoy`
  - both injury sources store snapshots and discrepancy audit events
  - BOM weather loads via sanctioned feeds, not blocked HTML scraping
  - odds snapshots produce bookmaker rows plus medians
  - Squiggle benchmark rows are stored for the configured benchmark source

## Assumptions and Defaults
- The repo stays Python-first; no separate ingestion worker/service is introduced.
- `fitzRoy` runs through a hybrid `Rscript` bridge, so local/CI environments will need R plus the required R packages installed.
- The only source that requires a new secret is The Odds API, via `AFL_AGENT_ODDS_API_KEY`.
- BOM will use [Weather Data Services](https://www.bom.gov.au/catalogue/data-feeds.shtml) and sanctioned automated-access products only; if those feeds are insufficient for a field, the field remains null in v1.
- Source basis:
  - [fitzRoy docs](https://jimmyday12.github.io/fitzRoy/articles/main-fetch-functions.html)
  - [AFL injury page](https://www.afl.com.au/news/injury-news)
  - [FootyWire injury list](https://www.footywire.com/afl/footy/injury_list)
  - [BOM Weather Data Services](https://www.bom.gov.au/catalogue/data-feeds.shtml)
  - [The Odds API AFL docs](https://the-odds-api.com/sports/afl-odds.html)
  - [Squiggle API](https://api.squiggle.com.au/)
