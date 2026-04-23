# AFL Prediction Agent V1 Spec

## 1. Purpose

Build a personal AFL prediction framework that produces one locked agent verdict set for each round of the AFL men’s home and away season.

For every match in scope, the system must output:

1. agent selected winner
2. agent assigned win probability for both teams
3. agent predicted margin
4. agent confidence score
5. top drivers behind the call
6. concise, structured analyst summary
7. deterministic baseline references for comparison

The output is designed for personal tipping and betting support. It is a decision support system, not an auto betting system.

## 2. Product goal

The primary goal is to be as accurate as possible on match winners while also producing sharp, useful probabilities.

The secondary goal is to stress test and evaluate whether a structured multi agent reasoning system can produce reliable AFL forecasts from prepared match data.

The margin forecast is included as a meaningful secondary output that helps with line based thinking and overall confidence in the match call.

## 3. V1 scope

### In scope

1. AFL men’s competition only
2. Home and away season only
3. One prediction run per round
4. Predictions generated when official squads or lineups for the round are available
5. Deterministic data pipeline, feature generation, baseline modelling, and dossier construction
6. Agent based reasoning over structured match dossiers
7. Agent selected final winner, win probabilities, margin, and confidence
8. Timestamped raw data snapshots for reproducibility and later replay
9. Historical training and backtesting window from 2012 onward
10. Run metadata storage for every agentic prediction run

### Out of scope for v1

1. AFLW
2. Finals
3. Live or in game predictions
4. News and sentiment ingestion
5. Player prop markets
6. Total points, team totals, or line cover as primary outputs
7. Automated bet placement
8. Multi round scheduling logic beyond one round level run
9. Implicit memory across rounds
10. Self training from prior agent outputs

## 4. Locked decisions

1. Primary prediction targets are match winner and margin.
2. Operational cadence is one run per round.
3. Predictions are generated once, when official squads or lineups for the round are available.
4. Historical modelling window is 2012 onward, with stronger weighting on the most recent 5 seasons.
5. Raw source data is stored as timestamped snapshots.
6. V1 excludes finals.
7. V1 excludes news and sentiment.
8. ETL, feature generation, baseline probabilities, baseline margin, and dossier construction are deterministic.
9. The final public verdict for winner, win probability, margin, and confidence is produced by the final decision agent.
10. Deterministic baseline models are advisory inputs and benchmarks only. They do not directly decide the final public verdict.
11. Prompts, model version, inference settings, structured inputs, intermediate agent outputs, and final outputs are stored for audit history only.
12. Stored run metadata does not influence future predictions unless a memory layer is explicitly designed later.
13. There is no implicit memory across rounds in v1.

## 5. Core design principle

The system should borrow the useful part of a multi analyst framework without letting raw scraping or unstructured model improvisation drive the process.

That means:

1. deterministic layers turn raw data into structured match dossiers
2. specialist analyst agents reason over one slice of the dossier each
3. opposing case agents can argue for each side of the match
4. a final decision agent reviews the evidence and makes the verdict
5. every step is logged so the run can be audited, compared, and replayed later

## 6. Success criteria

### External success metric

1. round by round winner accuracy

### Internal quality metrics

1. Brier score for final agent win probabilities
2. log loss for final agent win probabilities
3. calibration by probability bucket
4. mean absolute error for final agent margin
5. root mean squared error for final agent margin
6. comparison against bookmaker implied probability baseline
7. comparison against Squiggle baseline
8. comparison against deterministic baseline model outputs
9. agent versus baseline disagreement performance

Winner accuracy is the headline metric. Probability quality and margin quality are internal diagnostics that prevent the system from sounding smart while being statistically soft.

## 7. Historical window and evaluation setup

### Historical scope

1. use data from 2012 onward
2. give greater modelling weight to the last 5 seasons
3. keep older seasons available for feature history and robustness checks
4. exclude finals from training and evaluation for v1 unless explicitly tagged for later experiments

### Evaluation protocol

1. use walk forward backtesting by round and season
2. every prediction must only use data that would have been available at the actual prediction lock time
3. use snapshot based odds, injuries, weather, and team selection data to avoid leakage
4. store every historical run with data versions, prompt versions, model versions, structured inputs, intermediate outputs, final outputs, and timestamp
5. treat each agent configuration as its own evaluation track
6. do not compare runs from materially different prompts or model versions as if they were identical systems

## 8. Source of truth by data type

### Primary sources

1. Fixtures, results, ladders, and lineups: official AFL source via fitzRoy
2. Historical team and player stats from 2012 onward: official AFL source via fitzRoy
3. Long range historical fallback and archive depth: AFL Tables
4. Injuries and availability: AFL.com injury list
5. Secondary cross check for injuries and team context: FootyWire
6. Weather: BOM
7. Odds: chosen odds feed using Australian bookmakers only
8. Benchmark comparison feed: Squiggle

### Source precedence rules

1. If AFL and a non official source disagree on fixture, result, or named lineups, trust the AFL source.
2. If AFL.com and FootyWire disagree on injury status, trust AFL.com.
3. If multiple bookmaker prices exist, calculate a market median head to head price at snapshot time.
4. If a non primary source is used as fallback, record the fallback source in metadata.

## 9. Operating rules for squads, injuries, and late outs

### Prediction lock

1. A round prediction run is created once the full set of matches in scope has official squads or lineups available.
2. If a full round is not available at the same time, use a fixed round lock rule. The system runs when the last standard team announcement required for that round is available.
3. Every prediction is tied to a single round snapshot timestamp.

### Squads and lineups

1. Only officially named players at lock time are treated as selected.
2. Lineup status is stored exactly as published, including interchange, emergencies, sub if available, and role labels if available.

### Injuries and availability

1. Injury status is pulled from AFL.com at lock time.
2. The system stores player status labels such as test, tbc, concussion protocol, estimated return, and any equivalent source wording.
3. A team availability score is created from selected players, unavailable players, and uncertainty around key players.

### Late outs

1. V1 does not rerun after late outs.
2. If a late out occurs after the round snapshot, store it as an audit event.
3. Mark the match with a material lineup change flag.
4. The official v1 prediction remains the locked pre match output.

## 10. High level system architecture

### Deterministic layers

1. ingestion layer
2. normalisation and entity resolution layer
3. snapshot storage layer
4. feature generation layer
5. deterministic baseline winner model
6. deterministic baseline margin model
7. structured dossier builder
8. output validator and consistency checker

### Agent layers

1. form analyst agent
2. selection analyst agent
3. venue and weather analyst agent
4. market analyst agent
5. home case agent
6. away case agent
7. final decision agent

### Guardrails

1. agents do not directly scrape raw sources in the decision step
2. agents only consume structured dossier inputs and approved prior agent outputs from the same run
3. the final decision agent can use deterministic baseline references but is not forced to follow them
4. a deterministic validator checks output schema and internal consistency before storage

## 11. Canonical entities and storage model

### Core entities

1. competitions
2. seasons
3. rounds
4. teams
5. players
6. venues
7. matches

### Snapshot entities

1. round_snapshots
2. match_snapshots
3. lineup_snapshots
4. injury_snapshots
5. weather_snapshots
6. odds_snapshots
7. source_fetch_logs

### Modelling entities

1. team_match_stats
2. player_match_stats
3. feature_sets
4. baseline_model_runs
5. baseline_predictions
6. benchmark_predictions
7. prediction_drivers
8. audit_events

### Agent run entities

1. agent_run_configs
2. prompt_templates
3. rendered_prompts
4. agent_inputs
5. agent_outputs
6. analyst_reports
7. debate_outputs
8. final_agent_verdicts
9. llm_usage_logs
10. validation_logs

### Snapshot and run rule

Every external fetch or agentic run artifact that can affect evaluation must be stored with:

1. source or component name
2. fetched or executed at timestamp
3. effective match or round scope
4. raw payload or structured parsed payload
5. parse or validation status
6. model run linkage where relevant
7. prompt version and rendered prompt where relevant
8. model provider and exact model version where relevant
9. temperature and key inference settings where relevant

## 12. Why run metadata is stored

Run metadata is stored for auditability, debugging, replay, benchmarking, and version comparison.

It is not predictive input by default.

### Stored for history and replay

1. prompt template id
2. full rendered prompt
3. model provider and exact model version
4. temperature and key inference settings
5. structured dossier inputs seen by each agent
6. intermediate agent outputs
7. final verdict
8. timestamps, usage, and validation events

### Explicit rule

1. these records are stored as run history only
2. they do not feed future match predictions in v1
3. there is no automatic memory or self influence across rounds
4. any future memory layer must be explicitly designed and separately versioned

## 13. Data ingestion requirements

### Fixture and result ingestion

1. ingest season, round, match, venue, home team, away team, start time, and final result
2. maintain stable external ids where available
3. map all sources into a single canonical match id

### Lineup ingestion

1. ingest official named players per team at lock time
2. preserve source structure where useful
3. record any missing or ambiguous player mappings for review

### Injury ingestion

1. ingest player status, injury note, expected return timing, and team association
2. resolve player names to canonical player ids
3. store unresolved entries separately for manual mapping

### Weather ingestion

1. map each venue to a BOM location or station mapping table
2. capture forecast conditions at prediction time for each match
3. store temperature, rain probability, expected rainfall, wind, and any available severe weather context

### Odds ingestion

1. store bookmaker level head to head prices at snapshot time
2. calculate market median implied probabilities after margin normalisation
3. retain bookmaker source data for later market movement analysis

### Squiggle ingestion

1. ingest Squiggle predictions and tips for benchmark comparison
2. keep them fully separate from source of truth match data

## 14. Feature set for v1

### Team performance features

1. rolling win rate over recent matches
2. rolling scoring for and scoring against
3. home and away splits
4. venue specific performance where sample size is reasonable
5. rest days since last match
6. travel burden proxy, such as interstate travel and short turnarounds
7. ladder position and rolling form momentum

### Team stats features

Use the best available official AFL team stats from fitzRoy, prioritised into rolling per match differentials where possible.

Candidate groups include:

1. inside 50 differential
2. clearance differential
3. contested possession differential
4. disposal efficiency or turnover related indicators where available
5. marks inside 50 or scoring shot creation indicators where available
6. tackle pressure indicators where available
7. hitouts or stoppage related indicators where relevant

### Selection and availability features

1. number of named changes from previous match
2. continuity score for named best 22 or closest equivalent
3. unavailable key player count
4. selected player experience and games played aggregates
5. team selection strength based on rolling player ratings

### Player rating aggregation

For each named player, compute a rolling player contribution rating from recent matches. Then aggregate named players into team level lineup strength features.

The initial version should be deliberately simple:

1. recent form window, such as last 5 to 10 matches played
2. position aware or role aware weighting only if the data is reliable enough
3. missing player penalty based on the gap between named replacement and absent player rating

### Weather features

1. rain probability band
2. expected rainfall band
3. wind strength band
4. temperature band
5. wet weather adjustment flag

### Market features

1. bookmaker median implied home win probability
2. bookmaker median implied away win probability
3. bookmaker confidence proxy from market spread
4. optional market movement signal only if snapshot history is mature enough

### Benchmark features

1. Squiggle win probability or tip where available
2. Squiggle predicted margin where available

## 15. Deterministic baseline models

Deterministic models remain part of the system, but they are no longer the final decision makers.

### Winner baseline

1. build a baseline probability model that outputs home and away win probabilities
2. start with regularised logistic regression for interpretability and sanity checks
3. test a stronger nonlinear option such as gradient boosted trees
4. keep the production baseline stable and versioned

### Margin baseline

1. build a baseline regression model for match margin
2. start with ridge regression or elastic net
3. test a stronger nonlinear option such as gradient boosted regression
4. keep the production baseline stable and versioned

### Role of baseline outputs

1. baseline probabilities and margin are included in the match dossier
2. they act as reference points for the agent system
3. they are benchmark tracks for later evaluation
4. they do not directly determine the final public verdict

## 16. Agentic reasoning framework

### Analyst agents

#### Form analyst

Summarises recent team form, scoring profile, opponent strength, and core statistical momentum.

#### Selection analyst

Summarises named changes, unavailable players, lineup continuity, and team selection strength.

#### Venue and weather analyst

Summarises venue effect, home ground edge, travel context, and likely weather impact.

#### Market analyst

Summarises bookmaker median probability, market confidence, and whether the baseline agrees or disagrees with the market.

### Case agents

#### Home case agent

Builds the strongest structured case for the home team using the dossier and analyst outputs.

#### Away case agent

Builds the strongest structured case for the away team using the dossier and analyst outputs.

### Final decision agent

The final decision agent is the source of the public verdict.

It must review:

1. match metadata
2. deterministic feature summary
3. baseline winner and margin references
4. analyst agent reports
5. home and away case arguments
6. uncertainty flags

It must output:

1. predicted winner
2. home win probability
3. away win probability
4. predicted margin
5. confidence score
6. top drivers
7. main uncertainty note
8. rationale summary

## 17. Final decision protocol and output constraints

The final decision agent is allowed to disagree with the deterministic baseline.

However, its output must satisfy hard consistency rules:

1. home and away probabilities must sum to 100 percent
2. predicted winner must match the higher win probability
3. predicted margin sign must align with the predicted winner
4. confidence must be expressed on a 0 to 100 scale
5. output must be schema valid JSON or an equivalent strongly typed structure

### Validation and retry

1. if the output violates hard consistency rules, the validator rejects it
2. the system can request one bounded correction pass from the final decision agent
3. all validation failures and retries must be stored

## 18. Structured match dossier

The deterministic system should produce a compact structured object for each match containing:

1. match metadata
2. baseline model probabilities and margin
3. team performance features
4. selection and availability features
5. venue and weather features
6. market features
7. Squiggle benchmark data
8. top quantitative drivers
9. uncertainty flags
10. source snapshot ids and timestamps

This dossier is the approved reasoning input for the agent system.

## 19. LLM summary and reasoning layer

The analyst agents, case agents, and final decision agent all operate on structured inputs only.

### Responsibilities

1. extract meaningful patterns from the dossier
2. argue for and against each side
3. make the final public verdict
4. explain why the system landed there

### Constraints

1. no direct access to raw scraping in the reasoning step
2. no external browsing in the reasoning step
3. no hidden memory from prior rounds
4. reasoning must stay anchored to the dossier and same run outputs

## 20. Output contract for each match

Each match prediction must contain:

1. season
2. round
3. match id
4. home team
5. away team
6. venue
7. prediction lock timestamp
8. agent predicted winner
9. agent home win probability
10. agent away win probability
11. agent predicted margin
12. agent confidence score
13. top drivers
14. analyst summaries
15. case agent summaries
16. final rationale summary
17. bookmaker median snapshot
18. Squiggle benchmark snapshot if available
19. deterministic baseline winner probability
20. deterministic baseline margin
21. final decision model version
22. prompt version set
23. feature set version
24. run id

## 21. Backtesting requirements

### Replay logic

1. recreate each historical round using only snapshot eligible information available at the original lock time
2. where full historical snapshot data does not yet exist, simulate it using the nearest valid historical source data and clearly label the replay as approximate
3. once live snapshotting begins, use only true stored snapshots for forward evaluation
4. replay agentic forecasts using fixed prompt versions and fixed model versions where possible

### Benchmark comparisons

1. compare against favourite based tipping from bookmaker median implied probability
2. compare against Squiggle tip or probability where available
3. compare against deterministic baseline winner and margin outputs
4. compare against naive home team baseline
5. compare against simple recent form baseline

### Reporting

1. overall season winner accuracy
2. winner accuracy by confidence band
3. margin error distribution
4. calibration plots by probability bucket
5. agreement and disagreement rates versus market, Squiggle, and deterministic baseline
6. performance by prompt version and model version

## 22. Operational workflow

### Pre round workflow

1. ingest upcoming fixtures
2. wait for official round selection lock rule
3. fetch and store lineup snapshot
4. fetch and store injury snapshot
5. fetch and store weather snapshot
6. fetch and store odds snapshot
7. generate canonical feature set
8. run deterministic baseline winner and margin models
9. build structured match dossiers
10. run analyst agents
11. run home and away case agents
12. run final decision agent
13. validate and store final outputs
14. persist full run metadata and final round prediction set

### Post round workflow

1. ingest final results
2. compute winner correctness and margin error for agent and baseline tracks
3. record benchmark comparisons
4. store any late out audit events
5. update rolling historical features for future rounds

## 23. Non functional requirements

1. every prediction must be reproducible from stored snapshots, prompt versions, and run metadata to the extent allowed by the underlying model provider
2. data lineage must be visible for every match prediction
3. source failures must degrade gracefully and mark affected features or summaries
4. unresolved player or venue mappings must not silently pass through as valid
5. a failed agent step must be logged clearly and isolated by stage
6. the system should be able to replay historical rounds under a fixed configuration
7. deterministic baseline outputs must remain available even if the final agent step fails

## 24. Failure handling

### Missing lineups

If official lineups are not available for a match at lock time, exclude the match from the round run and mark the reason.

### Missing injury data

Proceed with the run if lineups are available, but set an injury data missing flag in the dossier and in the final rationale.

### Missing weather data

Proceed with a neutral weather assumption and set a weather missing flag.

### Missing odds data

Proceed with the football only dossier, set a market data missing flag, and omit market analyst commentary if needed.

### Mapping failures

If player or venue mapping fails, log the error, quarantine the affected entity for review, and avoid pretending the data is clean.

### Agent failure

If an analyst or case agent fails, store the error and continue only if the minimum required dossier remains valid.

If the final decision agent fails completely, store the deterministic baseline outputs and mark the final agent verdict as unavailable rather than fabricating one.

## 25. Acceptance criteria for v1

V1 is ready when all of the following are true:

1. the system can ingest and normalise all in scope data sources for an AFL round
2. the system can store timestamped snapshots for lineups, injuries, weather, and odds
3. the system can generate one locked structured dossier for every eligible match in a round
4. the system can produce analyst reports, home and away cases, and a final agent verdict for every eligible match
5. each match output includes winner, win probabilities, margin, confidence, and top drivers
6. each run stores prompts, model metadata, structured inputs, intermediate outputs, and final outputs for audit history
7. the system can replay historical rounds from 2012 onward using the defined evaluation framework
8. benchmark comparisons against bookmaker favourite, Squiggle, and deterministic baseline are included
9. run versioning and prediction lineage are working end to end

## 26. Recommended build order

### Phase 1, data spine

1. canonical entities and schema
2. fixture and result ingestion
3. lineup ingestion
4. odds, injury, and weather snapshotting
5. source precedence and mapping logic

### Phase 2, deterministic backbone

1. baseline winner model
2. baseline margin model
3. walk forward backtest harness
4. benchmark reporting
5. structured dossier builder

### Phase 3, agent system

1. analyst agent prompts and schemas
2. home and away case agents
3. final decision agent
4. validator and retry logic
5. run metadata storage

### Phase 4, review layer

1. match view and round view outputs
2. historical comparison by agent config
3. prompt and model version reporting
4. disagreement analysis versus baseline and market

## 27. Remaining implementation choices

These do not block the spec but must be selected during build:

1. exact odds provider
2. exact BOM venue mapping method
3. exact official team announcement lock rule for each round edge case
4. exact player rating formula for lineup strength
5. exact baseline model choice after evaluation
6. exact agent prompt structure and response schemas
7. exact model provider and temperature policy for the agent system

## 28. Final v1 statement

V1 is a round based AFL men’s match prediction system that produces locked pre match forecasts for winner and margin through a structured agentic reasoning process.

Deterministic systems prepare the data, create features, generate baseline references, and package a match dossier.

Specialist agents analyse that dossier, argue each side, and a final decision agent makes the public verdict.

Every run is logged in detail for auditability, replay, and evaluation, but prior run metadata does not automatically influence future predictions.

