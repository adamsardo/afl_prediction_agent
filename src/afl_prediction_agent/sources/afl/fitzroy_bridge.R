suppressPackageStartupMessages({
  library(jsonlite)
  library(fitzRoy)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("dataset argument is required")
}

dataset <- args[[1]]
payload_json <- if (length(args) >= 2) args[[2]] else "{}"
payload <- fromJSON(payload_json, simplifyVector = TRUE)

fetch_dataset <- function(dataset, payload) {
  source_name <- if (!is.null(payload$source)) payload$source else "AFL"
  if (dataset == "fixtures") {
    return(fetch_fixture(
      season = payload$season,
      round_number = payload$round_number,
      comp = if (!is.null(payload$comp)) payload$comp else "AFLM",
      source = source_name
    ))
  }
  if (dataset == "results") {
    return(fetch_results(
      season = payload$season,
      round_number = payload$round_number,
      comp = if (!is.null(payload$comp)) payload$comp else "AFLM",
      source = source_name
    ))
  }
  if (dataset == "lineups") {
    return(fetch_lineup(
      season = payload$season,
      round_number = payload$round_number,
      comp = if (!is.null(payload$comp)) payload$comp else "AFLM",
      source = source_name
    ))
  }
  if (dataset == "team_stats") {
    return(fetch_team_stats(
      season = payload$season,
      round_number = payload$round_number,
      comp = if (!is.null(payload$comp)) payload$comp else "AFLM",
      source = source_name
    ))
  }
  if (dataset == "player_stats") {
    return(fetch_player_stats(
      season = payload$season,
      round_number = payload$round_number,
      comp = if (!is.null(payload$comp)) payload$comp else "AFLM",
      source = source_name
    ))
  }
  if (dataset == "player_details") {
    return(fetch_player_details(
      season = payload$season,
      comp = if (!is.null(payload$comp)) payload$comp else "AFLM",
      source = source_name
    ))
  }
  stop(paste("Unsupported dataset:", dataset))
}

result <- fetch_dataset(dataset, payload)
cat(toJSON(result, dataframe = "rows", na = "null", auto_unbox = TRUE))

