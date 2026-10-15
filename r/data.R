#' Open a CFB Forecast SQLite database.
#' @param path SQLite database path.
#' @return A DBI connection.
#' @export
open_cfb_db <- function(path = "artifacts/cfb_forecast.sqlite") {
  DBI::dbConnect(RSQLite::SQLite(), path)
}

#' Read forecasts for a season.
#' @export
read_forecasts <- function(conn, season = NULL) {
  sql <- "SELECT * FROM forecasts"
  if (!is.null(season)) sql <- paste0(sql, " WHERE season = ", as.integer(season))
  sql <- paste0(sql, " ORDER BY kickoff")
  DBI::dbGetQuery(conn, sql)
}

#' Read latest watchdog events.
#' @export
read_watchdog_events <- function(conn, limit = 100L) {
  DBI::dbGetQuery(conn, paste0(
    "SELECT * FROM watchdog_events ORDER BY observed_at DESC LIMIT ", as.integer(limit)
  ))
}

#' Read the published forecast log.
#' @export
read_forecast_log <- function(conn, season = NULL) {
  sql <- "SELECT run_id, season, game_id, game_date, kickoff, away_team, home_team, expected_away_score, expected_home_score, winner, confidence, home_win_probability, logged_at FROM forecast_log"
  if (!is.null(season)) sql <- paste0(sql, " WHERE season = ", as.integer(season))
  sql <- paste0(sql, " ORDER BY logged_at, kickoff")
  DBI::dbGetQuery(conn, sql)
}

#' Read the source catalog stored in SQLite.
#' @export
read_sources <- function(conn) {
  DBI::dbGetQuery(conn, "SELECT source_key, provider, source_url, schema_version FROM source_registry ORDER BY source_key")
}

#' Close a CFB Forecast SQLite connection.
#' @export
close_cfb_db <- function(conn) DBI::dbDisconnect(conn)
