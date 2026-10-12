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

#' Close a CFB Forecast SQLite connection.
#' @export
close_cfb_db <- function(conn) DBI::dbDisconnect(conn)
