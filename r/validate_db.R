#!/usr/bin/env Rscript
# Independent recomputation of forecast quality from the project SQLite database.
# Reads forecast_validation (written by Python after real results arrive), computes log loss,
# Brier score and margin error per forecast run, and writes them back to r_validation.
# Python then compares these numbers with its own and refuses any mismatch.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Expected the database path")
conn <- DBI::dbConnect(RSQLite::SQLite(), args[[1]])
on.exit(DBI::dbDisconnect(conn), add = TRUE)
v <- DBI::dbGetQuery(conn, "SELECT run_id, home_win_probability AS p, home_win AS y, projected_home_margin AS pm, actual_margin AS am FROM forecast_validation")
if (!nrow(v)) {
  message("No validated forecasts yet.")
  quit(status = 0)
}
if (any(!is.finite(v$p)) || any(v$p < 0 | v$p > 1) || !all(v$y %in% c(0, 1))) stop("Invalid validation rows")
v$p <- pmin(pmax(v$p, 1e-9), 1 - 1e-9)
out <- do.call(rbind, lapply(split(v, v$run_id), function(d) data.frame(
  run_id = d$run_id[[1]],
  computed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  games = nrow(d),
  log_loss = -mean(d$y * log(d$p) + (1 - d$y) * log(1 - d$p)),
  brier = mean((d$p - d$y)^2),
  margin_mae = if (all(is.na(d$pm))) NA_real_ else mean(abs(d$pm - d$am), na.rm = TRUE),
  stringsAsFactors = FALSE
)))
DBI::dbExecute(conn, "DELETE FROM r_validation")
DBI::dbWriteTable(conn, "r_validation", out, append = TRUE)
message(sprintf("R validated %d forecast runs", nrow(out)))
