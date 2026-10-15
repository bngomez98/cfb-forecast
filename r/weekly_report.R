#!/usr/bin/env Rscript
# Print the latest published forecast board from the project SQLite database.
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1]] else "."
source(file.path(root, "r", "data.R"))
database <- Sys.getenv("CFB_DATABASE", file.path(root, "artifacts", "cfb_forecast.sqlite"))
conn <- open_cfb_db(database)
on.exit(close_cfb_db(conn), add = TRUE)
board <- read_forecast_log(conn)
if (!nrow(board)) {
  message("No forecast log rows are stored yet.")
  quit(status = 0)
}
latest <- board[board$logged_at == max(board$logged_at), ]
write.csv(latest, file.path(root, "artifacts", "logs", "r_forecast_report.csv"), row.names = FALSE)
print(latest[, c("kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score", "winner", "confidence")])
