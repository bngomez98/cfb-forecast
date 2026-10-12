#!/usr/bin/env Rscript
# Source-analysis stage of the weekly CFB forecast cycle.
#
# Runs BEFORE any Python forecasting: audits every schedule_*.csv and
# adv_team_*.* source in the data directory, fits a ridge closed-form model on
# completed games strictly before the as-of cutoff (point-in-time, no future
# results), and writes one JSON document to stdout that the Python pipeline
# consumes verbatim (model coefficients + data-quality flags).
#
# Usage: Rscript r/analyze_sources.R <data_dir> <as_of_iso> <season> [prev_model_json]
suppressWarnings(suppressMessages({
  ok_adv <- requireNamespace("arrow", quietly = TRUE)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: analyze_sources.R <data_dir> <as_of> <season> [prev_model_json]")
data_dir <- args[1]; as_of <- args[2]; season <- as.numeric(args[3])
prev_path <- if (length(args) >= 4) args[4] else ""
as_of_ts <- as.POSIXct(as_of, tz = "UTC")
if (is.na(as_of_ts)) stop("unparseable as_of")

files <- list.files(data_dir, pattern = "^schedule_.*\\.(csv|parquet)$", full.names = TRUE)
if (!length(files)) stop("no schedule sources found")
parts <- lapply(files, function(f) read.csv(f, stringsAsFactors = FALSE))
common <- Reduce(intersect, lapply(parts, names))
sched <- do.call(rbind, lapply(parts, function(d) d[, common, drop = FALSE]))

num <- function(x) suppressWarnings(as.numeric(x))
sched$game_id <- num(sched$game_id); sched$season <- num(sched$season)
sched$week <- num(sched$week); sched$start_date <- as.POSIXct(sched$start_date, tz = "UTC")
sched$home_points <- num(sched$home_points); sched$away_points <- num(sched$away_points)
sched$completed <- tolower(as.character(sched$completed)) %in% c("true", "1")
fbs_col <- if ("home_division" %in% names(sched)) {
  as.character(sched$home_division) == "fbs" & as.character(sched$away_division) == "fbs"
} else if ("fbs_game" %in% names(sched)) {
  tolower(as.character(sched$fbs_game)) %in% c("true", "1")
} else stop("schedule lacks FBS identification columns")
sched <- sched[fbs_col, ]
sched <- sched[!is.na(sched$game_id) & !is.na(sched$start_date), ]
# the 2026 export reuses ESPN game IDs across seasons; key on season+game_id
dup_key <- paste(sched$season, sched$game_id)
duplicates <- sum(duplicated(dup_key))
missing_scores <- sum(sched$completed & (is.na(sched$home_points) | is.na(sched$away_points)))
future_completed <- sum(!is.na(sched$start_date) & sched$completed & sched$start_date > as_of_ts)
sched$result_usable <- sched$completed & !is.na(sched$home_points) & !is.na(sched$away_points) &
  sched$home_points != sched$away_points
# conservative availability proxy mirrors the Python engine: kickoff + 8h
sched$available_at <- sched$start_date + 8 * 3600

adv_files <- list.files(data_dir, pattern = "^adv_team_.*\\.(csv|parquet)$", full.names = TRUE)
adv <- NULL; adv_available <- FALSE
read_adv <- function(f) {
  if (grepl("\\.parquet$", f) && ok_adv) return(arrow::read_parquet(f) |> as.data.frame())
  tryCatch(read.csv(f, stringsAsFactors = FALSE), error = function(e) NULL)
}
for (f in adv_files) { d <- read_adv(f); if (!is.null(d)) adv <- rbind(adv, d) }
metrics <- c("points", "yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate")
summary_cols <- c("yardsPerPlay", "yardsPerPass", "yardsPerRush", "firstDownsCreatedRate")
if (!is.null(adv) && all(summary_cols %in% names(adv))) {
  adv_available <- TRUE
  adv$game_id <- num(adv$game_id); adv$pos_team_id <- num(adv$pos_team_id)
  names(adv)[match(summary_cols, names(adv))] <- metrics[-1]
}

# ---- point-in-time training set: final results available by the cutoff ----
train <- sched[sched$result_usable & sched$available_at <= as_of_ts & sched$season < season, ]
n_train <- nrow(train)
features_ok <- adv_available && n_train > 50
coef_json <- "{}"; rmse <- NA; fit_seasons <- NA_character_
if (features_ok) {
  idx <- match(paste(train$game_id, train$home_id), paste(adv$game_id, adv$pos_team_id))
  home <- adv[idx, metrics]
  idx2 <- match(paste(train$game_id, train$away_id), paste(adv$game_id, adv$pos_team_id))
  away <- adv[idx2, metrics]
  complete <- stats::complete.cases(home, away, train$home_points, train$away_points)
  home <- home[complete, ]; away <- away[complete, ]
  hp <- train$home_points[complete]; ap <- train$away_points[complete]
  # rows: offense features minus defense features, plus venue sign
  X <- rbind(cbind(home - away, 1), cbind(away - home, -1))
  y <- c(hp, ap)
  lambda <- 10
  XtX <- crossprod(X) + lambda * diag(ncol(X))
  beta <- tryCatch(solve(XtX, crossprod(X, y)), error = function(e) NULL)
  if (!is.null(beta) && all(is.finite(beta))) {
    pred <- as.matrix(X) %*% beta
    rmse <- sqrt(mean((pred[, 1] - y)^2))
    coef_json <- sprintf("{%s}", paste(sprintf("\"%s\": %.10g",
      paste0(metrics, "_diff"), beta[1:5]), sprintf("\"venue\": %.10g", beta[6]), sep = ","))
    fit_seasons <- paste(sort(unique(floor(train$season[complete]))), collapse = ",")
  } else features_ok <- FALSE
}

prev_note <- "none"
if (nzchar(prev_path) && file.exists(prev_path)) prev_note <- "compared"

cat(sprintf(
  '{"generated_by":"r/analyze_sources.R","as_of":"%s","season":%d,"schedule_files":%d,"games_total":%d,"duplicate_game_ids":%d,"missing_completed_scores":%d,"future_completed_games":%d,"adv_source_available":%s,"training_games_pit":%d,"ridge_fit":%s,"ridge_rmse":%s,"fit_seasons":"%s","previous_model":"%s"}',
  format(as_of_ts, "%Y-%m-%dT%H:%M:%SZ"), as.numeric(season), length(files), nrow(sched),
  duplicates, missing_scores, future_completed, tolower(as.character(adv_available)),
  n_train, tolower(as.character(features_ok)),
  if (is.na(rmse)) "null" else sprintf("%.4f", rmse),
  ifelse(is.na(fit_seasons), "", fit_seasons), prev_note))
cat("\n")
if (duplicates > 0) stop(sprintf("%d duplicate game IDs in sources", duplicates))
