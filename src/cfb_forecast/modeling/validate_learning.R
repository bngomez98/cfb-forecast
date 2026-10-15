args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Expected predictions CSV and output CSV")
data <- read.csv(args[1])
if (nrow(data) < 30) stop("Insufficient chronological validation games")
if (anyDuplicated(data$game_id)) stop("Duplicate validation games")
if (!all(data$home_win %in% c(0, 1))) stop("Invalid outcome labels")
p <- data$candidate_p
y <- data$home_win
if (any(!is.finite(p)) || any(p < 0 | p > 1)) stop("Invalid probabilities")
p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
result <- data.frame(games = length(y), log_loss = -mean(y * log(p) + (1 - y) * log(1 - p)), brier = mean((p - y)^2))
write.csv(result, args[2], row.names = FALSE)
