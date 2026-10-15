#!/usr/bin/env bash
set -Eeuo pipefail

# Clone or safely update the forecasting repository in WSL.
# First clone:
#   CFB_REPO_URL=https://github.com/OWNER/REPOSITORY.git bash scripts/setup/git_setup.sh
# Existing checkout:
#   bash scripts/setup/git_setup.sh

REPO_URL="${CFB_REPO_URL:-}"
REPO_DIR="${CFB_REPO_DIR:-$HOME/cfb_forecast}"
BRANCH="${CFB_BRANCH:-main}"

if [[ -d "$REPO_DIR/.git" ]]; then
  cd "$REPO_DIR"
  echo "Existing Git repository: $REPO_DIR"
  git fetch --all --prune
  current_branch="$(git branch --show-current)"
  if [[ -z "$(git status --porcelain)" ]]; then
    if git show-ref --verify --quiet "refs/remotes/origin/$current_branch"; then
      git pull --ff-only origin "$current_branch"
    else
      echo "No matching origin branch; leaving checkout unchanged."
    fi
  else
    echo "Local changes detected; fetch completed but pull was skipped."
    git status --short
  fi
else
  [[ -n "$REPO_URL" ]] || {
    echo "No repository exists at: $REPO_DIR"
    echo "Provide the GitHub URL on the first run:"
    echo "CFB_REPO_URL=https://github.com/OWNER/REPOSITORY.git bash scripts/setup/git_setup.sh"
    exit 2
  }
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo
echo "Repository: $(git rev-parse --show-toplevel)"
echo "Branch:     $(git branch --show-current)"
echo "Commit:     $(git rev-parse --short HEAD)"
echo "Remote:     $(git remote get-url origin 2>/dev/null || echo 'not configured')"
echo
git status --short
