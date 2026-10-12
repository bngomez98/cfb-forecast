#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
cd "$PROJECT_DIR"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo 'Not a Git repository' >&2; exit 1; }

echo '--- Repository'
git rev-parse --show-toplevel
echo '--- Branch'
git branch --show-current
echo '--- Status'
git status --short
echo '--- Remote'
git remote -v || true

echo
echo 'If no remote is listed:'
echo 'git remote add origin <repository-url>'
echo 'git push -u origin main'
