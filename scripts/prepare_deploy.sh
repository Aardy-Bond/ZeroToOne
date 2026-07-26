#!/usr/bin/env bash
# Wipe local junk before `databricks workspace import-dir` / Apps deploy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Cleaning caches and generated artifacts..."
rm -rf output __pycache__ .pytest_cache .mypy_cache .ruff_cache
mkdir -p output projects
touch output/.gitkeep projects/.gitkeep
rm -f projects/*.db projects/*.db-wal projects/*.db-shm projects/*.db-journal
find . -path ./.venv -prune -o -path './.git' -prune -o -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . -path ./.venv -prune -o -path './.git' -prune -o -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true

if [[ -f .env ]]; then
  echo "NOTE: .env exists locally — do NOT upload it. Set OPENAI_API_KEY in the Databricks App env UI."
fi

echo "Ready. Deploy with app.yaml → streamlit run src/dashboard/app.py"
echo "Example:"
echo "  databricks workspace import-dir . /Workspace/Users/<you>/apps/anubhuti --overwrite"
echo "  (exclude .venv / .git / .env manually, or zip a clean tree first)"
