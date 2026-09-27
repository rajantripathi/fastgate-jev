#!/usr/bin/env bash
# Usage: bash scripts/push_to_github.sh [repo-name]
# Creates github.com/rajantripathi/<repo-name> (public) and pushes this folder.
set -euo pipefail
REPO="${1:-fastgate-jev}"
OWNER="rajantripathi"
cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  git init -q
  git add .
  git commit -qm "FastGate: Jev System One decision layer for multilingual RAG (EN/UZ/RU)"
  git branch -M main
fi

if command -v gh >/dev/null 2>&1; then
  gh repo create "$OWNER/$REPO" --public --source=. --remote=origin --push \
    --description "Jev (TypeSafe AI) as a System One decision layer for a multilingual EN/UZ/RU RAG helpdesk, with an independent benchmark"
else
  echo "GitHub CLI not found. Create an empty repo at https://github.com/new named $REPO, then press Enter."
  read -r
  git remote add origin "https://github.com/$OWNER/$REPO.git" 2>/dev/null || true
  git push -u origin main
fi
echo "Done: https://github.com/$OWNER/$REPO"
