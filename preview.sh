#!/usr/bin/env bash
# Serve the site locally for preview before merging main -> live.
# Usage: ./preview.sh [port]
port="${1:-8000}"
echo "Site is running at http://localhost:${port}"
echo "When ready to publish, merge main -> live with:"
echo "  git checkout live && git merge main && git push origin live && git checkout main"
exec python3 -m http.server "$port"
