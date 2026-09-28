#!/usr/bin/env bash
# Serve the site locally for preview before merging main -> live.
# Usage: ./preview.sh [port]
exec python3 -m http.server "${1:-8000}"
