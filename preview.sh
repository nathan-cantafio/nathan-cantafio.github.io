#!/usr/bin/env bash
# Serve the site locally for preview before merging main -> live.
# Usage: ./preview.sh [port]
port="${1:-8000}"
echo "Site is running at http://localhost:${port}"
echo "When ready to publish, merge main -> live with:"
echo "  git checkout live && git merge main && git push origin live && git checkout main"
exec python3 -c '
import http.server, sys
class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
http.server.test(HandlerClass=NoCache, port=int(sys.argv[1]))
' "$port"
