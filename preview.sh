#!/usr/bin/env bash
# Serve the site locally for preview before merging main -> live.
# Usage: ./preview.sh [port]
#
# The header, sidebar and post lists are stamped into the pages by build.js,
# so this rebuilds first. After editing header.html, sidebar.html or a post,
# run `node build.js` (or restart this script) to see the change.
cd "$(dirname "$0")" || exit 1
port="${1:-8000}"
node build.js || echo "preview.sh: build failed; serving the pages as they are."
echo "Site is running at http://localhost:${port}"
echo "When ready to publish, merge main -> live with:"
echo "  git checkout live && git merge main && git push origin live && git checkout main"
exec python3 -c '
import http.server, os, sys
class Preview(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    # Show 404.html for missing pages, the way GitHub Pages does.
    def send_error(self, code, message=None, explain=None):
        if code != 404 or not os.path.exists("404.html"):
            return super().send_error(code, message, explain)
        with open("404.html", "rb") as f:
            body = f.read()
        self.send_response(404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
http.server.test(HandlerClass=Preview, port=int(sys.argv[1]))
' "$port"
