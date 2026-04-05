from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from biopharma_agent.vnext.dashboard import build_dashboard_payload
from biopharma_agent.vnext.storage import LocalResearchStore


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui" / "pm_dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Tatetuck PM dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on.")
    parser.add_argument("--store-dir", default=".tatetuck_store", help="Path to the Tatetuck store directory.")
    return parser.parse_args()


def build_handler(store_dir: str):
    store = LocalResearchStore(store_dir)

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "TatetuckDashboard/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/dashboard":
                payload = build_dashboard_payload(store=store)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path in {"/", "/index.html"}:
                return self._serve_file(UI_DIR / "index.html")
            if parsed.path == "/styles.css":
                return self._serve_file(UI_DIR / "styles.css")
            if parsed.path == "/app.js":
                return self._serve_file(UI_DIR / "app.js")

            self.send_error(404, "Not found")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _serve_file(self, path: Path) -> None:
            if not path.exists():
                self.send_error(404, "Not found")
                return
            data = path.read_bytes()
            content_type, _ = mimetypes.guess_type(path.name)
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DashboardHandler


def main() -> None:
    args = parse_args()
    handler = build_handler(args.store_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Tatetuck PM dashboard available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
