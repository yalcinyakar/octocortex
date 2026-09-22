from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlparse

from octocortex.simulation.environment import OctoSimulation


SIMULATION = OctoSimulation()
STATIC = files("octocortex").joinpath("static")


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _asset(self, name: str, content_type: str) -> None:
        try:
            payload = STATIC.joinpath(name).read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._asset("index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._asset("app.js", "text/javascript; charset=utf-8")
        elif path == "/styles.css":
            self._asset("styles.css", "text/css; charset=utf-8")
        elif path == "/api/state":
            self._json(SIMULATION.snapshot())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/step":
            self._json(SIMULATION.step())
        elif path == "/api/reset":
            self._json(SIMULATION.reset())
        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("OctoCortex running at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

