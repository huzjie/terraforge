"""Zero-dependency HTTP server using stdlib http.server."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..utils.logging import get_logger
from .router import RequestRouter

log = get_logger("terraforge.serving")


def _make_handler(router):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.info("%s %s", self.command, self.path)

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                self._send(200, router.route("/health", {}))
            elif self.path.split("?")[0] == "/v1/models":
                self._send(200, router.route("/v1/models", {}))
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = {}
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError:
                    self._send(400, {"error": "invalid json"})
                    return
            path = self.path.split("?")[0]
            try:
                self._send(200, router.route(path, body))
            except Exception as e:  # pragma: no cover
                self._send(500, {"error": str(e)})

    return Handler


def serve(cfg):
    from ..backends import build_backend
    backend = build_backend(cfg.backend, cfg)
    router = RequestRouter(backend, cfg)
    host, port = cfg.serving.host, cfg.serving.port
    server = HTTPServer((host, port), _make_handler(router))
    log.info(f"serving {backend.name} backend at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from terraforge.config import load_config
    serve(load_config(sys.argv[1] if len(sys.argv) > 1 else None))
