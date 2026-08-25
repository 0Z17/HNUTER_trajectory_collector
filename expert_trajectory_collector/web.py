"""Dependency-free HTTP transport for the expert collector service."""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .contracts import CollectorError
from .service import ExpertCollectorService


DEFAULT_WEB_ROOT = Path(__file__).resolve().parent / "web_assets"
MAX_REQUEST_BYTES = 2_000_000


class ExpertCollectorHandler(SimpleHTTPRequestHandler):
    """Thin JSON/static transport; all use-case logic lives in the service."""

    service: ExpertCollectorService

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path == "/api/tasks":
            self._json_response(200, self.service.list_tasks())
            return
        if path == "/api/health":
            self._json_response(200, {
                "status": "ok",
                "service": "HNUTER Expert Trajectory Studio",
            })
            return
        super().do_GET()

    def _request_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body must be between 1 byte and 2 MB")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        operations = {
            "/api/generate": self.service.generate,
            "/api/validate": self.service.validate,
            "/api/batch": self.service.batch,
            "/api/experts": self.service.collect,
            "/api/collection": self.service.build_collection_record,
        }
        operation = operations.get(urlsplit(self.path).path)
        if operation is None:
            self._json_response(404, {"error": "not found"})
            return
        try:
            self._json_response(200, operation(self._request_payload()))
        except (
            CollectorError, ValueError, KeyError, TypeError, RuntimeError,
            ImportError, json.JSONDecodeError,
        ) as error:
            self._json_response(400, {"error": str(error)})


def make_handler(
    service: ExpertCollectorService,
    web_root: Path = DEFAULT_WEB_ROOT,
) -> type[ExpertCollectorHandler]:
    if not web_root.is_dir():
        raise FileNotFoundError(f"Web assets not found: {web_root}")

    class BoundExpertCollectorHandler(ExpertCollectorHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(web_root), **kwargs)

    BoundExpertCollectorHandler.service = service
    return BoundExpertCollectorHandler


def serve(
    service: ExpertCollectorService, *, host: str = "127.0.0.1",
    port: int = 8765, web_root: Path = DEFAULT_WEB_ROOT,
) -> None:
    handler = make_handler(service, web_root)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"HNUTER Expert Trajectory Studio: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
