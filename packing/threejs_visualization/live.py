"""Threaded HTTP server for live Three.js visualization frames."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .assets import RuntimeAssets, validated_runtime_assets
from .frame import VisualizationFrame


class ThreeLiveServer:
    def __init__(
        self,
        plot_dir: str | Path,
        port: int = 8765,
        bind_host: str = "127.0.0.1",
        public_host: str = "127.0.0.1",
        poll_ms: int = 500,
        log_requests: bool = False,
        assets_root: str | Path | None = None,
    ) -> None:
        self.plot_dir = Path(plot_dir)
        self.port = int(port)
        self.bind_host = bind_host
        self.public_host = public_host
        self.poll_ms = max(1, int(poll_ms))
        self.log_requests = log_requests
        self.assets_root = assets_root
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._frames: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._session_id = str(time.time_ns())

    @property
    def url(self) -> str:
        return f"http://{self.public_host}:{self.port}/index.html"

    def url_for(self, path: str) -> str:
        return f"http://{self.public_host}:{self.port}/{path.lstrip('/')}"

    def start(self) -> str:
        with self._lifecycle_lock:
            return self._start_locked()

    def _start_locked(self) -> str:
        if self._httpd is not None:
            return self.url

        assets = validated_runtime_assets(self.assets_root)
        owner = self

        class LiveHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._response_started = False
                try:
                    owner._dispatch_get(self, assets)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as error:
                    if owner.log_requests:
                        self.log_error("request failed: %r", error)
                    if self._response_started:
                        # A second status line would corrupt an already-started
                        # HTTP response. Close instead and let the client retry.
                        self.close_connection = True
                        return
                    try:
                        owner._send_empty(self, 500)
                    except (BrokenPipeError, ConnectionResetError):
                        return

            def send_response(self, code: int, message: str | None = None) -> None:
                self._response_started = True
                super().send_response(code, message)

            def end_headers(self) -> None:
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def log_message(self, format: str, *args: object) -> None:
                if owner.log_requests:
                    super().log_message(format, *args)

        httpd = ThreadingHTTPServer((self.bind_host, self.port), LiveHandler)
        httpd.daemon_threads = True
        self.port = int(httpd.server_address[1])
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._httpd = httpd
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._httpd = None
            self._thread = None
            httpd.server_close()
            raise
        return self.url

    def push(self, frame: VisualizationFrame | dict[str, Any]) -> None:
        if not isinstance(frame, (VisualizationFrame, dict)):
            raise TypeError("frame must be a VisualizationFrame or dict")
        payload = frame.to_dict() if isinstance(frame, VisualizationFrame) else frame
        try:
            # The JSON round trip both validates the wire representation and
            # detaches queued data from caller-owned mutable containers.
            payload = json.loads(json.dumps(payload, allow_nan=False))
        except TypeError as error:
            raise TypeError(f"frame must be JSON serializable: {error}") from error
        except ValueError as error:
            if "Out of range float values" in str(error):
                raise ValueError(f"frame numbers must be finite: {error}") from error
            raise TypeError(f"frame must be JSON serializable: {error}") from error

        queued = {
            "session": self._session_id,
            "version": time.time_ns(),
            "frame": payload,
        }
        with self._lock:
            queued["index"] = len(self._frames)
            self._frames.append(queued)

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        httpd = self._httpd
        thread = self._thread
        if httpd is None:
            return
        self._httpd = None
        self._thread = None
        httpd.shutdown()
        httpd.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _dispatch_get(
        self, handler: BaseHTTPRequestHandler, assets: RuntimeAssets
    ) -> None:
        parsed = urlparse(handler.path)
        if parsed.path == "/frame.json":
            self._serve_frame(handler, parsed.query)
            return
        if parsed.path in ("/", "/index.html"):
            template = assets.live_html.read_text(encoding="utf-8")
            configured = template.replace(
                "PackingThree.startLive();",
                f"PackingThree.startLive({{interval: {self.poll_ms}}});",
                1,
            )
            self._send_bytes(
                handler,
                configured.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return

        routes = {
            "/static/style.css": (assets.style_css, "text/css; charset=utf-8"),
            "/static/renderer.bundle.js": (
                assets.renderer_js,
                "text/javascript; charset=utf-8",
            ),
            "/static/licenses/three-LICENSE.txt": (
                assets.three_license,
                "text/plain; charset=utf-8",
            ),
        }
        route = routes.get(parsed.path)
        if route is None:
            self._send_empty(handler, 404)
            return
        path, content_type = route
        self._send_bytes(handler, path.read_bytes(), content_type)

    def _serve_frame(self, handler: BaseHTTPRequestHandler, query_string: str) -> None:
        query = parse_qs(query_string, keep_blank_values=True)
        raw_index = query.get("index", ["0"])[0]
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            self._send_empty(handler, 400)
            return
        if index < 0:
            self._send_empty(handler, 400)
            return

        with self._lock:
            session = query.get("session", [None])[0]
            frame = self._frames[index] if 0 <= index < len(self._frames) else None
            should_reset = (
                session is not None and session != self._session_id
            ) or (
                session is None
                and index > 0
                and frame is None
                and bool(self._frames)
            )

        if should_reset:
            self._send_json(
                handler,
                {"reset": True, "session": self._session_id, "next_index": 0},
            )
        elif frame is None:
            self._send_empty(handler, 204)
        else:
            self._send_json(handler, frame)

    @staticmethod
    def _send_empty(handler: BaseHTTPRequestHandler, status: int) -> None:
        handler.send_response(status)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    @staticmethod
    def _send_bytes(
        handler: BaseHTTPRequestHandler, data: bytes, content_type: str
    ) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @classmethod
    def _send_json(cls, handler: BaseHTTPRequestHandler, payload: dict) -> None:
        cls._send_bytes(
            handler,
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )


def make_three_live_server(args) -> ThreeLiveServer:
    server = ThreeLiveServer(
        plot_dir=args.visual_dir,
        port=args.visual_port,
        bind_host=args.visual_bind_host,
        public_host=args.visual_public_host,
        poll_ms=args.visual_poll_ms,
        log_requests=getattr(args, "visual_log_requests", False),
    )
    url = server.start()
    print(f"live Three.js visualization: {url}")
    return server
