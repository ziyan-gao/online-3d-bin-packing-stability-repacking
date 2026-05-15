import json
import os
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


LIVE_PLOT_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Buffer Support Live</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #f5f6f8; }
    #root { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    let nextFrameIndex = 0;
    let rendering = false;
    let sessionId = null;

    async function poll() {
      if (rendering) return;
      try {
        const params = new URLSearchParams({
          index: nextFrameIndex,
          t: Date.now(),
        });
        if (sessionId !== null) {
          params.set('session', sessionId);
        }
        const res = await fetch('/frame.json?' + params.toString(), { cache: 'no-store' });
        if (res.status === 204) return;
        if (!res.ok) return;
        rendering = true;
        const payload = await res.json();
        if (payload && payload.reset) {
          sessionId = payload.session;
          nextFrameIndex = payload.next_index || 0;
          rendering = false;
          setTimeout(poll, 0);
          return;
        }
        if (!payload || payload.index !== nextFrameIndex || !payload.figure) {
          rendering = false;
          return;
        }
        sessionId = payload.session || sessionId;
        const f = payload.figure;
        await Plotly.react('root', f.data || [], f.layout || {}, {responsive: true});
        nextFrameIndex += 1;
        rendering = false;
        setTimeout(poll, 0);
      } catch (e) {
        rendering = false;
      }
    }
    setInterval(poll, __POLL_MS__);
    poll();
  </script>
</body>
</html>
"""


class LivePlotServer:
    def __init__(
        self,
        plot_dir: str,
        port: int = 8765,
        bind_host: str = "127.0.0.1",
        public_host: str = "127.0.0.1",
        poll_ms: int = 500,
        log_requests: bool = False,
    ) -> None:
        self.plot_dir = plot_dir
        self.port = port
        self.bind_host = bind_host
        self.public_host = public_host
        self.poll_ms = max(100, poll_ms)
        self.log_requests = log_requests
        self.html_path = os.path.join(plot_dir, "index.html")
        self._httpd = None
        self._frames = []
        self._lock = threading.Lock()
        self._session_id = str(time.time_ns())

    @property
    def url(self) -> str:
        return f"http://{self.public_host}:{self.port}/index.html"

    def start(self) -> str:
        self._write_template()

        log_requests = self.log_requests
        owner = self

        class LivePlotHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/frame.json":
                    owner._serve_frame(self, parsed)
                    return
                if parsed.path == "/figure.json":
                    owner._serve_latest(self)
                    return
                super().do_GET()

            def end_headers(self):
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def log_message(self, format, *args):
                if log_requests:
                    super().log_message(format, *args)

        handler = partial(LivePlotHandler, directory=self.plot_dir)
        self._httpd = ThreadingHTTPServer((self.bind_host, self.port), handler)
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        return self.url

    def push(self, fig) -> None:
        os.makedirs(self.plot_dir, exist_ok=True)
        frame = {
            "session": self._session_id,
            "version": time.time_ns(),
            "figure": fig.to_dict(),
        }
        with self._lock:
            frame["index"] = len(self._frames)
            self._frames.append(frame)

    def _serve_frame(self, handler: SimpleHTTPRequestHandler, parsed) -> None:
        query = parse_qs(parsed.query)
        try:
            index = int(query.get("index", ["0"])[0])
        except ValueError:
            handler.send_response(400)
            handler.end_headers()
            return

        with self._lock:
            session = query.get("session", [None])[0]
            frame = self._frames[index] if 0 <= index < len(self._frames) else None
            should_reset = (
                (session is not None and session != self._session_id)
                or (session is None and index > 0 and frame is None and bool(self._frames))
            )

        if should_reset:
            self._send_json(
                handler,
                {
                    "reset": True,
                    "session": self._session_id,
                    "next_index": 0,
                },
            )
            return

        if frame is None:
            handler.send_response(204)
            handler.end_headers()
            return

        self._send_json(handler, frame)

    def _serve_latest(self, handler: SimpleHTTPRequestHandler) -> None:
        with self._lock:
            frame = self._frames[-1] if self._frames else None
        if frame is None:
            handler.send_response(204)
            handler.end_headers()
            return
        self._send_json(
            handler,
            {
                "session": frame["session"],
                "version": frame["version"],
                "figure": frame["figure"],
            },
        )

    def _send_json(self, handler: SimpleHTTPRequestHandler, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _write_template(self) -> None:
        os.makedirs(self.plot_dir, exist_ok=True)
        rendered_template = LIVE_PLOT_TEMPLATE.replace("__POLL_MS__", str(self.poll_ms))
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(rendered_template)


def make_live_server(args) -> LivePlotServer:
    server = LivePlotServer(
        plot_dir=args.visual_dir,
        port=args.visual_port,
        bind_host=args.visual_bind_host,
        public_host=args.visual_public_host,
        poll_ms=args.visual_poll_ms,
    )
    url = server.start()
    print(f"live Plotly visualization: {url}")
    return server

