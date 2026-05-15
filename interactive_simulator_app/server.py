from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .simulator import InteractivePackingSimulator

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    simulator: InteractivePackingSimulator

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            self._send_static(path.removeprefix("/static/"))
            return
        if path == "/state":
            self._send_json(self.simulator.state())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self._read_json()
        if path == "/place":
            self._send_json(
                self.simulator.place(
                    x=payload.get("x", 0),
                    y=payload.get("y", 0),
                    rotation=payload.get("rotation"),
                )
            )
            return
        if path == "/grid-place":
            self._send_json(
                self.simulator.place_grid(
                    x=payload.get("x", 0),
                    y=payload.get("y", 0),
                    rotation=payload.get("rotation"),
                )
            )
            return
        if path == "/rotation":
            self._send_json(self.simulator.set_rotation(payload.get("rotation", 0)))
            return
        if path == "/reset":
            self._send_json(self.simulator.reset())
            return
        if path == "/same-item-height":
            self._send_json(self.simulator.set_same_item_height(payload.get("enabled", False)))
            return
        if path == "/container-size":
            try:
                self._send_json(
                    self.simulator.resize_container(
                        dx=payload.get("dx", 600),
                        dy=payload.get("dy", 600),
                        dz=payload.get("dz", 600),
                    )
                )
            except ValueError as exc:
                self._send_json({"error": str(exc), **self.simulator.state()})
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[simulator] {self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_static(self, relative_path: str) -> None:
        file_path = (STATIC_DIR / relative_path).resolve()
        if STATIC_DIR.resolve() not in file_path.parents or not file_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if file_path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self._send_file(file_path, content_type)

    def _send_file(self, file_path: Path, content_type: str) -> None:
        raw = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal interactive packing simulator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ds-name", default="random")
    parser.add_argument("--buffer-capacity", type=int, default=12)
    parser.add_argument("--container-size", type=int, nargs=3, default=(600, 600, 600))
    parser.add_argument("--k-placement", type=int, default=80)
    parser.add_argument("--buffer-space", type=int, default=0)
    parser.add_argument("--remove-inscribed-ems", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_server() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    args = parse_args()
    SimulatorRequestHandler.simulator = InteractivePackingSimulator(
        seed=args.seed,
        ds_name=args.ds_name,
        buffer_capacity=args.buffer_capacity,
        container_size=tuple(args.container_size),
        k_placement=args.k_placement,
        buffer_space=args.buffer_space,
        remove_inscribed_ems=args.remove_inscribed_ems,
    )
    server = ThreadingHTTPServer((args.host, args.port), SimulatorRequestHandler)
    print(f"Interactive simulator: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulator.")
    finally:
        server.server_close()
