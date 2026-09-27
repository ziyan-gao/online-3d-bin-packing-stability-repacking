import json
import socket
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

import packing.threejs_visualization.live as live_module
from packing.threejs_visualization.assets import validated_runtime_assets
from packing.threejs_visualization.frame import SceneFrame, VisualizationFrame
from packing.threejs_visualization.live import ThreeLiveServer, make_three_live_server


def _get(url: str):
    return urlopen(url, timeout=2)


def _frame(title: str) -> VisualizationFrame:
    empty = SceneFrame(bounds=[0, 0, 0, 1, 1, 1], objects=[])
    return VisualizationFrame(
        title=title,
        scenes={"container": empty, "buffer": empty, "holding": empty},
        legend=[],
    )


@pytest.fixture
def live_server(tmp_path):
    server = ThreeLiveServer(
        plot_dir=tmp_path,
        port=0,
        bind_host="127.0.0.1",
        public_host="localhost",
        poll_ms=25,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_serves_runtime_assets_on_actual_ephemeral_port(live_server):
    assert live_server.port > 0
    assert live_server.url == f"http://localhost:{live_server.port}/index.html"

    expected = {
        "/index.html": ("text/html", b"Packing live view"),
        "/static/style.css": ("text/css", b"scene-panel"),
        "/static/renderer.bundle.js": ("text/javascript", b"PackingThree"),
        "/static/licenses/three-LICENSE.txt": ("text/plain", b"MIT License"),
    }
    for path, (content_type, marker) in expected.items():
        with _get(live_server.url_for(path)) as response:
            assert response.headers["Content-Type"].startswith(content_type)
            assert response.headers["Cache-Control"] == "no-store"
            assert marker in response.read()


def test_served_live_html_uses_configured_poll_interval(live_server):
    with _get(live_server.url_for("/index.html")) as response:
        html = response.read().decode("utf-8")

    assert "PackingThree.startLive({interval: 25});" in html
    assert "PackingThree.startLive();" not in html


def test_pushes_dataclass_and_dict_frames_in_order(live_server):
    live_server.push(_frame("first"))
    live_server.push({"title": "second", "scenes": {}, "legend": []})

    with _get(live_server.url_for("/frame.json?index=0")) as response:
        first = json.load(response)
    with _get(live_server.url_for("frame.json?index=1")) as response:
        second = json.load(response)

    assert first["index"] == 0
    assert second["index"] == 1
    assert first["frame"]["title"] == "first"
    assert second["frame"]["title"] == "second"
    assert first["session"] == second["session"]
    assert isinstance(first["version"], int)


def test_missing_frame_and_stale_session_protocol(live_server):
    with _get(live_server.url_for("/frame.json?index=0")) as response:
        assert response.status == 204
        assert response.headers["Cache-Control"] == "no-store"

    live_server.push({"title": "available"})
    with _get(
        live_server.url_for("/frame.json?index=9&session=previous-session")
    ) as response:
        reset = json.load(response)

    assert reset == {
        "reset": True,
        "session": reset["session"],
        "next_index": 0,
    }
    assert reset["session"] != "previous-session"


@pytest.mark.parametrize("query", ["index=bad", "index=1.2", "index=", "index=-1"])
def test_invalid_frame_index_is_bad_request(live_server, query):
    with pytest.raises(HTTPError) as error:
        _get(live_server.url_for(f"/frame.json?{query}"))
    assert error.value.code == 400


@pytest.mark.parametrize(
    "path",
    [
        "/unknown",
        "/static/unknown.js",
        "/static/%2e%2e/frame.py",
        "/%2e%2e/live_plot.py",
    ],
)
def test_unknown_and_traversal_paths_are_not_found(live_server, path):
    with pytest.raises(HTTPError) as error:
        _get(live_server.url_for(path))
    assert error.value.code == 404


def test_push_rejects_invalid_json_without_consuming_an_index(live_server):
    with pytest.raises(TypeError, match="JSON serializable"):
        live_server.push({"bad": object()})
    with pytest.raises(ValueError, match="finite"):
        live_server.push({"bad": float("nan")})

    live_server.push({"title": "valid"})
    with _get(live_server.url_for("/frame.json?index=0")) as response:
        assert json.load(response)["frame"] == {"title": "valid"}


def test_push_reports_circular_containers_as_serialization_errors(live_server):
    circular = {}
    circular["self"] = circular

    with pytest.raises(TypeError, match="JSON serializable"):
        live_server.push(circular)


@pytest.mark.parametrize(
    "value",
    [
        None,
        3,
        "frame",
        ["not", "a", "frame"],
        SimpleNamespace(to_dict=lambda: {"title": "forged"}),
    ],
)
def test_push_accepts_only_visualization_frames_or_dicts(live_server, value):
    with pytest.raises(TypeError, match="VisualizationFrame or dict"):
        live_server.push(value)


def test_push_queues_a_deep_json_snapshot(live_server):
    frame = {
        "title": "original",
        "scenes": {"container": {"objects": [{"id": "item:before"}]}},
        "legend": ["before"],
    }
    live_server.push(frame)

    frame["title"] = "mutated"
    frame["scenes"]["container"]["objects"][0]["id"] = "item:after"
    frame["scenes"]["container"]["objects"].append(object())
    frame["legend"].append("after")

    with _get(live_server.url_for("/frame.json?index=0")) as response:
        queued = json.load(response)["frame"]

    assert queued == {
        "title": "original",
        "scenes": {"container": {"objects": [{"id": "item:before"}]}},
        "legend": ["before"],
    }


def test_stop_is_idempotent_and_releases_port(tmp_path):
    server = ThreeLiveServer(plot_dir=tmp_path, port=0)
    server.start()
    port = server.port

    server.stop()
    server.stop()

    replacement = socket.socket()
    try:
        replacement.bind(("127.0.0.1", port))
    finally:
        replacement.close()


def test_concurrent_starts_create_only_one_server(tmp_path, monkeypatch):
    instances = []
    instances_lock = threading.Lock()

    class FakeHTTPServer:
        def __init__(self, address, handler):
            self.server_address = (address[0], 43210)
            self.stopped = threading.Event()
            with instances_lock:
                instances.append(self)
            time.sleep(0.05)

        def serve_forever(self):
            self.stopped.wait()

        def shutdown(self):
            self.stopped.set()

        def server_close(self):
            pass

    monkeypatch.setattr(live_module, "ThreadingHTTPServer", FakeHTTPServer)
    server = ThreeLiveServer(plot_dir=tmp_path, port=0)
    callers_ready = threading.Barrier(8)

    def start_together():
        callers_ready.wait()
        server.start()

    callers = [threading.Thread(target=start_together) for _ in range(8)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=2)

    try:
        assert len(instances) == 1
    finally:
        server.stop()


def test_exception_after_response_start_does_not_write_second_status_line(
    live_server, monkeypatch
):
    def send_then_fail(handler, assets):
        live_server._send_bytes(handler, b"ok", "text/plain")
        raise RuntimeError("after headers")

    monkeypatch.setattr(live_server, "_dispatch_get", send_then_fail)
    client = socket.create_connection(("127.0.0.1", live_server.port), timeout=2)
    try:
        client.sendall(b"GET /index.html HTTP/1.0\r\nHost: localhost\r\n\r\n")
        chunks = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        client.close()

    response = b"".join(chunks)
    assert response.count(b"HTTP/1.0") == 1
    assert response.endswith(b"ok")


def test_missing_asset_is_named_before_server_starts(tmp_path):
    static = tmp_path / "static"
    (static / "licenses").mkdir(parents=True)
    required = {
        "live.html": "live",
        "replay.html": "replay",
        "style.css": "css",
        "renderer.bundle.js": "js",
        "licenses/three-LICENSE.txt": "license",
    }
    for relative, contents in required.items():
        path = static / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    (static / "renderer.bundle.js").unlink()

    with pytest.raises(FileNotFoundError, match=r"renderer\.bundle\.js"):
        validated_runtime_assets(static)
    server = ThreeLiveServer(plot_dir=tmp_path, port=0, assets_root=static)
    with pytest.raises(FileNotFoundError, match=r"renderer\.bundle\.js"):
        server.start()
    assert server._httpd is None


def test_make_server_uses_visual_arguments_and_prints_url(tmp_path, capsys):
    args = SimpleNamespace(
        visual_dir=str(tmp_path),
        visual_port=0,
        visual_bind_host="127.0.0.1",
        visual_public_host="localhost",
        visual_poll_ms=77,
        visual_log_requests=False,
    )

    server = make_three_live_server(args)
    try:
        assert server.poll_ms == 77
        assert capsys.readouterr().out.strip() == (
            f"live Three.js visualization: {server.url}"
        )
    finally:
        server.stop()
