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
  <title>Packing Live</title>
  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
  </script>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #f5f6f8; font-family: sans-serif; overflow: hidden; }
    #root { width: 100%; height: 100%; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 8px; padding: 8px; box-sizing: border-box; }
    #main { grid-row: 1 / span 2; }
    .panel { position: relative; min-width: 0; min-height: 0; background: #fff; border: 1px solid #d8dde6; }
    .label { position: absolute; left: 10px; top: 8px; z-index: 2; color: #1f2937; font-size: 13px; background: rgba(255,255,255,0.82); padding: 4px 6px; border-radius: 4px; }
    #legend { position: absolute; left: 14px; bottom: 14px; z-index: 3; max-width: 44%; color: #1f2937; background: rgba(255,255,255,0.88); border: 1px solid #d1d5db; padding: 8px 10px; font-size: 13px; line-height: 1.35; }
    canvas { display: block; }
  </style>
</head>
<body>
  <div id="root">
    <div id="main" class="panel"><div class="label" id="title">Packing</div><div id="legend"></div></div>
    <div id="buffer" class="panel"><div class="label">Buffer Items</div></div>
    <div id="holding" class="panel"><div class="label">Staging Area</div></div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const panes = ['main', 'buffer', 'holding'].reduce((acc, id) => {
      const el = document.getElementById(id);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0xffffff, 1);
      el.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      scene.add(new THREE.HemisphereLight(0xffffff, 0x9ca3af, 1.6));
      const light = new THREE.DirectionalLight(0xffffff, 1.4);
      light.position.set(1, 2, 3);
      scene.add(light);
      acc[id] = { el, renderer, scene, camera, controls, group: new THREE.Group(), initialized: false };
      scene.add(acc[id].group);
      return acc;
    }, {});

    function clear(group) {
      while (group.children.length) {
        const obj = group.children.pop();
        obj.traverse?.((child) => {
          child.geometry?.dispose?.();
          child.material?.dispose?.();
        });
      }
    }

    function addBox(group, b, { wireOnly = false, highlighted = false } = {}) {
      const geom = new THREE.BoxGeometry(b.dx, b.dz, b.dy);
      const mat = new THREE.MeshLambertMaterial({
        color: new THREE.Color(b.color || '#8dd3c7'),
        transparent: (b.opacity ?? 1) < 1,
        opacity: b.opacity ?? 1,
        depthWrite: (b.opacity ?? 1) >= 0.8,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(b.x + b.dx / 2, b.z + b.dz / 2, b.y + b.dy / 2);
      if (!wireOnly) group.add(mesh);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geom),
        new THREE.LineBasicMaterial({ color: highlighted ? 0x0050b4 : 0x202020, linewidth: highlighted ? 3 : 1 })
      );
      edges.position.copy(mesh.position);
      group.add(edges);
    }

    function addPoint(group, p) {
      const geom = new THREE.SphereGeometry(8, 12, 8);
      const mat = new THREE.MeshBasicMaterial({ color: 0xff6400 });
      const point = new THREE.Mesh(geom, mat);
      point.position.set(p.x, p.z, p.y);
      group.add(point);
    }

    function addFloor(group, width, depth) {
      const geom = new THREE.PlaneGeometry(Math.max(width, 1), Math.max(depth, 1));
      const mat = new THREE.MeshBasicMaterial({ color: 0xe5e7eb, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
      const floor = new THREE.Mesh(geom, mat);
      floor.rotation.x = Math.PI / 2;
      floor.position.set(width / 2, 0, depth / 2);
      group.add(floor);
    }

    function fitCamera(pane, size) {
      const [dx, dy, dz] = size;
      const maxDim = Math.max(dx, dy, dz, 1);
      pane.camera.near = 0.1;
      pane.camera.far = maxDim * 10;
      pane.camera.position.set(dx * 1.25, dz * 1.05, dy * 1.35);
      pane.controls.target.set(dx / 2, dz / 2, dy / 2);
      pane.controls.update();
      pane.initialized = true;
    }

    function renderScene(payload) {
      document.getElementById('title').textContent = payload.title || 'Packing';
      document.getElementById('legend').innerHTML = (payload.legend || []).join('<br>');
      const main = panes.main;
      clear(main.group);
      const [cx, cy, cz] = payload.container || [1, 1, 1];
      addFloor(main.group, cx, cy);
      addBox(main.group, { x: 0, y: 0, z: 0, dx: cx, dy: cy, dz: cz, color: '#9ca3af', opacity: 0.03 }, { wireOnly: true });
      (payload.placed || []).forEach((b) => addBox(main.group, b));
      (payload.ems || []).forEach((b) => addBox(main.group, b, { wireOnly: false }));
      (payload.anchors || []).forEach((p) => addPoint(main.group, p));
      if (!main.initialized) fitCamera(main, [cx, cy, cz]);

      renderLinear(panes.buffer, payload.buffer || []);
      renderLinear(panes.holding, payload.holding || []);
      resize();
    }

    function renderLinear(pane, items) {
      clear(pane.group);
      let x = 0, maxY = 1, maxZ = 1;
      items.forEach((item) => {
        addBox(
          pane.group,
          { ...item, x, y: 0, z: 0, opacity: item.highlighted ? 0.38 : 1 },
          { highlighted: item.highlighted }
        );
        x += item.dx + 50;
        maxY = Math.max(maxY, item.dy);
        maxZ = Math.max(maxZ, item.dz);
      });
      addFloor(pane.group, Math.max(x, 1), maxY + 100);
      if (!pane.initialized) fitCamera(pane, [Math.max(x, 1), maxY + 100, maxZ + 100]);
    }

    function resize() {
      Object.values(panes).forEach((pane) => {
        const rect = pane.el.getBoundingClientRect();
        pane.renderer.setSize(rect.width, rect.height, false);
        pane.camera.aspect = Math.max(rect.width, 1) / Math.max(rect.height, 1);
        pane.camera.updateProjectionMatrix();
      });
    }

    function animate() {
      requestAnimationFrame(animate);
      Object.values(panes).forEach((pane) => {
        pane.controls.update();
        pane.renderer.render(pane.scene, pane.camera);
      });
    }
    window.addEventListener('resize', resize);
    animate();

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
        if (!payload || payload.index !== nextFrameIndex || !payload.scene) {
          rendering = false;
          return;
        }
        sessionId = payload.session || sessionId;
        renderScene(payload.scene);
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

    def push(self, scene) -> None:
        os.makedirs(self.plot_dir, exist_ok=True)
        frame = {
            "session": self._session_id,
            "version": time.time_ns(),
            "scene": scene,
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
                "scene": frame["scene"],
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
    print(f"live Three.js visualization: {url}")
    return server
