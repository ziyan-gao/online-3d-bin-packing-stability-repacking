import json
import os


REPLAY_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pack Replay</title>
  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
  </script>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #f5f6f8; font-family: sans-serif; overflow: hidden; }
    #root { width: 100%; height: calc(100% - 64px); display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 8px; padding: 8px; box-sizing: border-box; }
    #main { grid-row: 1 / span 2; }
    .panel { position: relative; min-width: 0; min-height: 0; background: #fff; border: 1px solid #d8dde6; }
    .pane-label { position: absolute; left: 10px; top: 8px; z-index: 2; color: #1f2937; font-size: 13px; background: rgba(255,255,255,0.82); padding: 4px 6px; border-radius: 4px; }
    #legend { position: absolute; left: 14px; bottom: 14px; z-index: 3; max-width: 44%; color: #1f2937; background: rgba(255,255,255,0.88); border: 1px solid #d1d5db; padding: 8px 10px; font-size: 13px; line-height: 1.35; }
    #controls {
      height: 64px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 14px;
      box-sizing: border-box;
      background: #fff;
      border-top: 1px solid #ddd;
    }
    button { min-width: 64px; height: 34px; }
    input[type="range"] { flex: 1; }
    #label { min-width: 260px; font-size: 14px; color: #222; }
    canvas { display: block; }
  </style>
</head>
<body>
  <div id="root">
    <div id="main" class="panel"><div class="pane-label" id="title">Packing</div><div id="legend"></div></div>
    <div id="buffer" class="panel"><div class="pane-label">Buffer Items</div></div>
    <div id="holding" class="panel"><div class="pane-label">Staging Area</div></div>
  </div>
  <div id="controls">
    <button id="prev">Prev</button>
    <button id="play">Play</button>
    <button id="next">Next</button>
    <input id="slider" type="range" min="0" max="0" value="0" />
    <span id="label"></span>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const frames = __FRAMES__;
    const intervalMs = __INTERVAL_MS__;
    let index = 0;
    let timer = null;

    const slider = document.getElementById('slider');
    const label = document.getElementById('label');
    const play = document.getElementById('play');

    slider.max = Math.max(frames.length - 1, 0);

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
      const point = new THREE.Mesh(
        new THREE.SphereGeometry(8, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0xff6400 })
      );
      point.position.set(p.x, p.z, p.y);
      group.add(point);
    }

    function addFloor(group, width, depth) {
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(Math.max(width, 1), Math.max(depth, 1)),
        new THREE.MeshBasicMaterial({ color: 0xe5e7eb, transparent: true, opacity: 0.3, side: THREE.DoubleSide })
      );
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
      (payload.ems || []).forEach((b) => addBox(main.group, b));
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

    async function render(nextIndex) {
      if (!frames.length) return;
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      const frame = frames[index];
      renderScene(frame.scene);
      slider.value = index;
      label.textContent = `${index + 1}/${frames.length}: ${frame.title}`;
    }

    function stop() {
      if (timer !== null) clearInterval(timer);
      timer = null;
      play.textContent = 'Play';
    }

    document.getElementById('prev').onclick = () => { stop(); render(index - 1); };
    document.getElementById('next').onclick = () => { stop(); render(index + 1); };
    slider.oninput = () => { stop(); render(Number(slider.value)); };
    play.onclick = () => {
      if (timer !== null) {
        stop();
        return;
      }
      play.textContent = 'Pause';
      timer = setInterval(() => {
        if (index >= frames.length - 1) {
          stop();
          return;
        }
        render(index + 1);
      }, intervalMs);
    };

    render(0);
  </script>
</body>
</html>
"""


class InteractiveReplayRecorder:
    def __init__(
        self,
        out_path: str,
        interval_ms: int = 700,
    ) -> None:
        self.out_path = out_path
        self.interval_ms = int(interval_ms)
        self.frames = []

    def capture(self, title: str, scene) -> None:
        self.frames.append({"title": title, "scene": scene})

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        html = REPLAY_TEMPLATE.replace("__FRAMES__", json.dumps(self.frames))
        html = html.replace("__INTERVAL_MS__", str(self.interval_ms))
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved interactive replay: {self.out_path} ({len(self.frames)} frames)")
