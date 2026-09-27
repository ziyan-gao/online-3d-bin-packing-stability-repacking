import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";

import { FrameRenderer } from "../src/frame-renderer.js";
import { SceneView } from "../src/scene-view.js";
import {
  createPrimitive,
  spriteScaleForPixels,
  updateArrowheadScale,
} from "../src/primitives.js";
import { ReplayClock, startReplay } from "../src/replay.js";

test("CI hatch is a repeating transparent texture clipped by the polygon mesh", () => {
  const patch = createPrimitive({ id: "ci", kind: "contact_patch",
    geometry: { polygon_xy: [[0,0], [100,0], [0,100]], z_range: [10,10] },
    style: { pattern: "diagonal_hatch", opacity: 0.65 } });
  const mesh = patch.children[0];
  assert.ok(mesh.material.map?.isDataTexture);
  assert.equal(mesh.material.map.wrapS, THREE.RepeatWrapping);
  assert.equal(mesh.material.transparent, true);
  assert.equal(mesh.material.depthWrite, false);
  assert.equal(mesh.geometry.type, "ShapeGeometry");
  const pixels = mesh.material.map.image.data;
  assert.ok(pixels.some((value, index) => index % 4 === 3 && value < 255));
});

test("export button overrides the non-interactive scene title", () => {
  const css = readFileSync(new URL("../static/style.css", import.meta.url), "utf8");
  assert.match(css, /#export-container\s*\{[^}]*pointer-events:\s*auto\s*;/);
});

test("container PNG export renders at high resolution and restores renderer even on failure", () => {
  const view = Object.create(SceneView.prototype);
  let size = new THREE.Vector2(800, 600);
  let ratio = 2;
  let fail = false;
  const rendered = [];
  Object.assign(view, {
    available: true, disposed: false, viewportHeight: 600,
    content: new THREE.Group(), camera: new THREE.PerspectiveCamera(),
    scene: new THREE.Scene(),
    renderer: {
      getSize: (target) => target.copy(size),
      getPixelRatio: () => ratio,
      setPixelRatio: (value) => { ratio = value; },
      setSize: (w, h) => { size.set(w, h); },
      render: (scene) => { rendered.push([scene, size.x, size.y, ratio]); },
      domElement: { toDataURL: (type) => {
        assert.equal(type, "image/png");
        if (fail) throw new Error("encoding failed");
        return "data:image/png;base64,test";
      } },
    },
  });
  assert.equal(view.exportPNG(), "data:image/png;base64,test");
  assert.deepEqual(rendered[0], [view.scene, 2400, 1800, 1]);
  assert.deepEqual(size.toArray(), [800, 600]);
  assert.equal(ratio, 2);
  fail = true;
  assert.throws(() => view.exportPNG(), /encoding failed/);
  assert.deepEqual(size.toArray(), [800, 600]);
  assert.equal(ratio, 2);
});

test("scene view cleans partial WebGL setup when later initialization fails", (t) => {
  const previousWindow = globalThis.window;
  const previousDocument = globalThis.document;
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.document = previousDocument;
  });
  globalThis.window = { devicePixelRatio: 1 };
  globalThis.document = {
    createElement() {
      return { className: "", textContent: "", remove() {} };
    },
  };

  const calls = {
    animationStopped: 0,
    controlsDisposed: 0,
    interactionsDetached: 0,
    observerDisconnected: 0,
    rendererDisposed: 0,
    contextLost: 0,
    canvasRemoved: 0,
  };
  const canvas = {
    remove() { calls.canvasRemoved += 1; },
  };
  const renderer = {
    domElement: canvas,
    setPixelRatio() {},
    setAnimationLoop(value) {
      if (value === null) calls.animationStopped += 1;
    },
    dispose() { calls.rendererDisposed += 1; },
    forceContextLoss() { calls.contextLost += 1; },
  };
  const host = {
    appendChild() {},
  };
  const runtime = {
    createRenderer: () => renderer,
    createControls: () => ({
      dispose() {
        calls.controlsDisposed += 1;
        throw new Error("injected controls cleanup failure");
      },
    }),
    attachInteractions: () => () => { calls.interactionsDetached += 1; },
    createResizeObserver: () => ({
      observe() { throw new Error("injected observer failure"); },
      disconnect() { calls.observerDisconnected += 1; },
    }),
  };

  const view = new SceneView(host, "container", runtime);

  assert.equal(view.failed, true);
  assert.equal(view.available, false);
  view.dispose();
  view.dispose();
  assert.deepEqual(calls, {
    animationStopped: 1,
    controlsDisposed: 1,
    interactionsDetached: 1,
    observerDisconnected: 1,
    rendererDisposed: 1,
    contextLost: 1,
    canvasRemoved: 1,
  });
});

test("converts requested sprite pixels to non-attenuating perspective scale", () => {
  const scale = spriteScaleForPixels(16, 45, 300);
  assert.ok(Math.abs(scale - 16 * 2 * Math.tan(THREE.MathUtils.degToRad(22.5)) / 300) < 1e-12);
  assert.equal(spriteScaleForPixels(16, 45, 0), 0);
});

test("force arrow uses solid cylinder and cone with anchored endpoints at every zoom", () => {
  const arrow = createPrimitive({
    id: "arrow",
    kind: "force_arrow",
    geometry: { origin: [1, 2, 3], direction: [0, 0, 1], display_length: 20 },
    style: { color: "#f00", shaft_width_px: 3, head_width_px: 12, head_length_px: 16 },
    tooltip: { type: "force_arrow" },
  }, new Set());
  const shaft = arrow.children.find((child) => child.geometry?.type === "CylinderGeometry");
  const head = arrow.children.find((child) => child.geometry?.type === "ConeGeometry");

  assert.ok(shaft);
  assert.ok(head);
  assert.equal(arrow.children.some((child) => child.isSprite), false);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  for (const distance of [100, 1000, 5000]) {
    camera.position.set(1, 3, distance);
    camera.lookAt(1, 3, 2);
    camera.updateMatrixWorld();
    assert.equal(updateArrowheadScale(arrow, camera, 300), true);
    arrow.updateMatrixWorld(true);
    assert.ok(shaft.scale.x > 0 && head.scale.x > shaft.scale.x);
    assert.ok(shaft.scale.x <= head.scale.x * 0.30 + 1e-12);
    assert.ok(head.scale.y <= 10);
    assert.ok(shaft.localToWorld(new THREE.Vector3(0, -0.5, 0)).distanceTo(new THREE.Vector3(1, 3, 2)) < 1e-9);
    assert.ok(head.localToWorld(new THREE.Vector3(0, 0.5, 0)).distanceTo(new THREE.Vector3(1, 23, 2)) < 1e-9);
  }
});

test("replay clock clears its timer synchronously and can restart", () => {
  let nextId = 1;
  const scheduled = [];
  const cancelled = [];
  const clock = new ReplayClock(
    (callback, interval) => { scheduled.push({ callback, interval }); return nextId++; },
    (id) => cancelled.push(id),
  );

  assert.equal(clock.start(() => {}, 500), true);
  assert.equal(clock.timer, 1);
  assert.equal(clock.pause(), true);
  assert.equal(clock.timer, null);
  assert.deepEqual(cancelled, [1]);
  assert.equal(clock.start(() => {}, 250), true);
  assert.equal(clock.timer, 2);
  assert.deepEqual(scheduled.map(({ interval }) => interval), [500, 250]);
});

test("frame renderer disposal is idempotent", () => {
  const renderer = Object.create(FrameRenderer.prototype);
  const counts = { container: 0, buffer: 0, holding: 0 };
  renderer.views = Object.fromEntries(Object.keys(counts).map((name) => [name, {
    dispose() { counts[name] += 1; },
  }]));
  renderer.disposed = false;

  renderer.dispose();
  renderer.dispose();

  assert.deepEqual(counts, { container: 1, buffer: 1, holding: 1 });
});

test("replay pause remains restartable while terminal stop disposes once", () => {
  let nextId = 1;
  const clock = new ReplayClock(() => nextId++, () => {});
  let disposals = 0;
  const renderer = { dispose() { disposals += 1; } };
  const root = { querySelector() { return null; } };
  const controller = startReplay([], { root, renderer, clock });

  clock.start(() => {}, 100);
  assert.equal(controller.pause(), true);
  assert.equal(clock.timer, null);
  assert.equal(clock.start(() => {}, 100), true);
  controller.stop();
  controller.stop();

  assert.equal(clock.timer, null);
  assert.equal(disposals, 1);
});
