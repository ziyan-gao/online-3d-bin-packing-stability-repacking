import * as THREE from "three";
import { OrbitControls } from "../vendor/three/examples/jsm/controls/OrbitControls.js";
import { attachInteractions } from "./interactions.js";
import { createPrimitive, updateArrowheadScales } from "./primitives.js";
import { orderSceneObjects, toThreeCoordinates } from "./helpers.js";
import { displayObjects, DEFAULT_DISPLAY_OPTIONS } from "./display-controls.js";

function disposeMaterial(material) {
  for (const value of Object.values(material || {})) if (value?.isTexture) value.dispose();
  material?.dispose?.();
}

function disposeTree(root) {
  root.traverse((node) => {
    node.geometry?.dispose?.();
    if (Array.isArray(node.material)) node.material.forEach(disposeMaterial); else disposeMaterial(node.material);
  });
  root.clear();
}

function bestEffort(action) {
  try { action?.(); } catch { /* Continue releasing the remaining resources. */ }
}

const DEFAULT_RUNTIME = {
  createRenderer: (options) => new THREE.WebGLRenderer(options),
  createControls: (camera, canvas) => new OrbitControls(camera, canvas),
  attachInteractions,
  createResizeObserver: (callback) => new ResizeObserver(callback),
};

export class SceneView {
  constructor(host, name, runtime = DEFAULT_RUNTIME) {
    this.host = host;
    this.name = name;
    this.available = false;
    this.failed = false;
    this.renderer = null;
    this.controls = null;
    this.resizeObserver = null;
    this.detachInteractions = null;
    this.errorPanel = null;
    this.disposed = false;
    this.viewportHeight = 0;
    this.lineMaterials = new Set();
    this.hasCamera = false;
    this.displayOptions = { ...DEFAULT_DISPLAY_OPTIONS };
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
    this.content = new THREE.Group();
    this.scene.add(this.content, new THREE.AmbientLight(0xffffff, 0.76));
    const light = new THREE.DirectionalLight(0xffffff, 0.72);
    light.position.set(1, 2, 1);
    this.scene.add(light);
    try {
      if (!host) throw new Error(`Missing host for ${name} scene`);
      this.renderer = runtime.createRenderer({ antialias: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      host.appendChild(this.renderer.domElement);
      this.controls = runtime.createControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.detachInteractions = runtime.attachInteractions(this.renderer.domElement, this.camera, this.content, host);
      this.resizeObserver = runtime.createResizeObserver(() => this.resize());
      this.resizeObserver.observe(host);
      this.available = true;
      this.renderer.setAnimationLoop(() => this.render());
      this.resize();
    } catch (error) {
      this.available = false;
      this.failed = true;
      this.cleanupRuntimeResources();
      if (host) {
        this.errorPanel = document.createElement("div");
        this.errorPanel.className = "webgl-error";
        this.errorPanel.textContent = `WebGL renderer unavailable: ${error.message}`;
        host.appendChild(this.errorPanel);
      }
    }
  }

  cleanupRuntimeResources() {
    const renderer = this.renderer;
    const canvas = renderer?.domElement;
    bestEffort(() => renderer?.setAnimationLoop?.(null));
    bestEffort(() => this.resizeObserver?.disconnect?.());
    bestEffort(() => this.detachInteractions?.());
    bestEffort(() => this.controls?.dispose?.());
    bestEffort(() => renderer?.dispose?.());
    bestEffort(() => renderer?.forceContextLoss?.());
    bestEffort(() => canvas?.remove?.());
    this.resizeObserver = null;
    this.detachInteractions = null;
    this.controls = null;
    this.renderer = null;
  }

  resize() {
    if (!this.available || !this.renderer || !this.controls) return;
    const rawWidth = this.host.clientWidth;
    const rawHeight = this.host.clientHeight;
    const width = Math.max(1, rawWidth);
    const height = Math.max(1, rawHeight);
    this.viewportHeight = rawHeight > 0 ? rawHeight : 0;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    for (const material of this.lineMaterials) material.resolution.set(width, height);
    updateArrowheadScales(this.content, this.camera, this.viewportHeight);
  }

  applyCamera(sceneFrame) {
    if (!this.available || !this.renderer || !this.controls || this.hasCamera) return;
    const ranges = sceneFrame.camera?.ranges;
    const bounds = ranges
      ? [ranges.x?.[0], ranges.y?.[0], ranges.z?.[0], ranges.x?.[1], ranges.y?.[1], ranges.z?.[1]].map(Number)
      : sceneFrame.bounds;
    if (bounds.some((value) => !Number.isFinite(value))) bounds.splice(0, 6, ...sceneFrame.bounds);
    const centerPhysical = [(bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2, (bounds[2] + bounds[5]) / 2];
    const center = new THREE.Vector3(...toThreeCoordinates(centerPhysical));
    const span = Math.max(1, bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]);
    const eye = sceneFrame.camera?.eye || { x: 1.6, y: 1.6, z: 1.1 };
    const offset = toThreeCoordinates([Number(eye.x ?? 1.6) * span, Number(eye.y ?? 1.6) * span, Number(eye.z ?? 1.1) * span]);
    this.camera.position.copy(center).add(new THREE.Vector3(...offset));
    this.camera.near = Math.max(0.1, span / 1000);
    this.camera.far = Math.max(1000, span * 20);
    this.camera.updateProjectionMatrix();
    updateArrowheadScales(this.content, this.camera, this.viewportHeight);
    this.controls.target.copy(center);
    this.controls.update();
    this.hasCamera = true;
  }

  replace(sceneFrame) {
    this.lastSceneFrame = sceneFrame;
    if (!this.available || !this.renderer || !this.controls) return;
    disposeTree(this.content);
    this.lineMaterials.clear();
    for (const object of orderSceneObjects(displayObjects(sceneFrame.objects, this.displayOptions))) {
      const primitive = createPrimitive(object, this.lineMaterials);
      if (primitive) this.content.add(primitive);
    }
    this.applyCamera(sceneFrame);
    this.setAxisLabels(sceneFrame.camera?.axis_titles);
    this.resize();
  }

  setDisplayOptions(options) {
    this.displayOptions = { ...DEFAULT_DISPLAY_OPTIONS, ...options };
    if (this.lastSceneFrame) this.replace(this.lastSceneFrame);
  }

  setAxisLabels(labels = {}) {
    if (!this.available || !this.host) return;
    let element = this.host.querySelector(".axis-labels");
    if (!element) { element = document.createElement("div"); element.className = "axis-labels"; this.host.appendChild(element); }
    element.textContent = `X: ${labels.x || "X"} · Y: ${labels.y || "Y"} · Z: ${labels.z || "Z"}`;
  }

  render() {
    if (!this.available || !this.renderer || !this.controls) return;
    this.controls.update();
    updateArrowheadScales(this.content, this.camera, this.viewportHeight);
    this.renderer.render(this.scene, this.camera);
  }

  exportPNG() {
    if (!this.available || this.disposed) throw new Error("Container renderer unavailable");
    const renderer = this.renderer;
    const size = renderer.getSize(new THREE.Vector2());
    const ratio = renderer.getPixelRatio();
    const scale = 2400 / Math.max(size.x, size.y);
    try {
      renderer.setPixelRatio(1);
      renderer.setSize(Math.round(size.x * scale), Math.round(size.y * scale), false);
      // Preserve current framing and arrow proportions; only increase pixel density.
      updateArrowheadScales(this.content, this.camera, this.viewportHeight);
      renderer.render(this.scene, this.camera);
      // Capture synchronously while the WebGL drawing buffer is still valid.
      return renderer.domElement.toDataURL("image/png");
    } finally {
      renderer.setPixelRatio(ratio);
      renderer.setSize(size.x, size.y, false);
      renderer.render(this.scene, this.camera);
    }
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.available = false;
    disposeTree(this.content);
    this.lineMaterials.clear();
    this.cleanupRuntimeResources();
    bestEffort(() => this.errorPanel?.remove?.());
    this.errorPanel = null;
  }
}
