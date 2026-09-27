import * as THREE from "three";
import { toThreeCoordinates } from "./helpers.js";

function color(value, fallback) {
  try { return new THREE.Color(value || fallback); } catch { return new THREE.Color(fallback); }
}

function metadata(object) {
  return { sceneObjectId: object.id, tooltip: object.tooltip || {} };
}

function addEdges(group, geometry, position, edgeColor = "#111827", opacity = 1) {
  const material = new THREE.LineBasicMaterial({ color: color(edgeColor, "#111827"), transparent: opacity < 1, opacity });
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), material);
  edges.position.copy(position);
  group.add(edges);
}

function makeBox(object, wireOnly = false) {
  const size = object.geometry?.size || [1, 1, 1];
  const origin = object.geometry?.origin || [0, 0, 0];
  const dimensions = toThreeCoordinates(size);
  const center = toThreeCoordinates(origin.map((value, index) => Number(value) + Number(size[index] || 0) / 2));
  const geometry = new THREE.BoxGeometry(...dimensions);
  const group = new THREE.Group();
  const position = new THREE.Vector3(...center);
  if (!wireOnly) {
    const opacity = Number(object.style?.opacity ?? 1);
    const material = new THREE.MeshLambertMaterial({
      color: color(object.style?.color, "#94a3b8"), transparent: opacity < 1, opacity,
      depthWrite: opacity >= 0.9, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData = metadata(object);
    group.add(mesh);
  }
  addEdges(group, geometry, position, object.style?.edge_color || "#111827", Number(object.style?.opacity ?? 1));
  group.userData = metadata(object);
  return group;
}

function hatchTexture() {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const offset = (y * size + x) * 4;
      const stripe = (x - y + size) % size < 7;
      data[offset] = data[offset + 1] = data[offset + 2] = stripe ? 110 : 255;
      data[offset + 3] = stripe ? 255 : 35;
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  // ShapeGeometry UVs are physical XY coordinates: one stripe every 24 mm.
  texture.repeat.set(1 / 24, 1 / 24);
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function makePatch(object) {
  const polygon = object.geometry?.polygon_xy || [];
  if (polygon.length < 3) return null;
  const shape = new THREE.Shape(polygon.map(([x, y]) => new THREE.Vector2(Number(x), Number(y))));
  const geometry = new THREE.ShapeGeometry(shape);
  const opacity = Number(object.style?.opacity ?? 1);
  const hatched = object.style?.pattern === "diagonal_hatch";
  const material = new THREE.MeshLambertMaterial({
    color: color(object.style?.color, object.kind === "support_patch" ? "#ffd700" : "#d6a828"),
    transparent: hatched || opacity < 1, opacity, side: THREE.DoubleSide,
    depthWrite: !hatched && opacity >= 0.9,
    map: hatched ? hatchTexture() : null,
  });
  const group = new THREE.Group();
  const height = Number(object.geometry?.z_range?.[1] ?? object.geometry?.z_range?.[0] ?? 0);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = Math.PI / 2;
  mesh.position.y = height;
  mesh.userData = metadata(object);
  group.add(mesh);
  const points = polygon.map(([x, y]) => new THREE.Vector3(Number(x), height + 0.2, Number(y)));
  const borderGeometry = new THREE.BufferGeometry().setFromPoints(points);
  const border = new THREE.LineLoop(borderGeometry, new THREE.LineBasicMaterial({ color: color(object.style?.edge_color, "#000000"), transparent: opacity < 1, opacity }));
  border.userData = metadata(object);
  group.add(border);
  return group;
}

// Perspective world size per unit camera depth (CSS pixels).
export function spriteScaleForPixels(pixels, fovDegrees, viewportHeight) {
  const values = [pixels, fovDegrees, viewportHeight].map(Number);
  if (!values.every((value) => Number.isFinite(value) && value > 0)) return 0;
  return values[0] * 2 * Math.tan(THREE.MathUtils.degToRad(values[1] / 2)) / values[2];
}

function sizeArrow(group, unitsPerPixel) {
  const { length, shaftWidth, headWidth, headLength, shaftScale, headScale } = group.userData.arrowDimensions;
  const [shaft, head] = group.children;
  // Keep short arrows slender and never let the head extend past the force tip.
  const baseHeight = Math.min(headLength * unitsPerPixel, length * 0.5);
  const baseRadius = Math.min(headWidth * unitsPerPixel / 2, baseHeight * 0.6);
  const height = Math.min(headLength * unitsPerPixel * headScale, length * 0.5);
  const radius = Math.min(headWidth * unitsPerPixel * headScale / 2, height * 0.6);
  const shaftRadius = Math.min(
    Math.min(shaftWidth * unitsPerPixel / 2, baseRadius * 0.30) * shaftScale,
    radius * 0.9,
  );
  shaft.scale.set(shaftRadius, length - height, shaftRadius);
  shaft.position.y = (length - height) / 2;
  head.scale.set(radius, height, radius);
  head.position.y = length - height / 2;
}

export function updateArrowheadScale(group, camera, viewportHeight) {
  if (!group?.userData?.arrowDimensions || !camera?.isPerspectiveCamera) return false;
  camera.updateMatrixWorld();
  const center = group.localToWorld(new THREE.Vector3(0, group.userData.arrowDimensions.length / 2, 0));
  const depth = -center.applyMatrix4(camera.matrixWorldInverse).z;
  const units = spriteScaleForPixels(1, camera.getEffectiveFOV(), viewportHeight) * Math.max(camera.near, depth);
  if (!(units > 0)) return false;
  sizeArrow(group, units);
  return true;
}

export function updateArrowheadScales(root, camera, viewportHeight) {
  root?.traverse?.((object) => updateArrowheadScale(object, camera, viewportHeight));
}

function makeForceArrow(object) {
  const data = object.geometry || {};
  const direction = new THREE.Vector3(...toThreeCoordinates(data.direction || [0, 0, 1])).normalize();
  const length = Number(data.display_length || 0);
  if (!Number.isFinite(length) || length <= 0 || direction.lengthSq() === 0) return null;
  const opacity = Number(object.style?.opacity ?? 1);
  const material = new THREE.MeshLambertMaterial({
    color: color(object.style?.color, "#ef4444"),
    transparent: opacity < 1, opacity, depthWrite: opacity >= 0.9,
  });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 24), material);
  const head = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 32), material);
  shaft.userData = metadata(object);
  head.userData = metadata(object);
  const group = new THREE.Group();
  group.position.set(...toThreeCoordinates(data.origin || [0, 0, 0]));
  group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.userData = {
    ...metadata(object),
    arrowDimensions: {
      length,
      shaftWidth: Number(object.style?.shaft_width_px ?? 2),
      headWidth: Number(object.style?.head_width_px ?? 12),
      headLength: Number(object.style?.head_length_px ?? 16),
      shaftScale: Number(object.style?.shaft_scale ?? 1),
      headScale: Number(object.style?.head_scale ?? 1),
    },
  };
  group.add(shaft, head);
  sizeArrow(group, 1);
  return group;
}

function makeAnchor(object) {
  const radius = Math.max(1, Number(object.style?.size ?? 6) / 2);
  const geometry = new THREE.SphereGeometry(radius, 16, 12);
  const material = new THREE.MeshBasicMaterial({ color: color(object.style?.color, "#0f766e"), depthTest: false });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...toThreeCoordinates(object.geometry?.position));
  mesh.userData = metadata(object);
  return mesh;
}

export function createPrimitive(object, lineMaterials) {
  switch (object?.kind) {
    case "item": case "ems": case "highlighted_item": case "repacked_item": case "repack_space": return makeBox(object);
    case "container_wire": return makeBox(object, true);
    case "support_patch": case "contact_patch": return makePatch(object);
    case "force_arrow": return makeForceArrow(object, lineMaterials);
    case "anchor": return makeAnchor(object);
    default: console.warn(`Ignoring unknown scene object kind: ${object?.kind}`); return null;
  }
}
