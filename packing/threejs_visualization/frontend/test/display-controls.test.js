import test from "node:test";
import assert from "node:assert/strict";
import { displayObjects, DEFAULT_DISPLAY_OPTIONS } from "../src/display-controls.js";
import { createPrimitive } from "../src/primitives.js";
import { SceneView } from "../src/scene-view.js";

const arrow = { id: "a", kind: "force_arrow",
  geometry: { origin: [1, 2, 3], display_length: 100, force_newtons: 42 },
  style: { shaft_width_px: 14, head_width_px: 21, head_length_px: 28, opacity: 0.8 } };
const objects = [arrow, { kind: "container_wire" },
  { kind: "item", style: { opacity: 0.5 } },
  { kind: "contact_patch", style: { opacity: 0.65 } }];

test("resultant tracks the CI display gap without changing its physical origin", () => {
  const resultant = { ...arrow, geometry: { ...arrow.geometry,
    origin: [4, 7, 26], physical_origin: [4, 7, 20], lbcp_display_z: 22,
    representation: "ci_resultant" } };
  const moved = displayObjects([resultant], { ...DEFAULT_DISPLAY_OPTIONS, ciGap: 10 })[0];
  assert.deepEqual(moved.geometry.origin, [4, 7, 32]);
  assert.deepEqual(moved.geometry.physical_origin, [4, 7, 20]);
  assert.deepEqual(resultant.geometry.origin, [4, 7, 26]);
});

test("patch colors and CI gap do not move LBCP or force origins and reset exactly", () => {
  const ci = { kind: "contact_patch", geometry: { z_range: [23,26], lbcp_display_z: 22 },
    style: { color: "#ffaa00", pattern: "diagonal_hatch", opacity: 0.65 } };
  const lbcp = { kind: "support_patch", geometry: { z_range: [20,22] },
    style: { color: "#ffd700", opacity: 1 } };
  const source = [arrow, ci, lbcp];
  const before = JSON.stringify(source);
  const result = displayObjects(source, { ...DEFAULT_DISPLAY_OPTIONS,
    ciColor: "#ff0000", lbcpColor: "#0000ff", ciGap: 0 });
  assert.equal(result[1].style.color, "#ff0000");
  assert.equal(result[2].style.color, "#0000ff");
  assert.equal(result[1].geometry.z_range[1], 22);
  assert.equal(result[1].style.pattern, "diagonal_hatch");
  assert.deepEqual(result[2].geometry, lbcp.geometry);
  assert.deepEqual(result[0].geometry.origin, arrow.geometry.origin);
  assert.equal(JSON.stringify(source), before);
  assert.deepEqual(displayObjects(source, DEFAULT_DISPLAY_OPTIONS)[1], ci);
});

test("CI and LBCP alpha are independent, support zero and reset to source values", () => {
  const patches = [...objects, { kind: "support_patch", style: { opacity: 0.4 } }];
  const before = JSON.stringify(patches);
  const result = displayObjects(patches, { ...DEFAULT_DISPLAY_OPTIONS, ciAlpha: 0, lbcpAlpha: 0.8 });
  assert.equal(result[3].style.opacity, 0);
  assert.equal(result[4].style.opacity, 0.8);
  assert.equal(result[0].style.opacity, 0.8);
  assert.equal(result[2].style.opacity, 0.5);
  assert.equal(JSON.stringify(patches), before);
  const reset = displayObjects(patches, DEFAULT_DISPLAY_OPTIONS);
  assert.equal(reset[3].style.opacity, 0.65);
  assert.equal(reset[4].style.opacity, 0.4);
});

test("display overrides leave LP data and patches unchanged, including zero alpha", () => {
  const before = JSON.stringify(objects);
  const result = displayObjects(objects, { ...DEFAULT_DISPLAY_OPTIONS,
    length: 2, shaft: 0.5, head: 1.5, arrowAlpha: 0, boxAlpha: 0, containerWire: false });
  assert.equal(result.length, 3);
  assert.equal(result[0].geometry.display_length, 200);
  assert.equal(result[0].geometry.force_newtons, 42);
  assert.equal(result[0].style.shaft_scale, 0.5);
  assert.equal(result[0].style.head_scale, 1.5);
  assert.equal(result[0].style.opacity, 0);
  assert.equal(result[1].style.opacity, 0);
  assert.deepEqual(result[2], objects[3]);
  assert.equal(JSON.stringify(objects), before);
  assert.deepEqual(displayObjects(objects, DEFAULT_DISPLAY_OPTIONS)[2], objects[2]);
});

test("shaft slider changes cylinder without resizing cone; head slider preserves shaft", () => {
  const base = createPrimitive(arrow);
  const thin = createPrimitive(displayObjects([arrow], { ...DEFAULT_DISPLAY_OPTIONS, shaft: 0.5 })[0]);
  const large = createPrimitive(displayObjects([arrow], { ...DEFAULT_DISPLAY_OPTIONS, head: 1.5 })[0]);
  assert.equal(thin.children[0].scale.x, base.children[0].scale.x * 0.5);
  assert.deepEqual(thin.children[1].scale, base.children[1].scale);
  assert.equal(large.children[0].scale.x, base.children[0].scale.x);
  assert.ok(large.children[1].scale.x > base.children[1].scale.x);
});

test("view retains overrides and rebuilds the last frame without changing its data", () => {
  const view = Object.create(SceneView.prototype);
  view.lastSceneFrame = { objects };
  let frame;
  view.replace = (value) => { frame = value; };
  view.setDisplayOptions({ ...DEFAULT_DISPLAY_OPTIONS, boxAlpha: 0.2 });
  assert.equal(frame, view.lastSceneFrame);
  assert.equal(view.displayOptions.boxAlpha, 0.2);
  assert.equal(objects[2].style.opacity, 0.5);
});
