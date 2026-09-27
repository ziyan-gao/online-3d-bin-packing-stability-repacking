import test from "node:test";
import assert from "node:assert/strict";

import {
  formatTooltip,
  normalizeFrame,
  orderSceneObjects,
  toThreeCoordinates,
} from "../src/helpers.js";
import { applyViewsSafely, FrameRenderer } from "../src/frame-renderer.js";

test("maps physical XYZ into Three.js XZY coordinates", () => {
  assert.deepEqual(toThreeCoordinates([1, 2, 3]), [1, 3, 2]);
});

test("orders maximum force arrows before minimum force arrows", () => {
  const objects = [
    { id: "min", kind: "force_arrow", geometry: { scenario: "force_min" } },
    { id: "item", kind: "item", geometry: {} },
    { id: "max", kind: "force_arrow", geometry: { scenario: "force_max" } },
  ];
  assert.deepEqual(orderSceneObjects(objects).map(({ id }) => id), ["item", "max", "min"]);
});

test("formats tooltip metadata with readable labels and values", () => {
  assert.equal(
    formatTooltip({ type: "force_arrow", force_n: 12.5, vertex_mm: [1, 2, 3] }),
    "Force arrow\nForce: 12.5 N\nVertex: 1 × 2 × 3 mm",
  );
});

test("formats contact pressure as an equivalent diagnostic rather than area", () => {
  assert.equal(
    formatTooltip({
      type: "contact_patch",
      area_mm2: 125,
      max_equivalent_average_pressure_n_per_mm2: 0.04,
      color_meaning: "max-equivalent average pressure",
      pressure_diagnostic: "not local pressure",
    }),
    [
      "Contact patch",
      "Area: 125 mm²",
      "Maximum equivalent average pressure: 0.04 N/mm²",
      "Color meaning: max-equivalent average pressure",
      "Pressure diagnostic: Diagnostic only; not local pressure",
    ].join("\n"),
  );
});

test("uses pressure units before the generic area suffix rule", () => {
  assert.equal(
    formatTooltip({ type: "contact_patch", pressure_n_per_mm2: 2.5 }),
    "Contact patch\nPressure: 2.5 N/mm²",
  );
});

test("normalizes missing and malformed payload collections", () => {
  const frame = normalizeFrame({ title: "Empty", scenes: { container: null } });
  assert.equal(frame.title, "Empty");
  assert.deepEqual(frame.legend, []);
  assert.deepEqual(frame.scenes.container.objects, []);
  assert.deepEqual(frame.scenes.buffer.objects, []);
  assert.deepEqual(frame.scenes.holding.objects, []);
});

test("continues applying available views when one view is unavailable", () => {
  const applied = [];
  const views = {
    container: { replace() { throw new Error("WebGL unavailable"); } },
    buffer: { replace(scene) { applied.push(scene.name); } },
    holding: { replace(scene) { applied.push(scene.name); } },
  };
  const scenes = {
    container: { name: "container" },
    buffer: { name: "buffer" },
    holding: { name: "holding" },
  };

  const failures = applyViewsSafely(views, scenes, () => {});

  assert.deepEqual(applied, ["buffer", "holding"]);
  assert.equal(failures.length, 1);
  assert.equal(failures[0].name, "container");
});

test("frame application still updates the legend when every view is unavailable", () => {
  const renderer = Object.create(FrameRenderer.prototype);
  renderer.title = { textContent: "" };
  renderer.views = Object.fromEntries(["container", "buffer", "holding"].map((name) => [
    name,
    { replace() { throw new Error(`${name} unavailable`); } },
  ]));
  renderer.reportViewError = () => {};
  let renderedLegend = null;
  renderer.renderLegend = (legend) => { renderedLegend = legend; };

  renderer.apply({ title: "Still useful", legend: [{ kind: "item", label: "Item" }] });

  assert.equal(renderer.title.textContent, "Still useful");
  assert.deepEqual(renderedLegend, [{ kind: "item", label: "Item" }]);
});
