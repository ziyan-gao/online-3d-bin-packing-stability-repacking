import { Color } from "three";

export const DEFAULT_DISPLAY_OPTIONS = Object.freeze({
  shaft: 1, head: 1, length: 1, arrowAlpha: null, boxAlpha: null, containerWire: true,
  ciAlpha: null, lbcpAlpha: null,
  ciColor: null, lbcpColor: null, ciGap: null,
});

export function displayObjects(objects, options = DEFAULT_DISPLAY_OPTIONS) {
  return objects.filter((object) => object.kind !== "container_wire" || options.containerWire !== false)
    .map((object) => {
      if (object.kind === "force_arrow") {
        const geometry = { ...object.geometry, display_length: object.geometry.display_length * options.length };
        if (geometry.representation === "ci_resultant" && options.ciGap != null && geometry.lbcp_display_z != null) {
          geometry.origin = [geometry.origin[0], geometry.origin[1], geometry.lbcp_display_z + options.ciGap];
        }
        return { ...object,
          geometry,
          style: { ...object.style, shaft_scale: options.shaft, head_scale: options.head,
            opacity: options.arrowAlpha ?? object.style?.opacity ?? 1 },
        };
      }
      if (["item", "highlighted_item", "repacked_item"].includes(object.kind) && options.boxAlpha !== null) {
        return { ...object, style: { ...object.style, opacity: options.boxAlpha } };
      }
      const patchAlpha = object.kind === "contact_patch" ? options.ciAlpha
        : object.kind === "support_patch" ? options.lbcpAlpha : null;
      if (["contact_patch", "support_patch"].includes(object.kind)) {
        const ci = object.kind === "contact_patch";
        const patchColor = ci ? options.ciColor : options.lbcpColor;
        const style = { ...object.style };
        if (patchAlpha != null) style.opacity = patchAlpha;
        if (patchColor != null) style.color = patchColor;
        let geometry = object.geometry;
        if (ci && options.ciGap != null && geometry?.lbcp_display_z != null) {
          const shift = geometry.lbcp_display_z + options.ciGap - geometry.z_range[1];
          geometry = { ...geometry, z_range: geometry.z_range.map((z) => z + shift) };
        }
        return { ...object, ...(geometry ? { geometry } : {}), style };
      }
      return object;
    });
}

export function attachDisplayControls(host, onChange) {
  if (!host) return null;
  const doc = host.ownerDocument;
  const panel = doc.createElement("details");
  panel.className = "display-controls";
  const summary = doc.createElement("summary");
  summary.textContent = "显示设置";
  panel.append(summary);
  let options = { ...DEFAULT_DISPLAY_OPTIONS };
  let currentObjects = [];
  const inputs = new Map();
  const colorInputs = new Map();
  const listeners = [];
  const listen = (node, type, handler) => {
    node.addEventListener(type, handler);
    listeners.push(() => node.removeEventListener(type, handler));
  };
  const definitions = [
    ["shaft", "箭杆粗细", 0.1, 3, 0.05],
    ["head", "圆锥大小", 0.25, 3, 0.05],
    ["length", "箭头长度", 0.1, 3, 0.05],
    ["arrowAlpha", "箭头 alpha", 0, 1, 0.05],
    ["boxAlpha", "Box alpha", 0, 1, 0.05],
    ["ciAlpha", "CI alpha", 0, 1, 0.05],
    ["lbcpAlpha", "LBCP alpha", 0, 1, 0.05],
    ["ciGap", "CI–LBCP 间距", 0, 30, 0.5],
  ];
  const sourceAlpha = (kind, fallback) =>
    currentObjects.find((object) => object.kind === kind)?.style?.opacity ?? fallback;
  const alphaSources = {
    arrowAlpha: ["force_arrow", 1], boxAlpha: ["item", 0.5],
    ciAlpha: ["contact_patch", 0.65], lbcpAlpha: ["support_patch", 1],
  };
  const sync = () => {
    for (const [key, { input, output }] of inputs) {
      const ci = currentObjects.find((object) => object.kind === "contact_patch");
      const defaultGap = ci?.geometry?.lbcp_display_z != null
        ? ci.geometry.z_range[1] - ci.geometry.lbcp_display_z : 4;
      const value = options[key] ?? (key === "ciGap" ? defaultGap : sourceAlpha(...alphaSources[key]));
      input.value = String(value);
      output.value = Number(value).toFixed(2) + (key === "ciGap" ? " mm" : ["shaft", "head", "length"].includes(key) ? "×" : "");
    }
    for (const [key, { input, kind }] of colorInputs) {
      const source = currentObjects.find((object) => object.kind === kind)?.style?.color ?? "#ffd700";
      input.value = options[key] ?? ("#" + new Color(source).getHexString());
    }
    checkbox.checked = options.containerWire;
  };
  for (const [key, title, min, max, step] of definitions) {
    const label = doc.createElement("label");
    const caption = doc.createElement("span");
    caption.textContent = title;
    const input = doc.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.setAttribute("aria-label", title);
    const output = doc.createElement("output");
    inputs.set(key, { input, output });
    listen(input, "input", () => {
      options[key] = Number(input.value);
      sync();
      onChange({ ...options });
    });
    label.append(caption, input, output);
    panel.append(label);
  }
  for (const [key, title, kind] of [["ciColor", "CI 颜色", "contact_patch"], ["lbcpColor", "LBCP 颜色", "support_patch"]]) {
    const label = doc.createElement("label");
    const caption = doc.createElement("span");
    caption.textContent = title;
    const input = doc.createElement("input");
    input.type = "color";
    input.setAttribute("aria-label", title);
    colorInputs.set(key, { input, kind });
    listen(input, "input", () => {
      options[key] = input.value;
      onChange({ ...options });
    });
    label.append(caption, input);
    panel.append(label);
  }
  const wireLabel = doc.createElement("label");
  wireLabel.className = "wire-control";
  const checkbox = doc.createElement("input");
  checkbox.type = "checkbox";
  listen(checkbox, "change", () => {
    options.containerWire = checkbox.checked;
    onChange({ ...options });
  });
  wireLabel.append(checkbox, doc.createTextNode("显示 Container 边框"));
  const reset = doc.createElement("button");
  reset.type = "button";
  reset.textContent = "重置";
  listen(reset, "click", () => {
    options = { ...DEFAULT_DISPLAY_OPTIONS };
    sync();
    onChange({ ...options });
  });
  panel.append(wireLabel, reset);
  host.append(panel);
  sync();
  return {
    update(objects) { currentObjects = objects; sync(); },
    dispose() { listeners.forEach((remove) => remove()); panel.remove(); },
  };
}
