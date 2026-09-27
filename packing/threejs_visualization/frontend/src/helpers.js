const SCENE_NAMES = ["container", "buffer", "holding"];

export function toThreeCoordinates(point = []) {
  const [x = 0, y = 0, z = 0] = point;
  return [Number(x), Number(z), Number(y)];
}

export function orderSceneObjects(objects = []) {
  const rank = (object) => {
    if (object?.kind !== "force_arrow") return 0;
    return object.geometry?.scenario === "force_max" ? 1 : 2;
  };
  return [...objects].sort((left, right) => rank(left) - rank(right));
}

function humanize(key) {
  const labels = {
    force_n: "Force",
    vertex_mm: "Vertex",
    dimensions_mm: "Dimensions",
    virtual_dimensions_mm: "Virtual dimensions",
    displayed_box: "Displayed box",
    position_mm: "Position",
    area_mm2: "Area",
    force_min_n: "Minimum force",
    force_max_n: "Maximum force",
    pressure_n_per_mm2: "Pressure",
    max_equivalent_average_pressure_n_per_mm2: "Maximum equivalent average pressure",
    color_meaning: "Color meaning",
    pressure_diagnostic: "Pressure diagnostic",
  };
  return labels[key] || key.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function formatValue(key, value) {
  const units = key.endsWith("_n_per_mm2") ? " N/mm²" : key.endsWith("_mm2") ? " mm²" : key.endsWith("_n") ? " N" : key.endsWith("_mm") ? " mm" : "";
  const body = Array.isArray(value) ? value.join(" × ") : typeof value === "object" ? JSON.stringify(value) : String(value);
  if (key === "pressure_diagnostic" && !/diagnostic/i.test(body)) return `Diagnostic only; ${body}`;
  return `${body}${units}`;
}

export function formatTooltip(metadata = {}) {
  const entries = Object.entries(metadata).filter(([, value]) => value !== undefined && value !== null);
  if (!entries.length) return "";
  const type = entries.find(([key]) => key === "type")?.[1];
  const title = type ? humanize(String(type)) : "Object";
  const rows = entries
    .filter(([key]) => key !== "type")
    .map(([key, value]) => `${humanize(key)}: ${formatValue(key, value)}`);
  return [title, ...rows].join("\n");
}

function normalizeScene(scene) {
  const source = scene && typeof scene === "object" ? scene : {};
  return {
    ...source,
    bounds: Array.isArray(source.bounds) && source.bounds.length === 6 ? source.bounds : [0, 0, 0, 1, 1, 1],
    camera: source.camera && typeof source.camera === "object" ? source.camera : {},
    objects: Array.isArray(source.objects) ? source.objects : [],
  };
}

export function normalizeFrame(payload) {
  const source = payload && typeof payload === "object" ? payload : {};
  const scenes = {};
  for (const name of SCENE_NAMES) scenes[name] = normalizeScene(source.scenes?.[name]);
  return {
    ...source,
    title: typeof source.title === "string" ? source.title : "Packing visualization",
    legend: Array.isArray(source.legend) ? source.legend : [],
    scenes,
  };
}
