import { normalizeFrame } from "./helpers.js";
import { SceneView } from "./scene-view.js";
import { attachDisplayControls } from "./display-controls.js";

export function applyViewsSafely(views, scenes, report = console.error) {
  const failures = [];
  for (const [name, view] of Object.entries(views)) {
    try {
      view.replace(scenes[name]);
    } catch (error) {
      failures.push({ name, error });
      report(`Unable to update ${name} scene`, error);
    }
  }
  return failures;
}

export class FrameRenderer {
  constructor(root = document) {
    this.disposed = false;
    this.reportViewError = console.error;
    this.title = root.querySelector("#frame-title");
    this.metrics = root.querySelector("#frame-metrics");
    this.legend = root.querySelector("#legend");
    this.views = Object.fromEntries(["container", "buffer", "holding"].map((name) => [name, new SceneView(root.querySelector(`[data-scene="${name}"]`), name)]));
    this.exportButton = root.querySelector("#export-container");
    this.exportHandler = () => {
      try {
        const link = document.createElement("a");
        link.href = this.views.container.exportPNG();
        link.download = `container-${new Date().toISOString().replaceAll(":", "-")}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();
      } catch (error) {
        console.error("Container PNG export failed", error);
        window.alert(`Container PNG export failed: ${error.message}`);
      }
    };
    this.exportButton?.addEventListener("click", this.exportHandler);
    this.displayControls = attachDisplayControls(
      root.querySelector(".scene-panel.container"),
      (options) => this.views.container.setDisplayOptions(options),
    );
  }

  apply(payload) {
    if (this.disposed) return;
    const frame = normalizeFrame(payload);
    this.displayControls?.update(frame.scenes.container.objects);
    if (this.title) this.title.textContent = frame.title;
    if (this.metrics) {
      const metrics = frame.metrics || {};
      const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
      const holding = Number.isInteger(metrics.holding_count) ? metrics.holding_count : "—";
      this.metrics.textContent = `利用率 ${percent(metrics.utilization)} · 含间隙 ${percent(metrics.virtual_utilization)} · Holding ${holding}`;
    }
    applyViewsSafely(this.views, frame.scenes, this.reportViewError);
    this.renderLegend(frame.legend);
  }

  renderLegend(entries) {
    if (!this.legend) return;
    this.legend.replaceChildren(...entries.map((entry) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("i");
      swatch.style.background = entry.color || "#94a3b8";
      item.append(swatch, document.createTextNode(entry.label || entry.kind || "Object"));
      return item;
    }));
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.exportButton?.removeEventListener("click", this.exportHandler);
    this.displayControls?.dispose();
    for (const view of Object.values(this.views)) view.dispose();
  }
}
