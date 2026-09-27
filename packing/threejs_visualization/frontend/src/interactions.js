import * as THREE from "three";
import { formatTooltip } from "./helpers.js";

export function attachInteractions(canvas, camera, content, host) {
  const raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = 5;
  const pointer = new THREE.Vector2();
  const tooltip = document.createElement("div");
  tooltip.className = "scene-tooltip";
  host.appendChild(tooltip);

  function hide() { tooltip.hidden = true; }
  function move(event) {
    const bounds = canvas.getBoundingClientRect();
    pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(content.children, true).find(({ object }) => object.userData?.tooltip);
    const text = hit ? formatTooltip(hit.object.userData.tooltip) : "";
    if (!text) return hide();
    tooltip.textContent = text;
    tooltip.hidden = false;
    tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
    tooltip.style.top = `${event.clientY - bounds.top + 12}px`;
  }
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerleave", hide);
  hide();
  return () => { canvas.removeEventListener("pointermove", move); canvas.removeEventListener("pointerleave", hide); tooltip.remove(); };
}
