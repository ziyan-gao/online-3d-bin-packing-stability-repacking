    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";

    // DOM handles and browser-side simulator state.
    const plot = document.getElementById("plot");
    const bufferPlot = document.getElementById("bufferPlot");
    const rot0 = document.getElementById("rot0");
    const rot1 = document.getElementById("rot1");
    const anchorMode = document.getElementById("anchorMode");
    const gridMode = document.getElementById("gridMode");
    const supportToggle = document.getElementById("supportToggle");
    const resetBtn = document.getElementById("reset");
    const applySize = document.getElementById("applySize");
    const containerDx = document.getElementById("containerDx");
    const containerDy = document.getElementById("containerDy");
    const containerDz = document.getElementById("containerDz");
    const sameItemHeight = document.getElementById("sameItemHeight");
    const fixedItemHeight = document.getElementById("fixedItemHeight");
    const PLACED_ITEM_OPACITY = 0.86;
    let state = null;
    let showSupport = true;
    let placementMode = "anchor";
    let hoverCandidate = null;
    let requestInFlight = false;
    let threeRenderer = null;
    let threeScene = null;
    let threeCamera = null;
    let threeControls = null;
    let raycaster = null;
    let pointer = null;
    let containerGroup = null;
    let placedGroup = null;
    let supportGroup = null;
    let candidateGroup = null;
    let previewGroup = null;
    let hoveredCandidateMesh = null;
    let threeReady = false;
    let resetCameraOnNextRender = false;

    // Small formatting and color helpers shared by the two Plotly scenes.
    function dimsText(item) {
      if (!item) return "-";
      return `${item.dx} x ${item.dy} x ${item.dz}`;
    }

    function utilizationText() {
      return `${(state.utilization * 100).toFixed(1)}%`;
    }

    function activeActions() {
      return state.actions.filter(a => a.rotation === state.rotation);
    }

    function colorForDims(item) {
      const key = (item.dx * 3 + item.dy * 5 + item.dz * 7) % 360;
      const [r, g, b] = hslToRgb(key / 360, 0.68, 0.56);
      return `rgb(${r}, ${g}, ${b})`;
    }

    function hslToRgb(h, s, l) {
      let r, g, b;
      if (s === 0) {
        r = g = b = l;
      } else {
        const hue2rgb = (p, q, t) => {
          if (t < 0) t += 1;
          if (t > 1) t -= 1;
          if (t < 1 / 6) return p + (q - p) * 6 * t;
          if (t < 1 / 2) return q;
          if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
          return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
      }
      return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    }

    // Right-side buffer items still use Plotly surface traces.
    function colorScale(color) {
      return [[0, color], [1, color]];
    }

    function boxSurfaceTraces(item, color, name, scene = undefined) {
      const x0 = item.x, x1 = item.x + item.dx;
      const y0 = item.y, y1 = item.y + item.dy;
      const z0 = item.z, z1 = item.z + item.dz;
      const faces = [
        { x: [[x0, x1], [x0, x1]], y: [[y0, y0], [y1, y1]], z: [[z0, z0], [z0, z0]] },
        { x: [[x0, x1], [x0, x1]], y: [[y0, y0], [y1, y1]], z: [[z1, z1], [z1, z1]] },
        { x: [[x0, x1], [x0, x1]], y: [[y0, y0], [y0, y0]], z: [[z0, z0], [z1, z1]] },
        { x: [[x0, x1], [x0, x1]], y: [[y1, y1], [y1, y1]], z: [[z0, z0], [z1, z1]] },
        { x: [[x0, x0], [x0, x0]], y: [[y0, y1], [y0, y1]], z: [[z0, z0], [z1, z1]] },
        { x: [[x1, x1], [x1, x1]], y: [[y0, y1], [y0, y1]], z: [[z0, z0], [z1, z1]] },
      ];
      return faces.map(face => ({
        type: "surface",
        name,
        x: face.x,
        y: face.y,
        z: face.z,
        surfacecolor: [[1, 1], [1, 1]],
        colorscale: colorScale(color),
        cmin: 0,
        cmax: 1,
        opacity: 1,
        showscale: false,
        hoverinfo: "none",
        showlegend: false,
        lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1, fresnel: 0 },
        scene,
      }));
    }

    function boxOutlineTrace(item, name, color = "rgba(15,23,42,0.9)", scene = undefined) {
      const x0 = item.x, x1 = item.x + item.dx;
      const y0 = item.y, y1 = item.y + item.dy;
      const z0 = item.z, z1 = item.z + item.dz;
      const p = [
        [x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0], [x0,y0,z0],
        [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1], [x0,y0,z1],
        [x1,y0,z1], [x1,y0,z0], [x1,y1,z0], [x1,y1,z1],
        [x0,y1,z1], [x0,y1,z0], [x0,y0,z0], [x0,y0,z1]
      ];
      const trace = wireTrace(p, `${name} edge`, color, 4);
      trace.scene = scene;
      return trace;
    }

    function wireTrace(points, name, color = "rgba(17,24,39,0.72)", width = 4) {
      return {
        type: "scatter3d",
        mode: "lines",
        name,
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        z: points.map(p => p[2]),
        line: { color, width },
        hoverinfo: "none",
        showlegend: false,
      };
    }

    function sameCandidate(left, right) {
      return Boolean(
        left && right &&
        left.x === right.x &&
        left.y === right.y &&
        left.z === right.z &&
        left.rotation === right.rotation
      );
    }

    // Three.js main scene. Plotly stays on the right-side buffer view, while
    // the bin uses native 3D picking so hover does not rebuild a Plotly figure.
    function toThreePoint(x, y, z) {
      return new THREE.Vector3(x, z, y);
    }

    function threeColor(cssColor) {
      return new THREE.Color(cssColor.replace(/\s+/g, ""));
    }

    function clearGroup(group) {
      while (group.children.length) {
        const child = group.children.pop();
        child.traverse?.(obj => {
          obj.geometry?.dispose?.();
          if (Array.isArray(obj.material)) {
            obj.material.forEach(mat => mat.dispose?.());
          } else {
            obj.material?.dispose?.();
          }
        });
      }
    }

    function makeLine(points, color = 0x111827, linewidth = 2) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color, linewidth });
      return new THREE.Line(geometry, material);
    }

    function addBox(group, item, color, opacity = 1, outline = true) {
      const geometry = new THREE.BoxGeometry(item.dx, item.dz, item.dy);
      const material = new THREE.MeshLambertMaterial({
        color: threeColor(color),
        opacity,
        transparent: opacity < 1,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(item.x + item.dx / 2, item.z + item.dz / 2, item.y + item.dy / 2);
      group.add(mesh);
      if (outline) {
        const edges = new THREE.EdgesGeometry(geometry);
        const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x111827 });
        const lines = new THREE.LineSegments(edges, edgeMaterial);
        lines.position.copy(mesh.position);
        group.add(lines);
      }
      return mesh;
    }

    function addContainerWire() {
      clearGroup(containerGroup);
      const c = state.container;
      const geometry = new THREE.BoxGeometry(c.dx, c.dz, c.dy);
      const edges = new THREE.EdgesGeometry(geometry);
      const material = new THREE.LineBasicMaterial({ color: 0x111827 });
      const wire = new THREE.LineSegments(edges, material);
      wire.position.set(c.dx / 2, c.dz / 2, c.dy / 2);
      containerGroup.add(wire);
      const floorGrid = new THREE.GridHelper(
        Math.max(c.dx, c.dy),
        Math.max(4, Math.floor(Math.max(c.dx, c.dy) / 60)),
        0xcbd5e1,
        0xe2e8f0
      );
      floorGrid.position.set(c.dx / 2, 0, c.dy / 2);
      containerGroup.add(floorGrid);
    }

    function addSupportPatch(record) {
      const points = record.supportPolygon || [];
      if (points.length < 3) return;
      const z = record.z1 + 3;
      const shape = new THREE.Shape(points.map(point => new THREE.Vector2(point[0], point[1])));
      const geometry = new THREE.ShapeGeometry(shape);
      const material = new THREE.MeshBasicMaterial({
        color: 0xffd700,
        side: THREE.DoubleSide,
        transparent: false,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = Math.PI / 2;
      mesh.position.y = z;
      supportGroup.add(mesh);

      const loop = points.concat([points[0]]).map(point => toThreePoint(point[0], point[1], z + 1));
      const border = makeLine(loop, 0x000000);
      supportGroup.add(border);
    }

    function candidateItems() {
      if (placementMode === "anchor") {
        return activeActions().map(action => ({ ...action, valid: true, placeable: true, stable: true }));
      }
      return state.gridCandidates || [];
    }

    function addCandidateMarkers() {
      clearGroup(candidateGroup);
      const sphere = new THREE.SphereGeometry(7, 16, 16);
      for (const candidate of candidateItems()) {
        const isHovered = sameCandidate(candidate, hoverCandidate);
        const color = isHovered ? 0xfacc15 : (candidate.valid ? 0x0f766e : 0x64748b);
        const material = new THREE.MeshBasicMaterial({
          color,
          transparent: !candidate.valid,
          opacity: candidate.valid ? 1 : 0.35,
          depthTest: false,
        });
        const marker = new THREE.Mesh(sphere, material);
        marker.position.copy(toThreePoint(candidate.x, candidate.y, candidate.z + candidate.dz + 12));
        marker.scale.setScalar(isHovered ? 1.7 : (placementMode === "anchor" ? 1.15 : 0.75));
        marker.userData.candidate = candidate;
        candidateGroup.add(marker);
      }
    }

    function updatePreview() {
      clearGroup(previewGroup);
      if (hoverCandidate) {
        addBox(
          previewGroup,
          hoverCandidate,
          hoverCandidate.valid ? "rgb(20,184,166)" : "rgb(239,68,68)",
          hoverCandidate.valid ? 0.82 : 0.24,
          true
        );
      }
      addCandidateMarkers();
      renderThree();
    }

    function initializeThreeScene() {
      if (threeReady) return;
      threeScene = new THREE.Scene();
      threeScene.background = new THREE.Color(0xffffff);
      threeCamera = new THREE.PerspectiveCamera(45, plot.clientWidth / plot.clientHeight, 1, 6000);
      threeCamera.position.set(900, 760, 900);
      threeRenderer = new THREE.WebGLRenderer({ antialias: true });
      threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      threeRenderer.setSize(plot.clientWidth, plot.clientHeight);
      plot.innerHTML = "";
      plot.appendChild(threeRenderer.domElement);

      threeControls = new OrbitControls(threeCamera, threeRenderer.domElement);
      threeControls.enableDamping = true;
      threeControls.target.set(300, 260, 300);

      const ambient = new THREE.AmbientLight(0xffffff, 0.78);
      const directional = new THREE.DirectionalLight(0xffffff, 0.65);
      directional.position.set(700, 1200, 900);
      threeScene.add(ambient, directional);

      containerGroup = new THREE.Group();
      placedGroup = new THREE.Group();
      supportGroup = new THREE.Group();
      candidateGroup = new THREE.Group();
      previewGroup = new THREE.Group();
      threeScene.add(containerGroup, placedGroup, supportGroup, candidateGroup, previewGroup);

      raycaster = new THREE.Raycaster();
      pointer = new THREE.Vector2();
      threeRenderer.domElement.addEventListener("pointermove", onThreePointerMove);
      threeRenderer.domElement.addEventListener("pointerleave", () => {
        hoverCandidate = null;
        hoveredCandidateMesh = null;
        updatePreview();
      });
      threeRenderer.domElement.addEventListener("click", onThreeClick);
      window.addEventListener("resize", resizeThreeScene);
      threeRenderer.setAnimationLoop(() => {
        threeControls.update();
        threeRenderer.render(threeScene, threeCamera);
      });
      threeReady = true;
    }

    function resizeThreeScene() {
      if (!threeReady || !plot.clientWidth || !plot.clientHeight) return;
      threeCamera.aspect = plot.clientWidth / plot.clientHeight;
      threeCamera.updateProjectionMatrix();
      threeRenderer.setSize(plot.clientWidth, plot.clientHeight);
      renderThree();
    }

    function renderThree() {
      if (!threeReady) return;
      threeRenderer.render(threeScene, threeCamera);
    }

    function pickCandidate(event) {
      const rect = threeRenderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, threeCamera);
      return raycaster.intersectObjects(candidateGroup.children, false)[0]?.object || null;
    }

    function onThreePointerMove(event) {
      if (requestInFlight) return;
      const mesh = pickCandidate(event);
      if (mesh === hoveredCandidateMesh) return;
      hoveredCandidateMesh = mesh;
      hoverCandidate = mesh ? { ...mesh.userData.candidate } : null;
      updatePreview();
    }

    function onThreeClick(event) {
      if (requestInFlight) return;
      const mesh = pickCandidate(event);
      if (!mesh) return;
      const candidate = mesh.userData.candidate;
      document.getElementById("message").textContent =
        `Placing at (${candidate.x}, ${candidate.y}, ${candidate.z})...`;
      hoverCandidate = null;
      hoveredCandidateMesh = null;
      if (placementMode === "anchor") {
        request("/place", { x: candidate.x, y: candidate.y, rotation: candidate.rotation });
      } else {
        request("/grid-place", { x: candidate.x, y: candidate.y, rotation: candidate.rotation });
      }
    }

    function resetThreeCamera() {
      if (!threeReady || !state) return;
      const c = state.container;
      threeControls.target.set(c.dx / 2, c.dz / 2, c.dy / 2);
      threeCamera.position.set(c.dx * 1.55, c.dz * 1.35, c.dy * 1.7);
      threeCamera.near = 1;
      threeCamera.far = Math.max(c.dx, c.dy, c.dz) * 10;
      threeCamera.updateProjectionMatrix();
      threeControls.update();
    }

    function renderThreeScene(resetCamera = false) {
      if (!state) return;
      const wasReady = threeReady;
      initializeThreeScene();
      clearGroup(placedGroup);
      clearGroup(supportGroup);
      clearGroup(previewGroup);
      addContainerWire();
      for (const item of state.placed) {
        addBox(placedGroup, item, colorForDims(item), PLACED_ITEM_OPACITY, true);
      }
      if (showSupport) {
        for (const record of state.support || []) {
          addSupportPatch(record);
        }
      }
      addCandidateMarkers();
      if (hoverCandidate) {
        addBox(
          previewGroup,
          hoverCandidate,
          hoverCandidate.valid ? "rgb(20,184,166)" : "rgb(239,68,68)",
          hoverCandidate.valid ? 0.82 : 0.24,
          true
        );
      }
      if (resetCamera || !wasReady) {
        resetThreeCamera();
      }
      resizeThreeScene();
      renderThree();
    }

    function renderScene() {
      renderThreeScene(resetCameraOnNextRender);
      resetCameraOnNextRender = false;
    }

    // Right-side buffer scene.
    function bufferBoxFromItem(item, idx, xOffset) {
      return {
        id: idx + 1,
        x: xOffset,
        y: 0,
        z: 0,
        dx: item.dx,
        dy: item.dy,
        dz: item.dz,
      };
    }

    function renderBufferScene() {
      if (!state) return;
      const items = [];
      if (state.currentItem) {
        items.push({ item: state.currentItem, label: "Current", current: true });
      }
      state.buffer.forEach((item, idx) => {
        items.push({ item, label: `Buffer ${idx + 1}`, current: false });
      });

      const traces = [];
      let xOffset = 0;
      let maxX = 300;
      let maxY = 300;
      let maxZ = 300;
      items.forEach((entry, idx) => {
        const box = bufferBoxFromItem(entry.item, idx, xOffset);
        traces.push(...boxSurfaceTraces(box, colorForDims(box), entry.label));
        traces.push(boxOutlineTrace(box, entry.label));
        xOffset += box.dx + 110;
        maxX = Math.max(maxX, xOffset);
        maxY = Math.max(maxY, box.dy);
        maxZ = Math.max(maxZ, box.dz);
      });

      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        showlegend: false,
        uirevision: "buffer-scene",
        scene: {
          xaxis: { title: "", range: [-20, maxX], backgroundcolor: "rgb(248,250,252)", gridcolor: "rgb(226,232,240)" },
          yaxis: { title: "", range: [-20, maxY + 40], backgroundcolor: "rgb(248,250,252)", gridcolor: "rgb(226,232,240)" },
          zaxis: { title: "", range: [-20, maxZ + 40], backgroundcolor: "rgb(248,250,252)", gridcolor: "rgb(226,232,240)" },
          aspectmode: "manual",
          aspectratio: {
            x: maxX / Math.max(maxX, maxY + 40, maxZ + 40),
            y: (maxY + 40) / Math.max(maxX, maxY + 40, maxZ + 40),
            z: (maxZ + 40) / Math.max(maxX, maxY + 40, maxZ + 40),
          },
          camera: { eye: { x: 1.55, y: 1.45, z: 0.95 } },
        },
      };
      Plotly.react(bufferPlot, traces, layout, { responsive: true, displaylogo: false });
    }

    // Browser events and API calls.
    async function request(path, payload = null) {
      requestInFlight = true;
      const options = payload
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
        : {};
      try {
        const response = await fetch(path, options);
        state = await response.json();
        if (state.error) {
          state.message = state.error;
        }
        renderSidebar();
        renderScene();
        renderBufferScene();
      } finally {
        requestInFlight = false;
      }
    }

    function renderSidebar() {
      document.getElementById("utilization").textContent = `Utilization: ${utilizationText()}`;
      document.getElementById("binUtilization").textContent = utilizationText();
      document.getElementById("placedCount").textContent = `${state.placed.length}`;
      containerDx.value = state.container.dx;
      containerDy.value = state.container.dy;
      containerDz.value = state.container.dz;
      sameItemHeight.checked = Boolean(state.sameItemHeight);
      fixedItemHeight.textContent = state.sameItemHeight ? `fixed ${state.fixedItemHeight} mm` : "original";
      document.getElementById("currentItem").textContent = dimsText(state.currentItem);
      document.getElementById("message").textContent = state.message || "";
      rot0.classList.toggle("active", state.rotation === 0);
      rot1.classList.toggle("active", state.rotation === 1);
      anchorMode.classList.toggle("active", placementMode === "anchor");
      gridMode.classList.toggle("active", placementMode === "grid");
      supportToggle.classList.toggle("active", showSupport);
      const list = document.getElementById("bufferList");
      list.innerHTML = "";
      state.buffer.forEach((item, idx) => {
        const row = document.createElement("div");
        row.className = "buffer-item";
        row.innerHTML = `
          <span class="buffer-swatch" style="background:${colorForDims(item)}"></span>
          <span>${idx + 1}</span>
          <span class="buffer-dims">${dimsText(item)}</span>
        `;
        list.appendChild(row);
      });
    }

    rot0.addEventListener("click", () => request("/rotation", { rotation: 0 }));
    rot1.addEventListener("click", () => request("/rotation", { rotation: 1 }));
    anchorMode.addEventListener("click", () => {
      placementMode = "anchor";
      hoverCandidate = null;
      renderSidebar();
      renderScene();
    });
    gridMode.addEventListener("click", () => {
      placementMode = "grid";
      hoverCandidate = null;
      renderSidebar();
      renderScene();
    });
    supportToggle.addEventListener("click", () => {
      showSupport = !showSupport;
      renderSidebar();
      renderScene();
    });
    applySize.addEventListener("click", () => {
      hoverCandidate = null;
      resetCameraOnNextRender = true;
      request("/container-size", {
        dx: Number(containerDx.value),
        dy: Number(containerDy.value),
        dz: Number(containerDz.value),
      });
    });
    sameItemHeight.addEventListener("change", () => {
      hoverCandidate = null;
      request("/same-item-height", { enabled: sameItemHeight.checked });
    });
    resetBtn.addEventListener("click", () => {
      resetCameraOnNextRender = true;
      request("/reset", {});
    });
    request("/state");
