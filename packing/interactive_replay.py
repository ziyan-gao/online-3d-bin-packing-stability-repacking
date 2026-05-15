import json
import os


REPLAY_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pack Replay</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #f5f6f8; font-family: sans-serif; }
    #root { width: 100%; height: calc(100% - 64px); }
    #controls {
      height: 64px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 14px;
      box-sizing: border-box;
      background: #fff;
      border-top: 1px solid #ddd;
    }
    button { min-width: 64px; height: 34px; }
    input[type="range"] { flex: 1; }
    #label { min-width: 260px; font-size: 14px; color: #222; }
  </style>
</head>
<body>
  <div id="root"></div>
  <div id="controls">
    <button id="prev">Prev</button>
    <button id="play">Play</button>
    <button id="next">Next</button>
    <input id="slider" type="range" min="0" max="0" value="0" />
    <span id="label"></span>
  </div>
  <script>
    const frames = __FRAMES__;
    const intervalMs = __INTERVAL_MS__;
    let index = 0;
    let timer = null;

    const root = document.getElementById('root');
    const slider = document.getElementById('slider');
    const label = document.getElementById('label');
    const play = document.getElementById('play');

    slider.max = Math.max(frames.length - 1, 0);

    function currentCamera() {
      const layout = root.layout || {};
      return {
        scene: layout.scene && layout.scene.camera,
        scene2: layout.scene2 && layout.scene2.camera,
        scene3: layout.scene3 && layout.scene3.camera,
      };
    }

    async function render(nextIndex) {
      if (!frames.length) return;
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      const frame = frames[index];
      const layout = JSON.parse(JSON.stringify(frame.figure.layout || {}));
      const camera = currentCamera();
      if (camera.scene && layout.scene) layout.scene.camera = camera.scene;
      if (camera.scene2 && layout.scene2) layout.scene2.camera = camera.scene2;
      if (camera.scene3 && layout.scene3) layout.scene3.camera = camera.scene3;
      layout.uirevision = 'pack-replay';
      await Plotly.react(root, frame.figure.data || [], layout, {responsive: true});
      slider.value = index;
      label.textContent = `${index + 1}/${frames.length}: ${frame.title}`;
    }

    function stop() {
      if (timer !== null) clearInterval(timer);
      timer = null;
      play.textContent = 'Play';
    }

    document.getElementById('prev').onclick = () => { stop(); render(index - 1); };
    document.getElementById('next').onclick = () => { stop(); render(index + 1); };
    slider.oninput = () => { stop(); render(Number(slider.value)); };
    play.onclick = () => {
      if (timer !== null) {
        stop();
        return;
      }
      play.textContent = 'Pause';
      timer = setInterval(() => {
        if (index >= frames.length - 1) {
          stop();
          return;
        }
        render(index + 1);
      }, intervalMs);
    };

    render(0);
  </script>
</body>
</html>
"""


class InteractiveReplayRecorder:
    def __init__(
        self,
        out_path: str,
        interval_ms: int = 700,
    ) -> None:
        self.out_path = out_path
        self.interval_ms = int(interval_ms)
        self.frames = []

    def capture(self, title: str, fig) -> None:
        self.frames.append({"title": title, "figure": fig.to_dict()})

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        html = REPLAY_TEMPLATE.replace("__FRAMES__", json.dumps(self.frames))
        html = html.replace("__INTERVAL_MS__", str(self.interval_ms))
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved interactive replay: {self.out_path} ({len(self.frames)} frames)")
