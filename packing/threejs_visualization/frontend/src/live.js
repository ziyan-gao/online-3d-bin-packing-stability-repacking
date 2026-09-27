import { FrameRenderer } from "./frame-renderer.js";

export function startLive(options = {}) {
  const root = options.root || document;
  const renderer = options.renderer || new FrameRenderer(root);
  const interval = Number(options.interval ?? 250);
  let stopped = false;
  let session = null;
  let nextIndex = 0;
  let timer = null;
  let request = null;

  async function poll() {
    try {
      request = new AbortController();
      const url = new URL(options.url || "/frame.json", window.location.href);
      url.searchParams.set("index", String(nextIndex));
      url.searchParams.set("t", String(Date.now()));
      if (session !== null) url.searchParams.set("session", session);
      const response = await fetch(url, { cache: "no-store", signal: request.signal });
      if (response.status === 204) return;
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const payload = await response.json();
      if (payload.reset) {
        session = payload.session_id ?? payload.session ?? null;
        nextIndex = Number(payload.next_index ?? 0);
        return;
      }
      const index = Number(payload.index ?? nextIndex);
      if (!payload.frame || index !== nextIndex) return;
      session = payload.session_id ?? payload.session ?? session;
      renderer.apply(payload.frame);
      nextIndex += 1;
      const status = root.querySelector("#live-status");
      if (status) status.textContent = "Connected";
    } catch (error) {
      if (!stopped && error.name !== "AbortError") {
        const status = root.querySelector("#live-status");
        if (status) status.textContent = `Waiting: ${error.message}`;
      }
    } finally {
      request = null;
      if (!stopped) timer = setTimeout(poll, interval);
    }
  }
  poll();
  return {
    stop() {
      if (stopped) return;
      stopped = true;
      if (timer !== null) { clearTimeout(timer); timer = null; }
      request?.abort();
      renderer.dispose();
    },
  };
}
