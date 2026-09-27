import { FrameRenderer } from "./frame-renderer.js";

export class ReplayClock {
  constructor(
    schedule = (callback, interval) => globalThis.setInterval(callback, interval),
    cancel = (timer) => globalThis.clearInterval(timer),
  ) {
    this.schedule = schedule;
    this.cancel = cancel;
    this.timer = null;
  }

  start(callback, interval) {
    if (this.timer !== null) return false;
    this.timer = this.schedule(callback, interval);
    return true;
  }

  pause() {
    if (this.timer === null) return false;
    const timer = this.timer;
    this.timer = null;
    this.cancel(timer);
    return true;
  }
}

export function startReplay(payload, options = {}) {
  const root = options.root || document;
  const renderer = options.renderer || new FrameRenderer(root);
  const clock = options.clock || new ReplayClock();
  const frames = Array.isArray(payload) ? payload : payload?.frames || [];
  const slider = root.querySelector("#replay-slider");
  const label = root.querySelector("#replay-position");
  const play = root.querySelector("#replay-play");
  let index = 0;
  let stopped = false;
  const show = (next) => {
    if (!frames.length) return;
    index = Math.max(0, Math.min(frames.length - 1, next));
    const record = frames[index];
    const frame = record?.frame ?? record;
    renderer.apply(record?.title ? { ...frame, title: record.title } : frame);
    if (slider) slider.value = String(index);
    if (label) label.textContent = `${index + 1} / ${frames.length}`;
  };
  const listeners = [];
  const listen = (element, type, handler) => {
    if (!element) return;
    element.addEventListener(type, handler);
    listeners.push(() => element.removeEventListener(type, handler));
  };
  if (slider) slider.max = String(Math.max(0, frames.length - 1));
  listen(slider, "input", () => show(Number(slider.value)));
  listen(root.querySelector("#replay-prev"), "click", () => show(index - 1));
  listen(root.querySelector("#replay-next"), "click", () => show(index + 1));
  listen(play, "click", () => {
    if (stopped) return;
    if (clock.pause()) { play.textContent = "Play"; return; }
    play.textContent = "Pause";
    clock.start(() => show(index >= frames.length - 1 ? 0 : index + 1), Number(options.interval ?? payload?.interval_ms ?? 500));
  });
  if (frames.length) show(0); else if (label) label.textContent = "No frames";
  return {
    show,
    pause() { const paused = clock.pause(); if (play) play.textContent = "Play"; return paused; },
    stop() {
      if (stopped) return;
      stopped = true;
      clock.pause();
      listeners.splice(0).forEach((remove) => remove());
      renderer.dispose();
    },
  };
}
