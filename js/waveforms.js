

const set_ws_status_color = (c) => (document.getElementById("ws").style.background = c)
let transcriptHandler = null;
window.registerTranscriptHandler = (fn) => {
  transcriptHandler = fn;
};

const wsPort = 5001;
const url = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${wsPort}`;

const SAMPLE_RATE = 16000;
const MAX_SAMPLES = SAMPLE_RATE * 5;
const ACTIVE_COLOR = "#15803d";   // deeper green for contrast
const DEFAULT_COLOR = "#2563eb";
const FILL_ALPHA = 0.2;

// already declared in melodeus.js
//const $ = (id) => document.getElementById(id);
// Each channel holds an array of segments: { data: number[], color: string }
const waveBuffers = { output: [], input: [], aec: [] };
let waveContainers;
let wavesSection;

const decodeF32 = (b64) => new Float32Array(Uint8Array.from(atob(b64 || ""), (c) => c.charCodeAt(0)).buffer);

const ensureCanvases = (key, channels) => {
  const container = waveContainers[key];
  if (!container || !channels) return;
  if (container.childElementCount === channels) return;
  container.innerHTML = "";
  for (let i = 0; i < channels; i++) {
    const canvas = document.createElement("canvas");
    canvas.height = 60;
    container.append(canvas);
  }
};

const splitChannels = (data, channels) => {
  const out = Array.from({ length: channels }, () => []);
  for (let i = 0; i < data.length; i += channels) {
    for (let c = 0; c < channels; c++) out[c].push(data[i + c] ?? 0);
  }
  return out;
};

const trimSegments = (segments) => {
  let total = segments.reduce((n, seg) => n + seg.data.length, 0);
  while (total > MAX_SAMPLES && segments.length) {
    const first = segments[0];
    const over = total - MAX_SAMPLES;
    if (first.data.length <= over) {
      total -= first.data.length;
      segments.shift();
    } else {
      segments[0] = { ...first, data: first.data.slice(over) };
      total = MAX_SAMPLES;
    }
  }
};

const pushSamples = (key, channels, data, vads = []) => {
  if (!channels || !data.length) return;
  ensureCanvases(key, channels);
  if (!waveBuffers[key].length || waveBuffers[key].length !== channels) {
    waveBuffers[key] = Array.from({ length: channels }, () => []);
  }
  const perChannel = splitChannels(data, channels);
  perChannel.forEach((samples, idx) => {
    const color = key === "output" ? DEFAULT_COLOR : (vads[idx] ? ACTIVE_COLOR : DEFAULT_COLOR);
    waveBuffers[key][idx].push({ data: samples, color });
    trimSegments(waveBuffers[key][idx]);
  });
};

const drawWave = (key) => {
  const container = waveContainers[key];
  const buffers = waveBuffers[key];
  buffers.forEach((segments, idx) => {
    const canvas = container.children[idx];
    const width = (canvas.width = canvas.clientWidth || 600);
    const height = (canvas.height = canvas.height || 60);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    const totalSamples = segments.reduce((n, seg) => n + seg.data.length, 0);
    if (!totalSamples) return;
    const step = Math.max(1, Math.floor(totalSamples / width));
    const mid = height / 2;

    let segIdxForColor = 0;
    let segStart = 0;
    for (let x = 0; x < width; x++) {
      const start = x * step;
      while (segIdxForColor < segments.length && start >= segStart + segments[segIdxForColor].data.length) {
        segStart += segments[segIdxForColor].data.length;
        segIdxForColor++;
      }
      const color = segments[segIdxForColor]?.color || DEFAULT_COLOR;

      let remaining = step;
      let curSegIdx = segIdxForColor;
      let curSegStart = segStart;
      let curOffset = start - curSegStart;
      let min = 1, max = -1;
      while (remaining > 0 && curSegIdx < segments.length) {
        const seg = segments[curSegIdx];
        const take = Math.min(remaining, seg.data.length - curOffset);
        for (let i = 0; i < take; i++) {
          const s = seg.data[curOffset + i];
          if (s < min) min = s;
          if (s > max) max = s;
        }
        remaining -= take;
        curSegIdx++;
        curOffset = 0;
      }
      ctx.save();
      ctx.globalAlpha = FILL_ALPHA;
      ctx.fillStyle = color;
      ctx.fillRect(x, 0, 1, height);
      ctx.restore();
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, mid + min * mid);
      ctx.lineTo(x, mid + max * mid);
      ctx.stroke();
    }
  });
};

const renderWaveforms = (payload) => {
  wavesSection.style.display = "flex";
  const vads = payload.vad || [];
  pushSamples("output", payload.out_ch, decodeF32(payload.output));
  pushSamples("input", payload.in_ch, decodeF32(payload.input), vads);
  pushSamples("aec", payload.in_ch, decodeF32(payload.aec), vads);
  drawWave("output");
  drawWave("input");
  drawWave("aec");
};

const connect = () => {
  const ws = new WebSocket(url());
  ws.onopen = () => set_ws_status_color("#3f3");
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === "waveform") renderWaveforms(data);
      else if (data.type === "stt" && transcriptHandler) transcriptHandler(data);
    } catch (e) {
      console.warn("Bad ws payload", e);
    }
    set_ws_status_color("#3f3");
  }
  // don't make both of these call set timeout or we get exponential growth
  ws.onclose = () => {
    set_ws_status_color("#f33");
    setTimeout(connect, 1000);
  };
  ws.onerror = () => {
    set_ws_status_color("#f33");
  };
};

document.addEventListener("DOMContentLoaded", () => {
    waveContainers = { output: $("wave-out"), input: $("wave-in"), aec: $("wave-aec") };
    wavesSection = $("waves");
    set_ws_status_color("#ff0");
    connect();
});
