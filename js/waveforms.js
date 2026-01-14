

const set_ws_status_color = (c) => (document.getElementById("ws").style.background = c)

const wsPort = 5001;
const url = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${wsPort}`;

const SAMPLE_RATE = 16000;
const MAX_SAMPLES = SAMPLE_RATE * 5;

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

const pushSamples = (key, channels, data) => {
  if (!channels || !data.length) return;
  ensureCanvases(key, channels);
  if (!waveBuffers[key].length || waveBuffers[key].length !== channels) {
    waveBuffers[key] = Array.from({ length: channels }, () => []);
  }
  const perChannel = splitChannels(data, channels);
  perChannel.forEach((samples, idx) => {
    const merged = waveBuffers[key][idx].concat(samples);
    if (merged.length > MAX_SAMPLES) merged.splice(0, merged.length - MAX_SAMPLES);
    waveBuffers[key][idx] = merged;
  });
};

const drawWave = (key) => {
  const container = waveContainers[key];
  const buffers = waveBuffers[key];
  buffers.forEach((samples, idx) => {
    const canvas = container.children[idx];
    const width = (canvas.width = canvas.clientWidth || 600);
    const height = (canvas.height = canvas.height || 60);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    if (!samples.length) return;
    const step = Math.max(1, Math.floor(samples.length / width));
    const mid = height / 2;
    ctx.strokeStyle = "#2563eb";
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const start = x * step;
      let min = 1,
        max = -1;
      for (let i = start; i < Math.min(start + step, samples.length); i++) {
        const s = samples[i];
        if (s < min) min = s;
        if (s > max) max = s;
      }
      ctx.moveTo(x, mid + min * mid);
      ctx.lineTo(x, mid + max * mid);
    }
    ctx.stroke();
  });
};

const renderWaveforms = (payload) => {
  wavesSection.style.display = "flex";
  pushSamples("output", payload.out_ch, decodeF32(payload.output));
  pushSamples("input", payload.in_ch, decodeF32(payload.input));
  pushSamples("aec", payload.in_ch, decodeF32(payload.aec));
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
    } catch (e) {
      console.warn("Bad ws payload", e);
    }
    set_ws_status_color("#3f3");
  }
  const stop = () => {
    set_ws_status_color("#f33");
    setTimeout(connect, 1000);
  }
  ws.onclose = stop;
  ws.onerror = stop;
};

document.addEventListener("DOMContentLoaded", () => {
    waveContainers = { output: $("wave-out"), input: $("wave-in"), aec: $("wave-aec") };
    wavesSection = $("waves");
    set_ws_status_color("#ff0");
    connect();
});
