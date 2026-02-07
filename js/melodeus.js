const $ = (id) => document.getElementById(id);
let ui;

const state = { devices: { inputs: [], outputs: [] }, selected: { inputs: [], outputs: [] }, ui: { showWaveforms: true } };
const contextRows = new Map();
const editContext = async (uuid) => {
  const row = contextRows.get(uuid);
  if (!uuid || !row) return;
  const draft = prompt("Edit message", row.dataset.message || "");
  if (draft === null) return;
  const author = row.dataset.author || "";
  try {
    await fetchJson(`/api/context/${encodeURIComponent(uuid)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ author, message: draft }),
    });
  } catch (e) {
    if (ui?.status) ui.status.textContent = e.message || "Failed to edit message";
  }
};
const deleteContext = async (uuid) => {
  if (!uuid) return;
  try {
    await fetchJson(`/api/context/${encodeURIComponent(uuid)}`, { method: "DELETE" });
  } catch (e) {
    if (ui?.status) ui.status.textContent = e.message || "Failed to delete message";
  }
};
const renderContextUpdate = (update) => {
  if (!update || update.type !== "context") return;
  const log = $("transcript-log");
  const uuid = update.uuid;
  if (!log || !uuid) return;
  const action = String(update.action || "").toLowerCase();
  if (action.includes("delete")) {
    const row = contextRows.get(uuid);
    if (row) row.remove();
    contextRows.delete(uuid);
    return;
  }
  let row = contextRows.get(uuid);
  if (!row) {
    row = document.createElement("div");
    row.className = "pill";
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "8px";
    row.dataset.uuid = uuid;
    log.append(row);
    contextRows.set(uuid, row);
  }
  row.dataset.uuid = uuid;
  row.dataset.author = update.author || "";
  row.dataset.message = update.message || "";
  row.replaceChildren();
  const author = update.author ? `${update.author}: ` : "";
  const text = document.createElement("span");
  text.textContent = `${author}${update.message || ""}`;
  const editBtn = document.createElement("button");
  editBtn.type = "button";
  editBtn.textContent = "Edit";
  editBtn.style.width = "32px";
  editBtn.style.height = "28px";
  editBtn.style.padding = "0";
  editBtn.style.marginLeft = "auto";
  editBtn.style.background = "#fff";
  editBtn.style.color = "var(--muted)";
  editBtn.style.border = "1px solid var(--border)";
  editBtn.style.borderRadius = "6px";
  editBtn.style.cursor = "pointer";
  editBtn.onclick = () => editContext(uuid);
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "×";
  btn.style.width = "28px";
  btn.style.height = "28px";
  btn.style.padding = "0";
  btn.style.marginLeft = "6px";
  btn.style.background = "transparent";
  btn.style.color = "var(--muted)";
  btn.style.border = "none";
  btn.style.cursor = "pointer";
  btn.onclick = () => deleteContext(uuid);
  row.append(text, editBtn, btn);
};
window.handleContextUpdate = renderContextUpdate;

const hostName = (cfg) => cfg.host_id || cfg.host_name || cfg.host || "unknown";
const fill = (sel, vals, fmt) => {
  const cur = sel.value;
  sel.innerHTML = "";
  vals.forEach((v) => sel.append(new Option(fmt ? fmt(v) : v, v)));
  sel.disabled = !vals.length;
  if (vals.includes(cur)) sel.value = cur;
};

const renderSelectors = (kind) => {
  const s = ui[kind];
  const configs = state.devices[s.key].flat();
  const hosts = [...new Set(configs.map(hostName))];
  fill(s.host, hosts);
  if (!hosts.length) return ["device", "channels", "rate", "format", "frame"].forEach((k) => s[k]?.replaceChildren());

  const host = s.host.value || hosts[0];
  s.host.value = host;
  const hostConfigs = configs.filter((c) => hostName(c) === host);
  const devices = [...new Set(hostConfigs.map((c) => c.device_name))];
  fill(s.device, devices, (d) => `${host} / ${d}`);
  const device = s.device.value || devices[0];
  s.device.value = device;
  const deviceConfigs = hostConfigs.filter((c) => c.device_name === device);
  fill(s.channels, [...new Set(deviceConfigs.map((c) => c.channels))]);
  fill(s.rate, [...new Set(deviceConfigs.map((c) => c.sample_rate))]);
  fill(s.format, [...new Set(deviceConfigs.map((c) => c.sample_format))]);
  if (s.frame) fill(s.frame, [...new Set(deviceConfigs.map((c) => c.frame_size))]);
};

const renderActive = (kind) => {
  const container = ui[kind].active;
  container.innerHTML = "";
  state.selected[ui[kind].key].forEach((cfg) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<div class="pill">${hostName(cfg)} / ${cfg.device_name} ${cfg.channels}ch @ ${cfg.sample_rate}Hz (${cfg.sample_format})</div>`;
    const btn = document.createElement("button");
    btn.textContent = "−";
    btn.style.width = "42px";
    btn.onclick = () => mutate("DELETE", kind, cfg);
    row.append(btn);
    container.append(row);
  });
};

const fetchJson = async (url, options) => {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

const load = async () => {
  ui.status.textContent = "Loading...";
  try {
    const [devices, selected] = await Promise.all([fetchJson("/api/devices"), fetchJson("/api/selected")]);
    state.devices = devices;
    state.selected = selected;
    renderSelectors("input");
    renderSelectors("output");
    renderActive("input");
    renderActive("output");
    ui.status.textContent = "Ready";
  } catch (e) {
    ui.status.textContent = e.message || "Failed to load";
  }
};

const pickConfig = (kind) => {
  const s = ui[kind];
  return state.devices[s.key].flat().find(
    (c) =>
      hostName(c) === s.host.value &&
      c.device_name === s.device.value &&
      c.channels === Number(s.channels.value) &&
      c.sample_rate === Number(s.rate.value) &&
      c.sample_format === s.format.value &&
      (!s.frame || c.frame_size === Number(s.frame.value)),
  );
};

const mutate = async (method, kind, cfg) => {
  if (!cfg) return (ui.status.textContent = "Select a device first");
  ui.status.textContent = method === "POST" ? "Adding..." : "Removing...";
  await fetchJson(method === "POST" ? "/api/select" : "/api/devices", {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: kind, config: cfg }),
  });
  await load();
};

const updateWaveToggle = () => {
  if (!ui.toggleWaves) return;
  ui.toggleWaves.textContent = state.ui.showWaveforms ? "Waveforms: On" : "Waveforms: Off";
  if (ui.waves) ui.waves.style.display = state.ui.showWaveforms ? "flex" : "none";
};

const loadUiConfig = async () => {
  try {
    const cfg = await fetchJson("/api/uiconfig");
    state.ui.showWaveforms = !!cfg.show_waveforms;
    updateWaveToggle();
  } catch (_) {
    /* ignore */
  }
};

const calibrate = async () => {
  ui.status.textContent = "Calibrating...";
  try {
    await fetchJson("/api/calibrate", { method: "POST" });
    ui.status.textContent = "Calibration done";
  } catch (e) {
    ui.status.textContent = e.message || "Failed to calibrate";
  }
};

const interrupt = async () => {
  ui.status.textContent = "Interrupting...";
  try {
    const res = await fetch("/api/interrupt", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    ui.status.textContent = "Interrupted";
  } catch (e) {
    ui.status.textContent = e.message || "Failed to interrupt";
  }
};

const playTestAudio = async () => {
  ui.status.textContent = "Playing test audio...";
  try {
    const res = await fetchJson("/play", { method: "POST" });
    const duration = typeof res.duration_sec === "number" ? `${res.duration_sec.toFixed(2)}s` : "test audio";
    const outputs = typeof res.outputs_started === "number" ? res.outputs_started : 0;
    ui.status.textContent = `Played ${duration} on ${outputs} output${outputs === 1 ? "" : "s"}`;
  } catch (e) {
    ui.status.textContent = e.message || "Failed to play test audio";
  }
};

document.addEventListener("DOMContentLoaded", () => {
  ui = {
    input: { key: "inputs", host: $("input-host"), device: $("input-device"), channels: $("input-channels"), rate: $("input-rate"), format: $("input-format"), frame: null, add: $("use-input"), active: $("input-active") },
    output: { key: "outputs", host: $("output-host"), device: $("output-device"), channels: $("output-channels"), rate: $("output-rate"), format: $("output-format"), frame: $("output-frame"), add: $("use-output"), active: $("output-active") },
    status: $("status"),
    refresh: $("refresh"),
    toggleWaves: $("toggle-waves-btn"),
    waves: $("waves"),
    calibrate: $("calibrate-btn"),
    testAudio: $("test-audio-btn"),
    interrupt: $("interrupt-btn"),
  };
  ui.refresh.onclick = load;
  ui.input.host.onchange = () => renderSelectors("input");
  ui.input.device.onchange = () => renderSelectors("input");
  ui.output.host.onchange = () => renderSelectors("output");
  ui.output.device.onchange = () => renderSelectors("output");
  ui.input.add.onclick = () => mutate("POST", "input", pickConfig("input"));
  ui.output.add.onclick = () => mutate("POST", "output", pickConfig("output"));
  if (ui.calibrate) ui.calibrate.onclick = calibrate;
  if (ui.testAudio) ui.testAudio.onclick = playTestAudio;
  if (ui.interrupt) ui.interrupt.onclick = interrupt;
  if (ui.toggleWaves) {
    ui.toggleWaves.onclick = async () => {
      const next = !state.ui.showWaveforms;
      try {
        await fetchJson("/api/uiconfig/waveforms", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ show_waveforms: next }),
        });
        state.ui.showWaveforms = next;
        updateWaveToggle();
      } catch (e) {
        ui.status.textContent = e.message || "Failed to toggle waveforms";
      }
    };
  }
  loadUiConfig();
  load();
});
