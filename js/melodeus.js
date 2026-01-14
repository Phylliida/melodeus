const $ = (id) => document.getElementById(id);
let ui;

const state = { devices: { inputs: [], outputs: [] }, selected: { inputs: [], outputs: [] } };
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

document.addEventListener("DOMContentLoaded", () => {
  ui = {
    input: { key: "inputs", host: $("input-host"), device: $("input-device"), channels: $("input-channels"), rate: $("input-rate"), format: $("input-format"), frame: null, add: $("use-input"), active: $("input-active") },
    output: { key: "outputs", host: $("output-host"), device: $("output-device"), channels: $("output-channels"), rate: $("output-rate"), format: $("output-format"), frame: $("output-frame"), add: $("use-output"), active: $("output-active") },
    status: $("status"),
    refresh: $("refresh"),
  };
  ui.refresh.onclick = load;
  ui.input.host.onchange = () => renderSelectors("input");
  ui.input.device.onchange = () => renderSelectors("input");
  ui.output.host.onchange = () => renderSelectors("output");
  ui.output.device.onchange = () => renderSelectors("output");
  ui.input.add.onclick = () => mutate("POST", "input", pickConfig("input"));
  ui.output.add.onclick = () => mutate("POST", "output", pickConfig("output"));
  load();
});
