(() => {
  const available = { input: [], output: [] };
  const selects = {
    input: document.getElementById("input-select"),
    output: document.getElementById("output-select"),
  };
  const activeLists = {
    input: document.getElementById("input-active"),
    output: document.getElementById("output-active"),
  };
  const statusEl = document.getElementById("status");
  const refreshBtn = document.getElementById("refresh");
  const addButtons = {
    input: document.getElementById("use-input"),
    output: document.getElementById("use-output"),
  };

  const fmt = (cfg) =>
    `${cfg.host_id || "unknown"} / ${cfg.device_name} - ${cfg.channels}ch @ ${cfg.sample_rate}Hz (${cfg.sample_format})`;

  const setStatus = (msg, isError = false) => {
    statusEl.textContent = msg;
    statusEl.style.color = isError ? "#fca5a5" : "var(--muted)";
  };

  const renderOptions = (type) => {
    const sel = selects[type];
    sel.innerHTML = "";
    const list = available[type];
    if (!list.length) {
      const opt = document.createElement("option");
      opt.textContent = "No devices found";
      opt.value = "";
      sel.appendChild(opt);
      sel.disabled = true;
      return;
    }
    sel.disabled = false;
    list.forEach((cfg, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = fmt(cfg);
      sel.appendChild(opt);
    });
  };

  const renderActive = (type, list) => {
    const container = activeLists[type];
    container.innerHTML = "";
    if (!list.length) {
      const pill = document.createElement("div");
      pill.className = "pill";
      pill.textContent = "None selected";
      container.appendChild(pill);
      return;
    }
    list.forEach((cfg) => {
      const pill = document.createElement("div");
      pill.className = "pill";
      pill.innerHTML = `<strong>${cfg.device_name}</strong><span style="color: var(--muted);">${cfg.host_id} - ${cfg.channels}ch @ ${cfg.sample_rate}Hz</span>`;
      container.appendChild(pill);
    });
  };

  const loadDevices = async () => {
    setStatus("Loading devices...");
    const res = await fetch("/api/devices");
    if (!res.ok) throw new Error(`Failed to load devices (${res.status})`);
    const data = await res.json();
    available.input = data.inputs || [];
    available.output = data.outputs || [];
    renderOptions("input");
    renderOptions("output");
    setStatus("Devices updated");
  };

  const loadSelected = async () => {
    const res = await fetch("/api/selected");
    if (!res.ok) throw new Error(`Failed to load active devices (${res.status})`);
    const data = await res.json();
    renderActive("input", data.inputs || []);
    renderActive("output", data.outputs || []);
  };

  const addDevice = async (type) => {
    const sel = selects[type];
    const list = available[type];
    if (sel.disabled || !list.length) {
      setStatus(`No ${type} devices available`, true);
      return;
    }
    const cfg = list[Number(sel.value) || 0];
    const res = await fetch("/api/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, config: cfg }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Failed to add ${type} device`);
    }
    setStatus(`Added ${type} device: ${cfg.device_name}`);
    await loadSelected();
  };

  const init = async () => {
    try {
      await Promise.all([loadDevices(), loadSelected()]);
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Something went wrong", true);
    }
  };

  refreshBtn.addEventListener("click", init);
  addButtons.input.addEventListener("click", () => addDevice("input").catch((err) => setStatus(err.message, true)));
  addButtons.output.addEventListener("click", () => addDevice("output").catch((err) => setStatus(err.message, true)));

  init();
})();
