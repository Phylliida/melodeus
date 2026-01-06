(() => {
  const available = { input: [], output: [] };
  const fields = {
    input: [
      "host_id",
      "device_name",
      "channels",
      "sample_rate",
      "sample_format",
    ],
    output: [
      "host_id",
      "device_name",
      "channels",
      "sample_rate",
      "sample_format",
      "frame_size",
    ],
  };

  const selects = {};
  const idMap = {
    host_id: "host",
    device_name: "device",
    sample_rate: "rate",
    sample_format: "format",
    frame_size: "frame",
  };
  const activeLists = {
    input: document.getElementById("input-active"),
    output: document.getElementById("output-active"),
  };
  const statusEl = document.getElementById("status");
  const refreshBtn = document.getElementById("refresh");

  ["input", "output"].forEach((type) => {
    fields[type].forEach((field) => {
      selects[`${type}-${field}`] = document.getElementById(
        `${type}-${idFor(field)}`
      );
    });
  });

  function idFor(field) {
    return idMap[field] || field.replace(/_/g, "-");
  }

  function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.style.color = isError ? "#b91c1c" : "var(--muted)";
  }

  function unique(arr) {
    return Array.from(new Set(arr));
  }

  function setOptions(select, values) {
    if (!select) return;
    const prev = select.value;
    const stringValues = values.map((v) => String(v));
    select.innerHTML = "";
    if (!values.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "None";
      select.appendChild(opt);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    values.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = String(v);
      opt.textContent = String(v);
      select.appendChild(opt);
    });
    select.value = stringValues.includes(prev) ? prev : stringValues[0];
  }

  function pickPool(type) {
    const items = available[type];
    const hostSel = selects[`${type}-host_id`];
    const devSel = selects[`${type}-device_name`];
    const host = hostSel?.value;
    const device = devSel?.value;
    let pool = items;
    if (host) pool = pool.filter((i) => String(i.host_id) === host);
    if (device) pool = pool.filter((i) => String(i.device_name) === device);
    if (!pool.length) pool = items;
    return pool;
  }

  function refreshSelectors(type) {
    const list = available[type];
    if (!list.length) {
      fields[type].forEach((field) => setOptions(selects[`${type}-${field}`], []));
      return;
    }

    // host options are global
    setOptions(
      selects[`${type}-host_id`],
      unique(list.map((i) => i.host_id || ""))
    );
    // device depends on host
    const host = selects[`${type}-host_id`].value;
    const devicePool = list.filter((i) => String(i.host_id) === host);
    setOptions(
      selects[`${type}-device_name`],
      unique(devicePool.map((i) => i.device_name))
    );

    const pool = pickPool(type);
    const optsFor = (field) => unique(pool.map((i) => i[field]));

    fields[type]
      .filter((f) => f !== "host_id" && f !== "device_name")
      .forEach((field) => {
        setOptions(selects[`${type}-${field}`], optsFor(field));
      });
  }

  function bestConfig(type) {
    const pool = pickPool(type);
    if (pool.length) return pool[0];
    return available[type][0] || null;
  }

  function renderActive(type, list) {
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
      pill.textContent = `${cfg.host_id || "unknown"} / ${cfg.device_name} (${cfg.channels}ch @ ${cfg.sample_rate}Hz)`;
      container.appendChild(pill);
    });
  }

  async function loadDevices() {
    setStatus("Loading...");
    const res = await fetch("/api/devices");
    if (!res.ok) throw new Error(`Failed to load devices (${res.status})`);
    const data = await res.json();
    available.input = data.inputs || [];
    available.output = data.outputs || [];
    refreshSelectors("input");
    refreshSelectors("output");
    setStatus("Devices updated");
  }

  async function loadSelected() {
    const res = await fetch("/api/selected");
    if (!res.ok) throw new Error(`Failed to load active devices (${res.status})`);
    const data = await res.json();
    renderActive("input", data.inputs || []);
    renderActive("output", data.outputs || []);
  }

  function currentConfig(type) {
    const base = bestConfig(type);
    if (!base) return {};
    const cfg = { ...base };
    fields[type].forEach((field) => {
      const val = selects[`${type}-${field}`]?.value;
      if (val === undefined) return;
      const asInt = Number(val);
      cfg[field] = Number.isNaN(asInt) ? val : asInt;
    });
    return cfg;
  }

  async function addDevice(type) {
    const cfg = currentConfig(type);
    if (!available[type].length) {
      setStatus(`No ${type} devices available`, true);
      return;
    }
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
  }

  async function init() {
    try {
      await Promise.all([loadDevices(), loadSelected()]);
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Something went wrong", true);
    }
  }

  refreshBtn.addEventListener("click", init);
  document.getElementById("use-input").addEventListener("click", () =>
    addDevice("input").catch((err) => setStatus(err.message, true))
  );
  document.getElementById("use-output").addEventListener("click", () =>
    addDevice("output").catch((err) => setStatus(err.message, true))
  );

  ["input", "output"].forEach((type) => {
    ["host_id", "device_name"].forEach((field) => {
      const sel = selects[`${type}-${field}`];
      if (sel) {
        sel.addEventListener("change", () => refreshSelectors(type));
      }
    });
  });

  init();
})();
