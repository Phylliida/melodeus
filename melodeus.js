const el = (id) => document.getElementById(id)
const payload = () => ({
  target_sample_rate: el("rate").value,
  frame_size: el("frame").value,
  filter_length: el("filter").value,
})
const init = () =>
  fetch("/aec", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload()),
  }).then(async (r) => (r.ok ? r.json() : Promise.reject(await r.json())))

const setOpts = (sel, items) => {
  sel.innerHTML = ""
  items.forEach(([v, l]) => sel.add(new Option(l, v)))
}

const label = (d) => `${d.host}/${d.device}`
let devices = { inputs: [], outputs: [] }
let added = []

const refresh = (prefix, key) => {
  const dev = devices[key][Number(el(`${prefix}-dev`).value) || 0]
  setOpts(el(`${prefix}-ch`), (dev?.channels || []).map((v) => [v, v]))
  setOpts(el(`${prefix}-rate`), (dev?.sample_rates || []).map((v) => [v, v]))
  setOpts(el(`${prefix}-fmt`), (dev?.sample_formats || []).map((v) => [v, v]))
}

const populate = (data) => {
  devices = data
  setOpts(
    el("in-dev"),
    data.inputs.map((d, i) => [i, label(d)])
  )
  setOpts(
    el("out-dev"),
    data.outputs.map((d, i) => [i, label(d)])
  )
  refresh("in", "inputs")
  refresh("out", "outputs")
  return data
}

const fetchDevices = () => fetch("/devices").then((r) => r.json()).then(populate)

el("in-dev").onchange = () => refresh("in", "inputs")
el("out-dev").onchange = () => refresh("out", "outputs")

const listAdded = () => {
  const box = el("added")
  box.innerHTML = ""
  added.forEach((d, i) => {
    const row = document.createElement("div")
    row.textContent = `${d.kind}: ${d.host}/${d.device} ${d.channels}ch @ ${d.sample_rate}Hz ${d.sample_format}`
    const btn = document.createElement("button")
    btn.textContent = "-"
    btn.onclick = () => removeDevice(i)
    row.appendChild(btn)
    box.appendChild(row)
  })
}

const selection = (prefix, key) => {
  const dev = devices[key][Number(el(`${prefix}-dev`).value) || 0]
  if (!dev) throw new Error("device missing")
  return {
    kind: key === "inputs" ? "input" : "output",
    host: dev.host,
    device: dev.device,
    channels: Number(el(`${prefix}-ch`).value),
    sample_rate: Number(el(`${prefix}-rate`).value),
    sample_format: el(`${prefix}-fmt`).value,
  }
}

const addDevice = (prefix, key) =>
  fetch("/device", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection(prefix, key)),
  })
    .then((r) => (r.ok ? r.json() : Promise.reject(r)))
    .then((resp) => {
      added.push(resp.added)
      listAdded()
    })

const removeDevice = (idx) => {
  const entry = added[idx]
  added = added.filter((_, i) => i !== idx)
  listAdded()
  if (!entry) return
  fetch("/device", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  }).catch(console.error)
}

el("init").onclick = () =>
  init()
    .then(fetchDevices)
    .then(() => (el("devices").style.display = "block"))
    .catch((e) => console.error(e))

el("in-add").onclick = () => addDevice("in", "inputs").catch(console.error)
el("out-add").onclick = () => addDevice("out", "outputs").catch(console.error)
