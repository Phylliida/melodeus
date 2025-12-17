const el = (id) => document.getElementById(id)
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

fetchDevices().then(() => (el("devices").style.display = "block")).catch(console.error)

el("in-add").onclick = () => addDevice("in", "inputs").catch(console.error)
el("out-add").onclick = () => addDevice("out", "outputs").catch(console.error)
el("play").onclick = () =>
  fetch("/play", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gain: Number(el("play-gain").value) || 1 }),
  }).catch(console.error)
el("cal").onclick = () =>
  fetch("/calibrate", { method: "POST" }).catch(console.error)
const wsPort = 8134
const dot = () => el("ws")
const paint = (c) => (dot().style.background = c)
const url = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${wsPort}`
const buf = (s) => Uint8Array.from(atob(s || ""), (c) => c.charCodeAt(0)).buffer
const split16 = (a, ch) => {
  if (!ch) return []
  const f = Math.floor(a.length / ch)
  return Array.from({ length: ch }, (_, c) => {
    const out = new Float32Array(f)
    for (let i = 0; i < f; i++) out[i] = a[i * ch + c] / 32768
    return out
  })
}
const splitf = (a, ch) => {
  if (!ch) return []
  const f = Math.floor(a.length / ch)
  return Array.from({ length: ch }, (_, c) => {
    const out = new Float32Array(f)
    for (let i = 0; i < f; i++) out[i] = a[i * ch + c]
    return out
  })
}
const cap = (prev, next, rate) => {
  const max = Math.max(1, Math.floor(rate * 2))
  const len = Math.min(max, prev.length + next.length)
  const out = new Float32Array(len)
  const keep = Math.max(0, len - next.length)
  if (keep) out.set(prev.slice(prev.length - keep))
  out.set(next.slice(next.length - Math.min(next.length, len)), keep)
  return out
}
const capAll = (prev, next, rate) => next.map((n, i) => cap(prev[i] || new Float32Array(), n, rate))
const draw = (c, data) => {
  if (!c) return
  const ctx = c.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, c.width, c.height)
  if (!data.length) return
  const mid = c.height / 2
  const step = Math.max(1, Math.floor(data.length / c.width))
  ctx.beginPath()
  for (let x = 0, i = 0; x < c.width && i < data.length; x++, i += step) {
    const y = mid - data[i] * (mid * 0.9)
    x ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
  }
  ctx.stroke()
}
const render = (boxId, arrs) => {
  const box = el(boxId)
  if (!box) return
  while (box.children.length > arrs.length) box.lastChild.remove()
  while (box.children.length < arrs.length) {
    const c = document.createElement("canvas")
    c.width = 640
    c.height = 80
    box.appendChild(c)
  }
  arrs.forEach((a, i) => draw(box.children[i], a))
}
const waves = { input: [], output: [], aec: [] }
const onDebug = (d) => {
  const rate = Number(d.rate) || 16000
  waves.output = capAll(waves.output, splitf(new Float32Array(buf(d.output)), d.out_ch || 1), rate)
  waves.input = capAll(waves.input, splitf(new Float32Array(buf(d.input)), d.in_ch || 1), rate)
  waves.aec = capAll(waves.aec, splitf(new Float32Array(buf(d.aec)), d.in_ch || 1), rate)
  el("waves").style.display = "block"
  render("wave-out", waves.output)
  render("wave-in", waves.input)
  render("wave-aec", waves.aec)
}
const setGain = (g) =>
  fetch("/gain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gain: g }),
  }).catch(console.error)
el("gain").oninput = (e) => setGain(Number(e.target.value) || 1)
setGain(Number(el("gain").value) || 1)
const connect = () => {
  const ws = new WebSocket(url())
  ws.onopen = () => paint("#3f3")
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data)
      if (data.type === "debug") onDebug(data)
    } catch (e) {}
    paint("#3f3")
  }
  const stop = () => paint("#f33")
  ws.onclose = stop
  ws.onerror = stop
}
paint("#ff0")
connect()
