const el = (id) => document.getElementById(id)
const setOpts = (sel, items) => {
  sel.innerHTML = ""
  items.forEach(([v, l]) => sel.add(new Option(l, v)))
}

const label = (d) => `${d.host}/${d.device}`
let devices = { inputs: [], outputs: [] }
let added = []
let last = {}
const themeColor = (name, fallback) => {
  const val = getComputedStyle(document.documentElement).getPropertyValue(name)
  return (val && val.trim()) || fallback
}

const refresh = (prefix, key) => {
  const dev = devices[key][Number(el(`${prefix}-dev`).value) || 0]
  setOpts(el(`${prefix}-ch`), (dev?.channels || []).map((v) => [v, v]))
  setOpts(el(`${prefix}-rate`), (dev?.sample_rates || []).map((v) => [v, v]))
  setOpts(el(`${prefix}-fmt`), (dev?.sample_formats || []).map((v) => [v, v]))
}

const populate = (data) => {
  devices = data
  const inOpts = data.inputs.map((d, i) => [i, label(d)])
  const outOpts = data.outputs.map((d, i) => [i, label(d)])
  setOpts(el("in-dev"), inOpts)
  setOpts(el("out-dev"), outOpts)
  const pick = (kind, opts) => {
    const saved = last[kind]
    if (!saved) return 0
    const idx = opts.findIndex(([, lbl]) => lbl === `${saved.host}/${saved.device}`)
    return idx >= 0 ? idx : 0
  }
  el("in-dev").value = pick("input", inOpts)
  el("out-dev").value = pick("output", outOpts)
  refresh("in", "inputs")
  refresh("out", "outputs")
  const setSel = (prefix, key) => {
    const saved = last[key]
    if (!saved) return
    el(`${prefix}-ch`).value = saved.channels
    el(`${prefix}-rate`).value = saved.sample_rate
    el(`${prefix}-fmt`).value = saved.sample_format
  }
  setSel("in", "input")
  setSel("out", "output")
  return data
}

const fetchDevices = () => fetch("/devices").then((r) => r.json()).then(populate)
const fetchLast = () => fetch("/last").then((r) => r.json()).then((d) => (last = d || {}))

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
el("cal").onclick = () => {
  showOverlay(true)
  fetch("/calibrate", { method: "POST" })
    .catch(console.error)
    .finally(() => showOverlay(false))
}
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
const capTyped = (prev, next, rate, Ctor) => {
  const max = Math.max(1, Math.floor(rate * 2))
  const len = Math.min(max, prev.length + next.length)
  const out = new Ctor(len)
  const keep = Math.max(0, len - next.length)
  if (keep) out.set(prev.slice(prev.length - keep))
  const copyLen = Math.min(next.length, len - keep)
  out.set(next.slice(next.length - copyLen), keep)
  return out
}
const capAll = (prev, next, rate, Ctor) => next.map((n, i) => capTyped(prev[i] || new Ctor(), n, rate, Ctor))
const draw = (c, data, mask) => {
  if (!c) return
  const ctx = c.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, c.width, c.height)
  if (!data.length) return
  const mid = c.height / 2
  const step = Math.max(1, Math.floor(data.length / c.width))
  const idle = themeColor("--wave-accent", "#7dd3fc")
  const speech = themeColor("--accent-2", "#a6f3d6")
  let lastState = mask?.length ? Boolean(mask[0]) : false
  let started = false
  ctx.lineWidth = 1.5
  ctx.lineJoin = "round"
  ctx.strokeStyle = lastState ? speech : idle
  ctx.beginPath()
  for (let x = 0, i = 0; x < c.width && i < data.length; x++, i += step) {
    const idx = Math.min(mask?.length ? mask.length - 1 : 0, Math.floor(i))
    const state = mask?.length ? Boolean(mask[idx]) : false
    if (state !== lastState && started) {
      ctx.stroke()
      ctx.beginPath()
      ctx.strokeStyle = state ? speech : idle
      started = false
    }
    const y = mid - data[i] * (mid * 0.9)
    if (!started) {
      ctx.moveTo(x, y)
      started = true
    } else {
      ctx.lineTo(x, y)
    }
    lastState = state
  }
  if (started) ctx.stroke()
}
const render = (boxId, arrs, masks = []) => {
  const box = el(boxId)
  if (!box) return
  while (box.children.length > arrs.length) box.lastChild.remove()
  while (box.children.length < arrs.length) {
    const c = document.createElement("canvas")
    c.width = 640
    c.height = 40
    c.style.width = "100%"
    c.style.height = "48px"
    box.appendChild(c)
  }
  arrs.forEach((a, i) => draw(box.children[i], a, masks[i]))
}
const waves = { input: [], output: [], aec: [] }
const masks = { input: [], output: [], aec: [] }
const onDebug = (d) => {
  const rate = Number(d.rate) || 16000
  const outChunks = splitf(new Float32Array(buf(d.output)), d.out_ch || 1)
  const inChunks = splitf(new Float32Array(buf(d.input)), d.in_ch || 1)
  const aecChunks = splitf(new Float32Array(buf(d.aec)), d.in_ch || 1)
  waves.output = capAll(waves.output, outChunks, rate, Float32Array)
  waves.input = capAll(waves.input, inChunks, rate, Float32Array)
  waves.aec = capAll(waves.aec, aecChunks, rate, Float32Array)
  const vadFlags = Array.isArray(d.vad) ? d.vad : []
  const aecMasks = aecChunks.map((chunk, idx) => {
    const m = new Uint8Array(chunk.length)
    if (vadFlags[idx]) m.fill(1)
    return m
  })
  masks.aec = capAll(masks.aec, aecMasks, rate, Uint8Array)
  el("waves").style.display = "block"
  render("wave-out", waves.output)
  render("wave-in", waves.input)
  render("wave-aec", waves.aec, masks.aec)
}
const setGain = (g) =>
  fetch("/gain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gain: g }),
  }).catch(console.error)
const clampGain = (v) => {
  const n = Number(v)
  if (!Number.isFinite(n)) return 1
  return Math.max(0, Math.min(50, n))
}
const applyGain = (val) => {
  const g = clampGain(val)
  el("gain").value = g
  const text = el("gain-text")
  if (text) text.value = g
  setGain(g)
}
el("gain").oninput = (e) => applyGain(e.target.value)
const gainText = el("gain-text")
if (gainText) {
  gainText.onchange = (e) => applyGain(e.target.value)
  gainText.onkeyup = (e) => {
    if (e.key === "Enter") applyGain(e.target.value)
  }
}
const overlay = el("overlay")
const showOverlay = (on) => {
  if (!overlay) return
  overlay.style.display = on ? "flex" : "none"
}
Promise.all([fetchLast(), fetchDevices()])
  .then(() => applyGain(el("gain").value))
  .catch(console.error)
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
