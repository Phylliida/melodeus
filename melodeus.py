import asyncio
import errno
from os import getenv
from pathlib import Path
import threading
import base64, json
from flask import Flask, jsonify, request
import websockets
import melaec3

root = Path(__file__).parent
app = Flask(__name__, static_folder=str(root), static_url_path="")
stream = None
HISTORY_LEN = 100
CALIBRATION = 20
AUDIO_BUF = 20
RESAMPLE_Q = 5
DEFAULT_FRAME = 160
WS_PORT = int(getenv("MELODEUS_WS_PORT", "8134"))

connections = set()
outputs = {}

def enc(buf: bytes) -> str:
    return base64.b64encode(bytes(buf)).decode()


async def broadcast(msg: str):
    dead = []
    for ws in list(connections):
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.discard(ws)


async def pump_debug():
    while True:
        if stream:
            try:
                i, o, a, _, _ = await stream.update_debug()
                await asyncio.sleep(0.001) # yield so ws can go
                payload = {
                    "type": "debug",
                    "rate": stream.config.target_sample_rate,
                    "in_ch": stream.num_input_channels,
                    "out_ch": stream.num_output_channels,
                    "input": enc(i),
                    "output": enc(o),
                    "aec": enc(a),
                }
                await broadcast(json.dumps(payload))
            except Exception as e:
                print("debug err", e)
                await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.05)


async def websocketServer(websocket):
    connections.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connections.discard(websocket)

async def serve_websocket():
    await websockets.serve(websocketServer, "0.0.0.0", WS_PORT)
    asyncio.create_task(pump_debug())
    await asyncio.Event().wait()

def start_ws():
    ws_thread = threading.Thread(target=lambda: asyncio.run(serve_websocket()), daemon=True)
    ws_thread.start()

def read(name: str) -> str:
    return root.joinpath(name).read_text()


def build_stream(rate: int, frame: int, length: int) -> melaec3.AecStream:
    cfg = melaec3.AecConfig(target_sample_rate=rate, frame_size=frame, filter_length=length)
    return melaec3.AecStream(cfg)


def grab_int(payload: dict, key: str) -> int:
    try:
        return int(payload[key])
    except Exception:
        raise ValueError(key)


def pack(groups):
    return [
        {
            "host": dev[0].host_name,
            "device": dev[0].device_name,
            "channels": sorted({cfg.channels for cfg in dev}),
            "sample_rates": sorted({cfg.sample_rate for cfg in dev if cfg.sample_rate != 1}),
            "sample_formats": sorted({str(cfg.sample_format) for cfg in dev}),
        }
        for dev in groups
        if dev
    ]


def find_config(kind: str, host: str, device: str, rate: int, ch: int, fmt: str):
    groups = (
        melaec3.get_supported_input_configs(
            history_len=HISTORY_LEN,
            num_calibration_packets=CALIBRATION,
            audio_buffer_seconds=AUDIO_BUF,
            resampler_quality=RESAMPLE_Q,
        )
        if kind == "input"
        else melaec3.get_supported_output_configs(
            history_len=HISTORY_LEN,
            num_calibration_packets=CALIBRATION,
            audio_buffer_seconds=AUDIO_BUF,
            resampler_quality=RESAMPLE_Q,
            frame_size=DEFAULT_FRAME,
        )
    )
    for devs in groups:
        for cfg in devs:
            if (
                cfg.host_name == host
                and cfg.device_name == device
                and cfg.sample_rate == rate
                and cfg.channels == ch
                and str(cfg.sample_format) == fmt
            ):
                return cfg
    return None

@app.get("/")
def index():
    return read("melodeus.html")


@app.post("/aec")
def init_aec():
    global stream
    outputs.clear()
    payload = request.get_json(silent=True) or {}
    try:
        rate = grab_int(payload, "target_sample_rate")
        frame = grab_int(payload, "frame_size")
        length = grab_int(payload, "filter_length")
    except ValueError as err:
        return jsonify(error=f"invalid {err.args[0]}"), 400
    stream = build_stream(rate, frame, length)
    return jsonify(target_sample_rate=rate, frame_size=frame, filter_length=length)


@app.get("/devices")
def devices():
    ins = melaec3.get_supported_input_configs(
        history_len=HISTORY_LEN,
        num_calibration_packets=CALIBRATION,
        audio_buffer_seconds=AUDIO_BUF,
        resampler_quality=RESAMPLE_Q,
    )
    outs = melaec3.get_supported_output_configs(
        history_len=HISTORY_LEN,
        num_calibration_packets=CALIBRATION,
        audio_buffer_seconds=AUDIO_BUF,
        resampler_quality=RESAMPLE_Q,
        frame_size=DEFAULT_FRAME,
    )
    return jsonify(inputs=pack(ins), outputs=pack(outs))


@app.post("/device")
async def add_device():
    if stream is None:
        return jsonify(error="aec not initialized"), 400
    payload = request.get_json(silent=True) or {}
    kind = payload.get("kind")
    if kind not in {"input", "output"}:
        return jsonify(error="kind must be input or output"), 400
    try:
        host = payload["host"]
        device = payload["device"]
        rate = int(payload["sample_rate"])
        ch = int(payload["channels"])
        fmt = str(payload["sample_format"])
    except Exception:
        return jsonify(error="invalid device payload"), 400
    cfg = find_config(kind, host, device, rate, ch, fmt)
    if cfg is None:
        return jsonify(error="config not found"), 404
    if kind == "input":
        await stream.add_input_device(cfg)
    else:
        outputs[(host, device, rate, ch, fmt)] = await stream.add_output_device(cfg)
    return jsonify(
        ok=True,
        added={
            "kind": kind,
            "host": host,
            "device": device,
            "sample_rate": rate,
            "channels": ch,
            "sample_format": fmt,
        },
    )


@app.delete("/device")
def remove_device():
    if stream is None:
        return jsonify(error="aec not initialized"), 400
    payload = request.get_json(silent=True) or {}
    kind = payload.get("kind")
    host = payload.get("host")
    device = payload.get("device")
    rate = payload.get("sample_rate")
    ch = payload.get("channels")
    fmt = payload.get("sample_format")
    if not all([kind, host, device, rate, ch, fmt]):
        return jsonify(error="invalid device payload"), 400
    cfg = find_config(kind, host, device, int(rate), int(ch), str(fmt))
    if cfg is None:
        return jsonify(error="config not found"), 404
    if kind == "input":
        stream.remove_input_device(cfg)
    else:
        stream.remove_output_device(cfg)
        outputs.pop((host, device, int(rate), int(ch), str(fmt)), None)
    return jsonify(ok=True, removed=payload)


if __name__ == "__main__":
    start_ws()
    app.run(host="0.0.0.0", port="5000", debug=False)
