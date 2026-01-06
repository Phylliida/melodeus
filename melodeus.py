from flask import Flask, abort, jsonify, request
from persistent_config import PersistentMelodeusConfig
from pathlib import Path
from audio_system import AudioSystem
from melaec3 import InputDeviceConfig, OutputDeviceConfig
import melaec3
import asyncio
import signal
import threading
from werkzeug.serving import make_server

root = Path(__file__).parent
app = Flask(__name__, static_folder=str(root), static_url_path="")

CONFIG_FILE = root / "config.yaml"

config = PersistentMelodeusConfig.load_config(CONFIG_FILE)
audio_system: AudioSystem | None = None
loop: asyncio.AbstractEventLoop | None = None
DEVICE_HISTORY_LEN = 100
DEVICE_CALIBRATION_PACKETS = 20
DEVICE_AUDIO_BUFFER_SECONDS = 20
DEVICE_RESAMPLER_QUALITY = 5
DEVICE_FRAME_SIZE = 160


def _run_coro(coro):
    if loop is None:
        abort(503, description="Audio loop not ready")
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _device_to_dict(cfg):
    return cfg.to_dict()


def _serve_http(stop_event: threading.Event):
    server = make_server("0.0.0.0", 5000, app)
    server.timeout = 1
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        server.server_close()


@app.get("/")
def index():
    return app.send_static_file("melodeus.html")


@app.get("/api/devices")
def list_devices():
    if audio_system is None:
        abort(503, description="Audio system not started")
    configs = {
        "inputs": melaec3.get_supported_input_configs(
            history_len=DEVICE_HISTORY_LEN,
            num_calibration_packets=DEVICE_CALIBRATION_PACKETS,
            audio_buffer_seconds=DEVICE_AUDIO_BUFFER_SECONDS,
            resampler_quality=DEVICE_RESAMPLER_QUALITY,
        ),
        "outputs": melaec3.get_supported_output_configs(
            history_len=DEVICE_HISTORY_LEN,
            num_calibration_packets=DEVICE_CALIBRATION_PACKETS,
            audio_buffer_seconds=DEVICE_AUDIO_BUFFER_SECONDS,
            resampler_quality=DEVICE_RESAMPLER_QUALITY,
            frame_size=DEVICE_FRAME_SIZE,
        ),
    }

    def flatten(config_list):
        flat = []
        for group in config_list or []:
            for cfg in group:
                flat.append(_device_to_dict(cfg))
        return flat

    return jsonify(
        {
            "inputs": flatten(configs.get("inputs")),
            "outputs": flatten(configs.get("outputs")),
        }
    )


@app.get("/api/selected")
def selected_devices():
    if audio_system is None:
        abort(503, description="Audio system not started")
    return jsonify(
        {
            "inputs": [_device_to_dict(cfg) for cfg in audio_system.state.input_devices],
            "outputs": [_device_to_dict(cfg) for cfg in audio_system.state.output_devices],
        }
    )


@app.post("/api/select")
def select_device():
    if audio_system is None:
        abort(503, description="Audio system not started")
    payload = request.get_json(force=True, silent=True) or {}
    device_type = payload.get("type")
    cfg_dict = payload.get("config")
    if device_type not in {"input", "output"} or not isinstance(cfg_dict, dict):
        abort(400, description="Missing or invalid device payload")

    if device_type == "input":
        cfg = InputDeviceConfig.from_dict(cfg_dict)
        _run_coro(audio_system.add_input_device(cfg))
    else:
        cfg = OutputDeviceConfig.from_dict(cfg_dict)
        _run_coro(audio_system.add_output_device(cfg))

    return jsonify({"status": "ok"})


async def main():
    global audio_system, loop
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    server_stop = threading.Event()

    def _request_shutdown():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_shutdown())

    server_thread = threading.Thread(target=_serve_http, args=(server_stop,), daemon=True)

    async with AudioSystem(config=config.audio) as audio_system_instance:
        audio_system = audio_system_instance
        server_thread.start()
        try:
            await stop_event.wait()
        finally:
            server_stop.set()
            server_thread.join(timeout=2)
            audio_system = None


if __name__ == "__main__":
    asyncio.run(main())
