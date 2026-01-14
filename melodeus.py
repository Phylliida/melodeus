from flask import Flask, abort, jsonify, request
from persistent_config import PersistentMelodeusConfig
from pathlib import Path
from audio_system import AudioSystem
from stt import AsyncSTT
import numpy as np
import base64
import json
from melaec3 import InputDeviceConfig, OutputDeviceConfig
import melaec3
import asyncio
import signal
import threading
import websockets
from werkzeug.serving import make_server

WEBSOCKET_PORT = 5001

def add_audio_system_device_callbacks(app, audio_system, loop: asyncio.AbstractEventLoop):
    """Register Flask routes that bridge into the running asyncio loop."""

    # nonsense needed to make async happy in flask
    async def run_in_loop(coro):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))

    @app.post("/api/select")
    async def select_device():
        if audio_system is None:
            abort(503, description="Audio system not started")
        payload = request.get_json(force=True, silent=True) or {}
        device_type = payload.get("type")
        cfg_dict = payload.get("config")
        if device_type not in {"input", "output"} or not isinstance(cfg_dict, dict):
            abort(400, description="Missing or invalid device payload")

        if device_type == "input":
            cfg = InputDeviceConfig.from_dict(cfg_dict)
            await run_in_loop(audio_system.add_input_device(cfg))
        else:
            cfg = OutputDeviceConfig.from_dict(cfg_dict)
            await run_in_loop(audio_system.add_output_device(cfg))

        return jsonify({"status": "ok"})

    @app.get("/api/selected")
    def selected_devices():
        if audio_system is None:
            abort(503, description="Audio system not started")
        return jsonify(
            {
                "inputs": [cfg.to_dict() for cfg in audio_system.state.input_devices],
                "outputs": [cfg.to_dict() for cfg in audio_system.state.output_devices],
            }
        )

    @app.get("/api/devices")
    def list_devices():
        if audio_system is None:
            abort(503, description="Audio system not started")
        configs = audio_system.get_supported_device_configs()

        def configs_to_dicts(config_list):
            return [[cfg.to_dict() for cfg in dev_configs] for dev_configs in config_list]

        return jsonify(
            {
                "inputs": configs_to_dicts(configs.get("inputs")),
                "outputs": configs_to_dicts(configs.get("outputs")),
            }
        )

    @app.delete("/api/devices")
    async def remove_device():
        if audio_system is None:
            abort(503, description="Audio system not started")
        payload = request.get_json(force=True, silent=True) or {}
        device_type = payload.get("type")
        cfg_dict = payload.get("config")
        if device_type not in {"input", "output"} or not isinstance(cfg_dict, dict):
            abort(400, description="Missing or invalid device payload")

        if device_type == "input":
            cfg = InputDeviceConfig.from_dict(cfg_dict)
            await run_in_loop(audio_system.remove_input_device(cfg))
        else:
            cfg = OutputDeviceConfig.from_dict(cfg_dict)
            await run_in_loop(audio_system.remove_output_device(cfg))

        return jsonify({"status": "ok"})


async def start_websocket_server(audio_system):
    connections = set()
    async def broadcast(msg: str):
        dead = []
        for ws in list(connections):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            connections.discard(ws)

    async def websocket_server(websocket):
        connections.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            connections.discard(websocket)

    async def serve_websocket():
        await websockets.serve(websocket_server, "0.0.0.0", WEBSOCKET_PORT)
        def b64(arr, dtype=np.float32):
            a = np.asarray(arr, dtype=dtype, order="C") # continguous
            return base64.b64encode(a.tobytes()).decode("ascii")

        async def audio_callback(input_channels, output_channels, audio_input_data, audio_output_data, aec_input_data, channel_vads):
            if len(aec_input_data) > 0: # don't spam with empty things or it gets congested
                payload = {
                    "type": "waveform",
                    "in_ch": input_channels,
                    "out_ch": output_channels,
                    "input": b64(audio_input_data),
                    "output": b64(audio_output_data),
                    "aec": b64(aec_input_data),
                    "vad": channel_vads,
                }
                payls = json.dumps(payload)
                await broadcast(payls)
                await asyncio.sleep(0) # hand to other stuff so we don't exhaust async
        await audio_system.add_callback(audio_callback)
        while True:
            await asyncio.sleep(0.1)

    ws_thread = threading.Thread(target=lambda: asyncio.run(serve_websocket()), daemon=True)
    ws_thread.start()  

async def main():
    root = Path(__file__).parent
    app = Flask(__name__, static_folder=str(root), static_url_path="")

    CONFIG_FILE = root / "config.yaml"
    config = PersistentMelodeusConfig.load_config(CONFIG_FILE)

    async with AudioSystem(config=config.audio) as audio_system:
        async with AsyncSTT(config=config.stt, audio_system=audio_system):
            await start_websocket_server(audio_system)

            @app.get("/")
            def index():
                return app.send_static_file("melodeus.html")

            loop = asyncio.get_running_loop()
            add_audio_system_device_callbacks(app, audio_system, loop)

            server = make_server("0.0.0.0", 5000, app)
            server.timeout = 0.1  # seconds per poll

            def serve_forever():
                with server:
                    server.serve_forever()

            server_thread = threading.Thread(target=serve_forever, daemon=True)
            server_thread.start()
            try:
                while True:
                    await asyncio.sleep(0.1)
            finally:
                server.shutdown()
                server_thread.join()


if __name__ == "__main__":
    asyncio.run(main())
