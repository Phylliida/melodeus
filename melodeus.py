from flask import Flask, abort, jsonify, request
from persistent_config import PersistentMelodeusConfig
from pathlib import Path
from audio_system import AudioSystem
from stt import AsyncSTT
from melaec3 import InputDeviceConfig, OutputDeviceConfig
import melaec3
import asyncio
import signal
import threading
from werkzeug.serving import make_server

async def add_audio_system_device_callbacks(app, audio_system):
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
            await audio_system.add_input_device(cfg)
        else:
            cfg = OutputDeviceConfig.from_dict(cfg_dict)
            await audio_system.add_output_device(cfg)

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
            await audio_system.remove_input_device(cfg)
        else:
            cfg = OutputDeviceConfig.from_dict(cfg_dict)
            await audio_system.remove_output_device(cfg)

        return jsonify({"status": "ok"})


async def main():
    root = Path(__file__).parent
    app = Flask(__name__, static_folder=str(root), static_url_path="")

    CONFIG_FILE = root / "config.yaml"
    config = PersistentMelodeusConfig.load_config(CONFIG_FILE)

    async with AudioSystem(config=config.audio) as audio_system:
        async with AsyncSTT(config=config.stt, audio_system=audio_system):
            @app.get("/")
            def index():
                return app.send_static_file("melodeus.html")
            await add_audio_system_device_callbacks(app, audio_system)
            server = make_server("0.0.0.0", 5000, app)
            server.timeout = 0.1  # seconds per poll
            try:
                while True:
                    server.handle_request()  # one request per call; returns immediately after timeout
                    await asyncio.sleep(0) # defer to other async stuff
            finally:
                server.server_close()



if __name__ == "__main__":
    asyncio.run(main())
