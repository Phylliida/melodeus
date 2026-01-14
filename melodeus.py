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


def add_audio_system_device_callbacks(app, audio_system, loop: asyncio.AbstractEventLoop):
    """Register Flask routes that bridge into the running asyncio loop."""

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
