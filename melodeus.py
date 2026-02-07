from flask import Flask, abort, jsonify, request
from pathlib import Path
from dataclasses import asdict
import uuid
import numpy as np
import math
import melaec3
import asyncio
import signal
import threading
import websockets
import traceback
from werkzeug.serving import make_server

from melaec3 import InputDeviceConfig, OutputDeviceConfig

from persistent_config import PersistentMelodeusConfig
from audio_system import AudioSystem, load_wav
from stt_elevenlabs import AsyncElevenLabsSTT
from tts import AsyncTTS
from ws_server import AsyncWebsocketServer
from context_manager import AsyncContextManager

WEBSOCKET_PORT = 5001
root = Path(__file__).parent

global loop
async def run_in_loop(coro):
    global loop
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))

def add_audio_system_device_callbacks(app, audio_system, ws_server, loop: asyncio.AbstractEventLoop):
    """Register Flask routes that bridge into the running asyncio loop."""

    # nonsense needed to make async happy in flask


    @app.post("/api/interrupt")
    async def interrupt():
        ws_server.interrupted = True


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

    @app.post("/api/calibrate")
    async def calibrate():
        if audio_system is None:
            abort(503, description="Audio system not started")
        await run_in_loop(audio_system.calibrate())
        return jsonify({"status": "ok"})

    @app.post("/play")
    async def play_sample():
        if audio_system is None:
            abort(503, description="Audio system not started")
        sample_rate, channels, audio_seconds, audio_data = load_wav(root / "example_talking.wav")
        resampler_quality = 5 # some default value is fine
        output_streams = []
        print(len(audio_data), "samples")
        # map all audio channels to first channel output, for testing
        channel_map = {channel: [0] for channel in range(channels)}
        for output_device in audio_system.get_connected_output_devices():
            output_stream = await run_in_loop(audio_system.begin_audio_stream(
                output_device,
                channels,
                channel_map,
                math.ceil(audio_seconds),
                sample_rate,
                resampler_quality))
            print(audio_seconds)
            output_stream.queue_audio(audio_data)
            output_streams.append((output_device, output_stream))
        done = False
        while not done:
            done = True
            for output_device, stream in output_streams:
                if stream.num_queued_samples != 0:
                    done = False
            # poll until done
            await asyncio.sleep(0.1)
        
        # clean up
        for output_device, stream in output_streams:
            await run_in_loop(audio_system.end_audio_stream(output_device, stream))
        
        return jsonify(
            {
                "status": "ok",
                "outputs_started": len(output_streams),
                "duration_sec": audio_seconds,
                "sample_rate": sample_rate,
            }
        )

async def main():
    app = Flask(__name__, static_folder=str(root), static_url_path="")

    CONFIG_FILE = root / "config.yaml"
    config = PersistentMelodeusConfig.load_config(CONFIG_FILE)
    global loop
    loop = asyncio.get_running_loop()
    async with AudioSystem(config=config.audio) as audio_system:
        async with AsyncElevenLabsSTT(key=config.secrets.elevenlabs_api_key, config=config.stt, audio_system=audio_system) as stt_system:
            async with AsyncTTS(key=config.secrets.elevenlabs_api_key, config=config.tts, audio_system=audio_system, stt=stt_system) as tts_system:
                async with AsyncContextManager(config=config.context, voices=config.tts.voices) as context:
                    async with AsyncWebsocketServer(config=config.ui, app=app, audio_system=audio_system, stt=stt_system, tts=tts_system, context=context) as ws_server:
                        @app.get("/")
                        def index():
                            return app.send_static_file("melodeus.html")

                        @app.delete("/api/context/<msg_uuid>")
                        async def delete_context_message(msg_uuid):
                            await run_in_loop(context.delete(uuid=msg_uuid))
                            return jsonify({"status": "ok"})

                        @app.post("/api/context/<msg_uuid>")
                        async def edit_context_message(msg_uuid):
                            payload = request.get_json(force=True, silent=True) or {}
                            if "message" not in payload:
                                abort(400, description="Missing message")
                            author = payload.get("author", "")
                            message = payload.get("message") or ""
                            await run_in_loop(context.update(uuid=msg_uuid, author=author, message=message))
                            return jsonify({"status": "ok"})
                        
                        add_audio_system_device_callbacks(app, audio_system, ws_server, loop)

                        server = make_server("0.0.0.0", 5045, app)
                        server.timeout = 0.1  # seconds per poll

                        def serve_forever():
                            with server:
                                server.serve_forever()

                        server_thread = threading.Thread(target=serve_forever, daemon=True)
                        server_thread.start()
                        try:
                            while True:
                                await asyncio.sleep(1)
                        finally:
                            server.shutdown()
                            server_thread.join()


if __name__ == "__main__":
    asyncio.run(main())
