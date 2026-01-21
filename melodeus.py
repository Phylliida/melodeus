from flask import Flask, abort, jsonify, request
from persistent_config import PersistentMelodeusConfig
from pathlib import Path
from dataclasses import asdict
from audio_system import AudioSystem, load_wav
from stt import AsyncSTT
from tts import AsyncTTS
import uuid
import numpy as np
import base64
import math
import json
from melaec3 import InputDeviceConfig, OutputDeviceConfig
import melaec3
import asyncio
import signal
import threading
import websockets
from werkzeug.serving import make_server
from context_manager import AsyncContextManager

WEBSOCKET_PORT = 5001
root = Path(__file__).parent

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
            await run_in_loop(asyncio.sleep(0.1))
        
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

async def start_websocket_server(app, audio_system, ui_config, stt_system, tts_system, context):
    connections = set()

    @app.get("/api/uiconfig")
    def get_ui():
        return jsonify({"show_waveforms": ui_config.show_waveforms})

    @app.post("/api/uiconfig/waveforms")
    def toggle_waveforms():
        payload = request.get_json(force=True, silent=True) or {}
        ui_config.show_waveforms = bool(payload.get("show_waveforms"))
        ui_config.persist_data()
        return jsonify({"show_waveforms": ui_config.show_waveforms})

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
            async def send_state_to_websocket(update):
                try:
                    update = update.to_dict()
                    update['type'] = 'context'
                    await ws.send(json.dumps(update))
                except Exception:
                    pass
            # fastforward to current history state
            await context.fetch_current_state(send_state_to_websocket)
            await websocket.wait_closed()
        finally:
            connections.discard(websocket)

    async def serve_websocket():
        await websockets.serve(websocket_server, "0.0.0.0", WEBSOCKET_PORT)
        def b64(arr, dtype=np.float32):
            a = np.asarray(arr, dtype=dtype, order="C") # continguous
            return base64.b64encode(a.tobytes()).decode("ascii")

        async def audio_callback(input_channels, output_channels, audio_input_data, audio_output_data, aec_input_data, channel_vads):
            if not ui_config.show_waveforms:
                return
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
        async def stt_callback(stt_result):
            await context.update(
                uuid=stt_result.message_id,
                author=stt_result.speaker_name,
                message=stt_result.text)
            await get_model_response("default")
        
        async def get_model_response(author_id):
            async def text_generator():
                yield "Hello there the green beans are tasty! Do you think so?"
                yield "Wow the green beans"
            
            response_uuid = str(uuid.uuid4())
            # wraps the text generator to add it to context
            async def text_generator_wrapper(text_generator):
                full_text = ""
                async for text in text_generator():
                    full_text += text
                    await context.update(
                        uuid=response_uuid,
                        author=author_id,
                        message=full_text
                    )
                    yield text

            async def first_audio_callback():
                pass
            async def interrupted(text, outputted_audio_duration):
                # if outputted less than 2 seconds of audio, just do blank
                if outputted_audio_duration < 2:
                    text = ""
                if text == "":
                    await context.delete(
                        uuid=response_uuid
                    )
                else:
                    await context.update(
                        uuid=response_uuid,
                        author=author_id,
                        message=text
                    )
    
            await tts_system.speak_text(
                tts_id=author_id,
                text_generator=text_generator_wrapper(text_generator),
                first_audio_callback=first_audio_callback,
                interrupted_callback=interrupted
            )

        async def context_callback(update):
            payload = asdict(update)
            payload['type'] = 'context'
            await broadcast(payload)

        await audio_system.add_callback(audio_callback)
        await stt_system.add_callback(stt_callback)
        await context.add_callback(context_callback)
        while True:
            await asyncio.sleep(0.1)

    ws_thread = threading.Thread(target=lambda: asyncio.run(serve_websocket()), daemon=True)
    ws_thread.start()  

async def main():
    app = Flask(__name__, static_folder=str(root), static_url_path="")

    CONFIG_FILE = root / "config.yaml"
    config = PersistentMelodeusConfig.load_config(CONFIG_FILE)

    async with AudioSystem(config=config.audio) as audio_system:
        async with AsyncSTT(config=config.stt, audio_system=audio_system) as stt_system:
            async with AsyncTTS(config=config.tts, audio_system=audio_system) as tts_system:
                async with AsyncContextManager(config=config.context, voices=config.tts.voices) as context:
                    await start_websocket_server(app, audio_system, config.ui, stt_system, tts_system, context)

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
