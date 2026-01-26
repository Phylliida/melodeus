import numpy as np
import traceback
import base64
from flask import Flask, abort, jsonify, request
import websockets
import asyncio
import uuid
import json

def serialize_context(update):
    return {
        "type": "context",
        "action": getattr(update.action, "value", update.action),
        "uuid": update.uuid,
        "author": update.author,
        "message": update.message,
    }

def b64(arr, dtype=np.float32):
    a = np.asarray(arr, dtype=dtype, order="C") # continguous
    return base64.b64encode(a.tobytes()).decode("ascii")

class AsyncWebsocketServer(object):
    def __init__(self, config, app, audio_system, stt, tts, context):
        self.app = app
        self.config = config
        self.audio_system = audio_system
        self.stt = stt
        self.tts = tts
        self.context = context
       
    async def __aenter__(self):
        @self.app.get("/api/uiconfig")
        def get_ui():
            return jsonify({"show_waveforms": self.config.show_waveforms})

        @self.app.post("/api/uiconfig/waveforms")
        def toggle_waveforms():
            payload = request.get_json(force=True, silent=True) or {}
            self.config.show_waveforms = bool(payload.get("show_waveforms"))
            self.config.persist_data()
            return jsonify({"show_waveforms": self.config.show_waveforms})

        self.connections = set()

        self.server = await websockets.serve(self.websocket_server, self.config.host, self.config.port)

        await self.audio_system.add_callback(self.audio_callback)
        await self.stt.add_callback(self.stt_callback)
        await self.context.add_callback(self.context_callback)
    
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        return
    
    async def broadcast(self, msg: str):
        dead = []
        for ws in list(self.connections):
            try:
                await ws.send(msg)
            except Exception:
                print(traceback.print_exc())
                dead.append(ws)
        for ws in dead:
            self.connections.discard(ws)

    async def websocket_server(self, websocket):
        self.connections.add(websocket)
        try:
            async def send_state_to_websocket(update):
                try:
                    await websocket.send(json.dumps(serialize_context(update)))
                except Exception:
                    pass
            # fastforward to current history state
            await self.context.fetch_current_state(send_state_to_websocket)
            await websocket.wait_closed()
        finally:
            self.connections.discard(websocket)

    async def audio_callback(self, input_channels, output_channels, audio_input_data, audio_output_data, aec_input_data, channel_vads):
        if not self.config.show_waveforms:
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
            payload = json.dumps(payload)
            await self.broadcast(payload)
            await asyncio.sleep(0) # hand to other stuff so we don't exhaust async
                
    async def stt_callback(self, stt_result):
        print(stt_result.text)
        await self.context.update(
            uuid=stt_result.message_id,
            author=stt_result.speaker_name,
            message=stt_result.text)
        await self.get_model_response("default")
    
    async def context_callback(self, update):
        payload = serialize_context(update)
        await self.broadcast(json.dumps(payload))

    async def get_model_response(self, author_id):
        async def text_generator():
            yield "Hello there the green beans are tasty! Do you think so?"
            yield "Wow the green beans"
        
        response_uuid = str(uuid.uuid4())
        # wraps the text generator to add it to context
        async def text_generator_wrapper(text_generator):
            full_text = ""
            async for text in text_generator():
                full_text += text
                await self.context.update(
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
                await self.context.delete(
                    uuid=response_uuid
                )
            else:
                await self.context.update(
                    uuid=response_uuid,
                    author=author_id,
                    message=text
                )

        await self.tts.speak_text(
            tts_id=author_id,
            text_generator=text_generator_wrapper(text_generator),
            first_audio_callback=first_audio_callback,
            interrupted_callback=interrupted
        )