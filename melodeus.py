from flask import Flask, jsonify, request
from persistent_config import PersistentMelodeusConfig
from pathlib import Path
from audio_system import AudioSystem
from stt import AsyncSTT
import asyncio

root = Path(__file__).parent
app = Flask(__name__, static_folder=str(root), static_url_path="")

CONFIG_FILE = root / "config.yaml"

config = PersistentMelodeusConfig.load_config(CONFIG_FILE)

async def get_speech(speech_event):
    print(speech_event)

async def run():
    async with AudioSystem(config=config.audio) as audio_system:
        async with AsyncSTT(config=config.stt, audio_system=audio_system) as stt:
            await stt.add_callback(get_speech)



            @app.get("/")
            def index():
                return read("melodeus.html")
            
            @app.get("")

            app.run(host="0.0.0.0", port="5000", debug=False)
asyncio.run(run())
