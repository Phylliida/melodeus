from flask import Flask, jsonify, request
from persistent_config import PersistentMelodeusConfig
from pathlib import Path
from audio_system import AudioSystem

root = Path(__file__).parent
app = Flask(__name__, static_folder=str(root), static_url_path="")

CONFIG_FILE = root / "config.yaml"

config = PersistentMelodeusConfig.load_config(CONFIG_FILE)

with AudioSystem() as audio_system:
    pass
