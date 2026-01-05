

from flask import Flask, jsonify, request
app = Flask(__name__, static_folder=str(root), static_url_path="")


from persistent_config import PersistentConfig
