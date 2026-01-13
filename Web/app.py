from flask import Flask, render_template, send_file, request
from background_runner import run_generator_async
import subprocess
import os
import uuid
import threading
import random

app = Flask(__name__)

# ===== PATH SETUP =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AUDIO_DIR = os.path.join(BASE_DIR, "Audio")
OUTPUT_DIR = os.path.join(AUDIO_DIR, "output")
GENERATOR_SCRIPT = os.path.join(AUDIO_DIR, "main.py")

os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_MAPPING = {
    "botfrag666": "none",
    "elooo2092": "white_noise",
    "echogreg": "fan",
    "kooooalaid": "none",
    "g3ooorge": "white_noise"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    user = data.get("user", "user1")
    bg_noise = data.get("bg_noise", "none")

    effects = data.get("effects", {})
    dog_howl = str(effects.get("dog_howl", False))
    car_horn = str(effects.get("car_horn", False))

    job_id = uuid.uuid4().hex
    filename = f"{job_id}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # run audio generator
    args = [
            "python",
            GENERATOR_SCRIPT,
            output_path,
            user,
            bg_noise,
            dog_howl,
            car_horn
        ]

    threading.Thread(
        target=run_generator_async,
        args=(args,),
        daemon=True
    ).start()
    
    return {
        "status": "started",
        "job_id": job_id,
        "audio_url": f"/audio/{job_id}"
    }

@app.route("/audio/<job_id>")
def get_audio(job_id):
    path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    if not os.path.exists(path):
        return {"status": "processing"}, 202

    return send_file(path, mimetype="audio/wav")

@app.route("/random_audio", methods=["POST"])
def random_audio():
    data = request.get_json()
    user = data.get("user")
    if not user or user not in USER_MAPPING:
        return {"error": "Invalid user"}, 400
    folder = USER_MAPPING[user]
    folder_path = os.path.join(OUTPUT_DIR, folder)
    if not os.path.exists(folder_path):
        return {"error": f"Folder {folder} not found"}, 404
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not files:
        return {"error": f"No audios in {folder}"}, 404
    chosen = random.choice(files)
    return {"filename": f"{folder}/{chosen}"}

@app.route("/audio_file/<path:filepath>")
def get_audio_file(filepath):
    full_path = os.path.join(OUTPUT_DIR, filepath)
    if not os.path.exists(full_path):
        return {"error": "File not found"}, 404
    # Ensure it's within OUTPUT_DIR
    if not os.path.abspath(full_path).startswith(os.path.abspath(OUTPUT_DIR)):
        return {"error": "Invalid path"}, 400
    return send_file(full_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
