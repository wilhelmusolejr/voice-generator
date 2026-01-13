import time
import random
import os
import numpy as np
import librosa
import soundfile as sf
import re
from multiprocessing import Process

# ==========================
# BASE CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

SR = 16000

user = "user1"

# ==========================
# ROUND LOGIC
# ==========================

ROUND_SEQUENCE = [
    "greetings",
    "strategy",
    "enemy_info",
    "random",
    "strategy",
    "enemy_info",
    "random",
    "enemy_info",
    "strategy",
    "enemy_info",
    "round_result",
]

PHASE_RULES = {
    "early": ["greetings", "strategy", "random"],
    "mid": ["strategy", "enemy_info", "random"],
    "late": ["enemy_info", "strategy", "random"],
    "end": ["round_result"],
}

PLAY_PROBABILITY = {
    "greetings": 0.6,
    "strategy": 0.7,
    "enemy_info": 0.8,
    "random": 0.4,
    "round_result": 1.0,
}

INTENSITY = {
    "greetings": 0.2,
    "strategy": 0.4,
    "enemy_info": 0.8,
    "random": 0.3,
    "round_result": 0.5,
}

# ==========================
# VOICE FX
# ==========================

def soften_voice(audio):
    audio *= 0.9
    return librosa.effects.preemphasis(audio, coef=0.85)

def mic_color(audio):
    return librosa.effects.preemphasis(audio, coef=0.93)

# ==========================
# CORE FUNCTIONS
# ==========================

def get_current_phase(elapsed):
    if elapsed < 30:
        return "early"
    elif elapsed < 90:
        return "mid"
    elif elapsed < 135:
        return "late"
    return "end"

def play_random_clip_from(source, state):
    folder = os.path.join(BASE_DIR, "voices", source)
    if not os.path.exists(folder):
        return

    files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
    if not files:
        return

    file = random.choice(files)
    audio, _ = librosa.load(os.path.join(folder, file), sr=SR)

    intensity = INTENSITY.get(source, 0.4)
    state["energy"] = state["energy"] * 0.7 + intensity * 0.3

    if random.random() < 0.2:
        audio = audio[: int(len(audio) * random.uniform(0.85, 0.95))]

    if user == "g3ooorge":
        audio = soften_voice(audio)

    if random.random() < 0.25:
        fade = np.linspace(1.0, random.uniform(0.7, 0.9), len(audio))
        audio *= fade

    gain_db = random.uniform(-1.0, 1.5) * state["energy"]
    audio *= 10 ** (gain_db / 20)

    audio = mic_color(audio)
    state["audio"] = np.concatenate([state["audio"], audio])

def add_silence(seconds, state):
    silence = np.zeros(int(seconds * SR))
    state["audio"] = np.concatenate([state["audio"], silence])

def mix_background_noise(speech, bg_noise, level=0.01):
    noise_path = os.path.join(BASE_DIR, "voices", "bg_noise", f"{bg_noise}.mp3")
    if not os.path.exists(noise_path):
        return speech

    noise, _ = librosa.load(noise_path, sr=SR)

    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))

    noise = noise[: len(speech)]
    return speech + noise * level

# ==========================
# ROUND GENERATION
# ==========================

def generate_round(state):
    state["energy"] *= random.uniform(0.6, 0.85)
    start = time.time()

    for source in ROUND_SEQUENCE:
        phase = get_current_phase(time.time() - start)

        if source not in PHASE_RULES[phase]:
            continue
        if random.random() > PLAY_PROBABILITY[source]:
            continue

        if random.random() < 0.15:
            add_silence(random.uniform(0.15, 0.6), state)
            continue

        play_random_clip_from(source, state)

        r = random.random()
        if r < 0.5:
            pause = random.uniform(0.05, 0.3)
        elif r < 0.9:
            pause = random.uniform(0.4, 1.2)
        else:
            pause = random.uniform(2.5, 5.0)

        add_silence(pause, state)

    add_silence(random.uniform(1.0, 3.0), state)

# ==========================
# AUDIO JOB
# ==========================

def generate_audio_job(bg_noise, version):
    state = {
        "audio": np.array([], dtype=np.float32),
        "energy": 0.3,
    }

    BASE_SECONDS = 1 * 3600 + 20 * 60
    EXTRA_SECONDS = random.randint(5 * 60, 15 * 60)
    TARGET_SECONDS = BASE_SECONDS + EXTRA_SECONDS

    print(f"[JOB START] {bg_noise} v{version}")

    while len(state["audio"]) / SR < TARGET_SECONDS:
        generate_round(state)

    audio = state["audio"]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    if bg_noise != "none":
        audio = mix_background_noise(audio, bg_noise)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    out_dir = os.path.join(OUTPUT_ROOT, bg_noise)
    os.makedirs(out_dir, exist_ok=True)

    # ------------
    file_name = 20
    files = os.listdir(out_dir)

    # Extract numbers from filenames and find the highest
    numbers = []
    for file in files:
        # Assuming filenames contain numbers like fan_1.wav
        match = re.search(r'\d+', file)
        if match:
            numbers.append(int(match.group()))

    if numbers:
        highest_number = max(numbers)
        file_name = highest_number + 1

    out_path = os.path.join(out_dir, f"{file_name}.wav")
    sf.write(out_path, audio, SR)

    print(f"[JOB DONE] {out_path}")

# ==========================
# PARALLEL RUNNER
# ==========================

def run_bg_noise_job(bg_noise, audios_to_add):
    for v in range(1, audios_to_add + 1):
        generate_audio_job(bg_noise, v)

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    bg_noises = ["fan", "white_noise", "none"]
    audios_to_add = 1

    processes = []

    for bg in bg_noises:
        p = Process(
            target=run_bg_noise_job,
            args=(bg, audios_to_add)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll audio generation jobs completed.")
