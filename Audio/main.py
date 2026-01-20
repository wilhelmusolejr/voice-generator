import time
import random
import os
import numpy as np
import librosa
import soundfile as sf
import re
from multiprocessing import Process

# ==========================
# USER CONFIGURATION
# ==========================

# Audio Settings
SR = 8000  # Sample rate (Hz)
USER_NAME = "user1"

# Duration Settings (in seconds)
BASE_DURATION_SECONDS = 1 * 3600 + 20 * 60  # 1 hour 20 minutes (80 min)
EXTRA_DURATION_MIN = 5 * 60  # Minimum extra duration (5 min)
EXTRA_DURATION_MAX = 15 * 60  # Maximum extra duration (15 min)

# Generation Settings
BACKGROUND_NOISES = ["fan", "white_noise", "none"]  # Types of background noise
AUDIOS_TO_GENERATE = 4  # Number of audio files to generate per background noise type
USE_MULTIPROCESSING = True  # Enable parallel processing

# Voice Processing Settings
SILENCE_CHANCE = 0.15  # Probability of adding silence instead of playing clip (0.0-1.0)
SILENCE_MIN = 0.15  # Minimum silence duration (seconds)
SILENCE_MAX = 0.6  # Maximum silence duration (seconds)
FADE_CHANCE = 0.25  # Probability of adding fade effect (0.0-1.0)
FADE_MIN = 0.7  # Minimum fade level
FADE_MAX = 0.9  # Maximum fade level
CLIP_TRIM_CHANCE = 0.2  # Probability of trimming clip end (0.0-1.0)
CLIP_TRIM_MIN = 0.85  # Minimum clip trim ratio
CLIP_TRIM_MAX = 0.95  # Maximum clip trim ratio

# Audio Mixing Settings
BG_NOISE_LEVEL = 0.01  # Background noise amplitude level
PEAK_NORMALIZATION = 0.9  # Peak normalization level (0.0-1.0)
FINAL_PEAK_NORMALIZATION = 0.95  # Final peak normalization after mixing

# ==========================
# BASE CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

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

    if random.random() < CLIP_TRIM_CHANCE:
        audio = audio[: int(len(audio) * random.uniform(CLIP_TRIM_MIN, CLIP_TRIM_MAX))]

    if USER_NAME == "g3ooorge":
        audio = soften_voice(audio)

    if random.random() < FADE_CHANCE:
        fade = np.linspace(1.0, random.uniform(FADE_MIN, FADE_MAX), len(audio))
        audio *= fade

    gain_db = random.uniform(-1.0, 1.5) * state["energy"]
    audio *= 10 ** (gain_db / 20)

    audio = mic_color(audio)
    state["audio"] = np.concatenate([state["audio"], audio])

def add_silence(seconds, state):
    silence = np.zeros(int(seconds * SR))
    state["audio"] = np.concatenate([state["audio"], silence])

def mix_background_noise(speech, bg_noise, level=None):
    if level is None:
        level = BG_NOISE_LEVEL
    noise_path = os.path.join(BASE_DIR, "voices", "bg_noise", f"{bg_noise}.mp3")
    if not os.path.exists(noise_path):
        return speech

    try:
        noise, _ = librosa.load(noise_path, sr=SR)

        if len(noise) < len(speech):
            noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))

        noise = noise[: len(speech)]
        return speech + noise * level
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        print(f"[WARNING] Failed to load background noise '{bg_noise}': {e}")
        print("[WARNING] Skipping background noise mixing for this file")
        return speech

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

        if random.random() < SILENCE_CHANCE:
            add_silence(random.uniform(SILENCE_MIN, SILENCE_MAX), state)
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

    EXTRA_SECONDS = random.randint(EXTRA_DURATION_MIN, EXTRA_DURATION_MAX)
    TARGET_SECONDS = BASE_DURATION_SECONDS + EXTRA_SECONDS

    print(f"[JOB START] {bg_noise} v{version}")

    while len(state["audio"]) / SR < TARGET_SECONDS:
        generate_round(state)

    audio = state["audio"]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * PEAK_NORMALIZATION

    if bg_noise != "none":
        audio = mix_background_noise(audio, bg_noise)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * FINAL_PEAK_NORMALIZATION

    out_dir = os.path.join(OUTPUT_ROOT, bg_noise)
    os.makedirs(out_dir, exist_ok=True)

    # ------------
    file_name = 0
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
    processes = []

    if USE_MULTIPROCESSING:
        for bg in BACKGROUND_NOISES:
            p = Process(
                target=run_bg_noise_job,
                args=(bg, AUDIOS_TO_GENERATE)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        for bg in BACKGROUND_NOISES:
            run_bg_noise_job(bg, AUDIOS_TO_GENERATE)

    print("\nAll audio generation jobs completed.")
