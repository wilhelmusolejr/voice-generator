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

# SR 16000 is excellent for mimicking Discord/VoIP quality
SR = 16000 

user = "user1"

# ==========================
# 1. UPDATED ROUND LOGIC
# ==========================

# Added "round_start" at the beginning
ROUND_SEQUENCE = [
    "greetings",
    "round_start",  
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

# Updated Rules to include round_start
PHASE_RULES = {
    "early": ["greetings", "round_start", "strategy", "random"],
    "mid": ["strategy", "enemy_info", "random"],
    "late": ["enemy_info", "strategy", "random"],
    "end": ["round_result"],
}

# Probability of a clip playing if it is selected
PLAY_PROBABILITY = {
    "greetings": 0.8,
    "round_start": 0.95, # Almost always play the start
    "strategy": 0.7,
    "enemy_info": 0.85,  # High priority info
    "random": 0.35,
    "round_result": 1.0, # Always comment on the result
}

# Energy multiplier (how loud/excited the clip is)
INTENSITY = {
    "greetings": 0.7,
    "round_start": 0.9,  # High energy start
    "strategy": 0.5,
    "enemy_info": 0.9,   # Urgent
    "random": 0.3,
    "round_result": 0.6,
}

# ==========================
# 2. IMPROVED VOICE FX
# ==========================

def soften_voice(audio):
    audio *= 0.9
    return librosa.effects.preemphasis(audio, coef=0.85)

def mic_color(audio):
    # Mimics the frequency curve of a cheap headset
    return librosa.effects.preemphasis(audio, coef=0.95)

def simple_limiter(audio, threshold=0.8):
    # Compresses loud peaks like a real gaming mic
    return np.clip(audio, -threshold, threshold)

# ==========================
# CORE FUNCTIONS
# ==========================

def get_current_phase(elapsed):
    # Logic updated to prioritize the immediate start
    if elapsed < 15:
        return "early"
    elif elapsed < 80:
        return "mid"
    elif elapsed < 120:
        return "late"
    return "end"

def play_random_clip_from(source, state):
    folder = os.path.join(BASE_DIR, "voices_ai", source)
    if not os.path.exists(folder):
        return

    files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
    if not files:
        return

    file = random.choice(files)
    audio, _ = librosa.load(os.path.join(folder, file), sr=SR)

    # Calculate Energy
    base_intensity = INTENSITY.get(source, 0.4)
    state["energy"] = state["energy"] * 0.7 + base_intensity * 0.3

    # FX: Random Truncation (Releasing PTT too early)
    if random.random() < 0.15:
        audio = audio[: int(len(audio) * random.uniform(0.90, 0.98))]

    # FX: Voice Softening
    if user == "g3ooorge":
        audio = soften_voice(audio)

    # FX: Fade out (Moving away from mic)
    if random.random() < 0.20:
        fade = np.linspace(1.0, random.uniform(0.7, 0.9), len(audio))
        audio *= fade

    # Apply Gain based on Energy
    gain_db = random.uniform(-1.0, 2.0) * state["energy"]
    audio *= 10 ** (gain_db / 20)

    # Apply Color and Limiting
    audio = mic_color(audio)
    audio = simple_limiter(audio)

    state["audio"] = np.concatenate([state["audio"], audio])

# NEW FEATURE: Interrupters (Keyboard clicks, coughs)
def play_interrupter(state):
    folder = os.path.join(BASE_DIR, "voices_ai", "interrupts")
    if not os.path.exists(folder):
        return # Skip if folder doesn't exist

    files = [f for f in os.listdir(folder) if f.endswith(".mp3") or f.endswith(".wav")]
    if not files:
        return

    # Only play interrupter occasionally (5% chance per silence block)
    if random.random() > 0.05:
        return

    file = random.choice(files)
    audio, _ = librosa.load(os.path.join(folder, file), sr=SR)
    
    # Make interrupters quiet (background noise)
    audio *= 0.15 
    state["audio"] = np.concatenate([state["audio"], audio])

def add_silence(seconds, state):
    # Check for interrupter before silence
    play_interrupter(state)
    
    silence = np.zeros(int(seconds * SR))
    state["audio"] = np.concatenate([state["audio"], silence])

def mix_background_noise(speech, bg_noise, base_level=0.012):
    noise_path = os.path.join(BASE_DIR, "voices_ai", "bg_noise", f"{bg_noise}.mp3")
    if not os.path.exists(noise_path):
        print(f"Warning: Bg noise {bg_noise} not found.")
        return speech

    noise, _ = librosa.load(noise_path, sr=SR)

    # Loop noise to match speech length
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[: len(speech)]

    # IMPROVEMENT: Dynamic Noise Level
    # This makes the fan noise "breathe" slightly, so it's not a static loop
    # It mimics the user moving slightly in their chair
    dynamic_level = np.linspace(base_level, base_level * random.uniform(0.8, 1.2), len(speech))
    
    return speech + (noise * dynamic_level)

# ==========================
# ROUND GENERATION
# ==========================

def generate_round(state):
    # Reset energy slightly at start of round
    state["energy"] = random.uniform(0.5, 0.8)
    start = time.time()

    for source in ROUND_SEQUENCE:
        phase = get_current_phase(time.time() - start)

        # Check Logic Rules
        if source not in PHASE_RULES[phase]:
            continue
        if random.random() > PLAY_PROBABILITY[source]:
            continue

        # Random hesitation before speaking
        if random.random() < 0.15:
            add_silence(random.uniform(0.2, 0.8), state)

        play_random_clip_from(source, state)

        # Post-speech pause logic
        r = random.random()
        if r < 0.5:
            # Micro pause
            pause = random.uniform(0.1, 0.4)
        elif r < 0.85:
            # Conversational pause
            pause = random.uniform(0.5, 1.5)
        else:
            # Long "focusing on game" pause
            pause = random.uniform(3.0, 7.0)

        add_silence(pause, state)

    # Long break between rounds
    add_silence(random.uniform(2.0, 5.0), state)

# ==========================
# AUDIO JOB
# ==========================

def generate_audio_job(bg_noise, version):
    state = {
        "audio": np.array([], dtype=np.float32),
        "energy": 0.5,
    }

    # Generate roughly 1 Hour 20 Mins of audio + Random extra
    BASE_SECONDS = 1 * 3600 + 20 * 60
    EXTRA_SECONDS = random.randint(5 * 60, 15 * 60)
    TARGET_SECONDS = BASE_SECONDS + EXTRA_SECONDS

    print(f"[JOB START] {bg_noise} v{version} - Target: {TARGET_SECONDS/60:.1f} mins")

    while len(state["audio"]) / SR < TARGET_SECONDS:
        generate_round(state)

    audio = state["audio"]

    # Normalize (Prevent Clipping)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    # Add Environment
    if bg_noise != "none":
        audio = mix_background_noise(audio, bg_noise)

    # Final Safety Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.98

    # Save File
    out_dir = os.path.join(OUTPUT_ROOT, bg_noise)
    os.makedirs(out_dir, exist_ok=True)

    files = os.listdir(out_dir)
    numbers = []
    for file in files:
        match = re.search(r'\d+', file)
        if match:
            numbers.append(int(match.group()))

    file_name = 1
    if numbers:
        file_name = max(numbers) + 1

    out_path = os.path.join(out_dir, f"{file_name}.wav")
    sf.write(out_path, audio, SR)

    print(f"[JOB DONE] Saved to: {out_path}")

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
    # Ensure you have your folders created inside 'voices/'
    # Structure: voices/round_start, voices/greetings, voices/interrupts, etc.
    
    bg_noises = ["none"]
    audios_to_add = 1

    processes = []

    print("Starting Generation...")

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