import time
import random
import os
import numpy as np
import librosa
import soundfile as sf

# ==========================
# GLOBAL AUDIO BUFFER
# ==========================
SR = 22050
output_audio = np.array([], dtype=np.float32)

total_silence_seconds = 0.0

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

current_energy = 0.3

# ==========================
# FUNCTIONS
# ==========================

def get_current_phase(elapsed_seconds):
    if elapsed_seconds < 30:
        return "early"
    elif elapsed_seconds < 90:
        return "mid"
    elif elapsed_seconds < 135:
        return "late"
    else:
        return "end"


def play_random_clip_from(source):
    global output_audio, current_energy

    folder = f"voices/{source}"
    files = [f for f in os.listdir(folder) if f.endswith(".mp3")]

    if not files:
        return

    file = random.choice(files)
    audio, _ = librosa.load(os.path.join(folder, file), sr=SR)

    intensity = INTENSITY.get(source, 0.4)

    # --- emotional carry (keep this) ---
    current_energy = (current_energy * 0.7) + (intensity * 0.3)

    # --- trailing off (human behavior) ---
    if random.random() < 0.2:
        cut = int(len(audio) * random.uniform(0.85, 0.95))
        audio = audio[:cut]

    # --- mic distance / posture change ---
    if random.random() < 0.25:
        fade = np.linspace(1.0, random.uniform(0.7, 0.9), len(audio))
        audio *= fade

    # --- very light gain variation ---
    gain_db = random.uniform(-1.0, 1.5) * current_energy
    audio *= 10 ** (gain_db / 20)

    output_audio = np.concatenate([output_audio, audio])

def add_silence(seconds):
    global output_audio, total_silence_seconds
    silence = np.zeros(int(seconds * SR))
    output_audio = np.concatenate([output_audio, silence])
    total_silence_seconds += seconds


# ==========================
# ROUND GENERATOR
# ==========================

def generate_round():
    global current_energy

    # small reset at round start (feels human)
    current_energy *= random.uniform(0.6, 0.85)

    start_time = time.time()

    for source in ROUND_SEQUENCE:
        elapsed = time.time() - start_time
        phase = get_current_phase(elapsed)

        if source not in PHASE_RULES[phase]:
            continue

        if random.random() > PLAY_PROBABILITY[source]:
            continue

        # false start (human hesitation)
        if random.random() < 0.15:
            add_silence(random.uniform(0.3, 1.0))
            continue

        play_random_clip_from(source)

        # human silence distribution
        r = random.random()
        if r < 0.5:
            pause = random.uniform(0.05, 0.3)   # very short (most common)
        elif r < 0.9:
            pause = random.uniform(0.4, 1.2)    # normal pause
        else:
            pause = random.uniform(2.5, 5.0)    # rare long pause

        add_silence(pause)

    # between-round pause
    add_silence(random.uniform(3, 5))


# ==========================
# MAIN LOOP — 10 MINUTES
# ==========================

TARGET_TOTAL_SECONDS = 4200 

print("Generating multi-round session...")

round_count = 0
while len(output_audio) / SR < TARGET_TOTAL_SECONDS:
    round_count += 1
    print(f"--- Round {round_count} ---")
    generate_round()

# ==========================
# FINALIZE AUDIO
# ==========================

peak = np.max(np.abs(output_audio))
if peak > 0:
    output_audio = output_audio / peak * 0.9

sf.write("output_final8.wav", output_audio, SR)
print("Generated output_final.wav")

total_audio_seconds = len(output_audio) / SR
silence_minutes = total_silence_seconds / 60
total_minutes = total_audio_seconds / 60
silence_percentage = (total_silence_seconds / total_audio_seconds) * 100

print(f"Total duration: {total_minutes:.2f} minutes")
print(f"Total silence: {silence_minutes:.2f} minutes")
print(f"Silence percentage: {silence_percentage:.1f}%")