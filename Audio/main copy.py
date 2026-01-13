import sys
import time
import random
import os
import numpy as np
import librosa
import soundfile as sf

# ==========================
# CLI ARGUMENTS (FROM FLASK)
# ==========================

# ---------------
# Defaults (standalone mode)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
version_num = 13

user = "user1"
# "fan" , "none", "white_noise"
bg_noise = "fan"
dog_howl = False
car_horn = False

output_path = os.path.join(
    BASE_DIR,
    "output",
    f"{bg_noise}_{version_num}.wav"
)

if len(sys.argv) >= 6:
    output_path = sys.argv[1]
    user = sys.argv[2]
    bg_noise = sys.argv[3]
    dog_howl = sys.argv[4].lower() == "true"
    car_horn = sys.argv[5].lower() == "true"

print("=== Generator Parameters ===")
print("Output Path:", output_path)
print("User:", user)
print("Background Noise:", bg_noise)
print("Dog Howl:", dog_howl)
print("Car Horn:", car_horn)
print("============================")

# sys.exit()

# ==========================
# GLOBAL AUDIO BUFFER
# ==========================

# SR = 22050
SR = 16000  
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

def soften_voice(audio):
    audio *= 0.9
    audio = librosa.effects.preemphasis(audio, coef=0.85)
    return audio

def harden_voice(audio):
    audio *= 1.1                      # louder
    audio = np.tanh(audio * 1.5)      # light saturation
    return audio

def feminine_tone(audio, sr):
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    return audio

def masculine_tone(audio, sr):
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
    audio *= 1.05
    return audio

def mic_color(audio):
    return librosa.effects.preemphasis(audio, coef=0.93)

def play_random_clip_from(source):
    global output_audio, current_energy

    folder = os.path.join(BASE_DIR, "voices", source)
    if not os.path.exists(folder):
        return

    files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
    if not files:
        return

    file = random.choice(files)
    audio, _ = librosa.load(os.path.join(folder, file), sr=SR)

    intensity = INTENSITY.get(source, 0.4)

    # 1️⃣ emotional carry (context, not sound)
    current_energy = (current_energy * 0.7) + (intensity * 0.3)

    # 2️⃣ trailing off (human behavior)
    if random.random() < 0.2:
        cut = int(len(audio) * random.uniform(0.85, 0.95))
        audio = audio[:cut]

    # 3️⃣ user voice character (soft / hard / style)
    if user == "g3ooorge":
        audio = soften_voice(audio)

    # 4️⃣ mic distance / posture shift
    if random.random() < 0.25:
        fade = np.linspace(1.0, random.uniform(0.7, 0.9), len(audio))
        audio *= fade

    # 5️⃣ loudness variation (AFTER tone shaping)
    gain_db = random.uniform(-1.0, 1.5) * current_energy
    audio *= 10 ** (gain_db / 20)

    # 6️⃣ mic color LAST (glues everything together)
    audio = mic_color(audio)

    output_audio = np.concatenate([output_audio, audio])

def add_silence(seconds):
    global output_audio, total_silence_seconds
    silence = np.zeros(int(seconds * SR))
    output_audio = np.concatenate([output_audio, silence])
    total_silence_seconds += seconds

def mix_background_noise(speech, noise_file, level=0.03):
    """
    Mix a background noise file under speech.
    level: 0.02–0.06 is subtle and safe
    """
    noise, _ = librosa.load(noise_file, sr=SR)

    # loop noise if too short
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)

    noise = noise[:len(speech)]

    # scale noise volume
    noise *= level

    return speech + noise

# ==========================
# ROUND GENERATOR
# ==========================

def generate_round():
    global current_energy

    # reset energy slightly per round
    current_energy *= random.uniform(0.6, 0.85)

    start_time = time.time()

    for source in ROUND_SEQUENCE:
        elapsed = time.time() - start_time
        phase = get_current_phase(elapsed)

        if source not in PHASE_RULES[phase]:
            continue

        if random.random() > PLAY_PROBABILITY[source]:
            continue

        # false start
        if random.random() < 0.15:
            add_silence(random.uniform(0.15, 0.6))
            continue

        play_random_clip_from(source)

        # silence distribution (20–30% target)
        r = random.random()
        if r < 0.5:
            pause = random.uniform(0.05, 0.3)
        elif r < 0.9:
            pause = random.uniform(0.4, 1.2)
        else:
            pause = random.uniform(2.5, 5.0)

        add_silence(pause)

    # between-round pause
    add_silence(random.uniform(1.0, 3.0))

# ==========================
# MAIN LOOP
# ==========================

BASE_SECONDS = 1 * 3600 + 20 * 60      # 1h 20m = 4800 seconds
EXTRA_SECONDS = random.randint(5 * 60, 15 * 60)
TARGET_TOTAL_SECONDS = BASE_SECONDS + EXTRA_SECONDS

print("Generating multi-round session...")

round_count = 0
while len(output_audio) / SR < TARGET_TOTAL_SECONDS:
    round_count += 1
    print(f"--- Round {round_count} ---")
    generate_round()
    current_seconds = len(output_audio) / SR
    print(f"Round {round_count}: total audio = {current_seconds:.1f}s")

# ==========================
# FINALIZE AUDIO
# ==========================

# normalize speech first
peak = np.max(np.abs(output_audio))
if peak > 0:
    output_audio = output_audio / peak * 0.9

# --- background noise ---
if bg_noise != "none":
    noise_path = os.path.join(BASE_DIR, "voices", "bg_noise", f"{bg_noise}.mp3")

    if os.path.exists(noise_path):
        print(f"[BG] Mixing background noise: {bg_noise}")
        output_audio = mix_background_noise(
            output_audio,
            noise_path,
            level=0.01
        )
    else:
        print(f"[WARN] Background noise file not found: {noise_path}")

# final safety normalize
peak = np.max(np.abs(output_audio))
if peak > 0:
    output_audio = output_audio / peak * 0.95

# ensure output directory exists
output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

sf.write(output_path, output_audio, SR)