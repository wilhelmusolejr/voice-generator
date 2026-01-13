import os
import random
import numpy as np
import librosa
import soundfile as sf

# ==========================
# CONFIG
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "output_clips")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 22050

TARGET_MINUTES = random.uniform(5, 10)
TARGET_SECONDS = TARGET_MINUTES * 60

# Loudness target (shared vibe)
TARGET_RMS = 0.035

# White noise level (very subtle)
NOISE_LEVEL = 0.0020

# Pause behavior
SHORT_PAUSE = (0.2, 0.5)
NORMAL_PAUSE = (0.6, 1.8)
LONG_PAUSE = (2.0, 5.0)

FADE_MS = 15  # micro fade for clip edges

# ==========================
# LOAD CLIPS
# ==========================
files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".wav", ".mp3"))
]

if not files:
    raise RuntimeError("No clips found in output_clips")

print(f"Loaded {len(files)} clips")
print(f"Target duration: {TARGET_MINUTES:.2f} minutes")

# ==========================
# AUDIO HELPERS
# ==========================
def rms_normalize(audio, target_rms):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio

def apply_fade(audio, fade_ms):
    fade_len = int(SR * fade_ms / 1000)
    if len(audio) <= fade_len * 2:
        return audio

    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)

    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio

def load_clip(filename):
    audio, _ = librosa.load(
        os.path.join(INPUT_DIR, filename),
        sr=SR,
        mono=True
    )

    audio = rms_normalize(audio, TARGET_RMS)
    audio = apply_fade(audio, FADE_MS)

    return audio

def noise(seconds):
    return np.random.normal(
        0, NOISE_LEVEL, int(seconds * SR)
    ).astype(np.float32)

# ==========================
# GENERATION
# ==========================
output_audio = np.array([], dtype=np.float32)
last_clip = None
total_seconds = 0.0

while total_seconds < TARGET_SECONDS:
    clip_name = random.choice(files)
    if clip_name == last_clip and len(files) > 1:
        clip_name = random.choice(files)

    clip = load_clip(clip_name)
    output_audio = np.concatenate([output_audio, clip])
    total_seconds += len(clip) / SR
    last_clip = clip_name

    # pause selection
    r = random.random()
    if r < 0.6:
        pause = random.uniform(*SHORT_PAUSE)
    elif r < 0.9:
        pause = random.uniform(*NORMAL_PAUSE)
    else:
        pause = random.uniform(*LONG_PAUSE)

    output_audio = np.concatenate([output_audio, noise(pause)])
    total_seconds += pause

# ==========================
# FINALIZE
# ==========================
# Add continuous noise bed (same vibe everywhere)
bed = np.random.normal(
    0, NOISE_LEVEL, len(output_audio)
).astype(np.float32)

final_audio = output_audio + bed

# Safety normalize
peak = np.max(np.abs(final_audio))
if peak > 0:
    final_audio = final_audio / peak * 0.95

output_file = os.path.join(
    OUTPUT_DIR,
    f"session_{int(TARGET_SECONDS)}s.wav"
)

sf.write(output_file, final_audio, SR)

print("✅ Generated:", output_file)
print(f"Final duration: {total_seconds/60:.2f} minutes")
