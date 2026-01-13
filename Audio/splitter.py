import os
import librosa
import soundfile as sf

# =====================
# PATHS
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_INPUT_DIR = os.path.join(BASE_DIR, "raw_input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_clips")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# AUDIO SETTINGS
# =====================
SR = 22050

# Aggressive voice splitting
TOP_DB = 20              # very sensitive silence detection
MIN_CLIP_DURATION = 0.25 # seconds
MERGE_GAP = 0.1          # split if silence >= 100ms

SUPPORTED_EXTENSIONS = (".mp3", ".wav")

# =====================
# PROCESS FILES
# =====================
print("Scanning raw_input folder...")

files = [
    f for f in os.listdir(RAW_INPUT_DIR)
    if f.lower().endswith(SUPPORTED_EXTENSIONS)
]

if not files:
    print("No audio files found in raw_input/")
    exit()

clip_index = 0

for filename in files:
    input_path = os.path.join(RAW_INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    print(f"\nProcessing: {filename}")

    audio, _ = librosa.load(input_path, sr=SR, mono=True)

    # =====================
    # VOICE ACTIVITY DETECTION
    # =====================
    intervals = librosa.effects.split(
        audio,
        top_db=TOP_DB
    )

    # =====================
    # MERGE CLOSE SEGMENTS
    # =====================
    merged = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue

        prev_start, prev_end = merged[-1]
        gap = (start - prev_end) / SR

        if gap <= MERGE_GAP:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    print(f"Detected {len(merged)} voice chunks")

    # =====================
    # EXPORT CLIPS
    # =====================
    for start, end in merged:
        duration = (end - start) / SR
        if duration < MIN_CLIP_DURATION:
            continue

        clip = audio[start:end]

        output_name = f"{base_name}_clip_{clip_index:04d}.mp3"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        sf.write(output_path, clip, SR, format="MP3")
        clip_index += 1

print(f"\n✅ Done! Exported {clip_index} clips to output_clips/")