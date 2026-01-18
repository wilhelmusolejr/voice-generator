import time
import random
import os
import numpy as np
import librosa
import soundfile as sf
import re

# ==========================
# BASE CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_AI_DIR = os.path.join(BASE_DIR, "voices_ai")
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

SR = 16000

# ==========================
# ROUND LOGIC FOR VOICES_AI
# ==========================

ROUND_DURATION = 150  # 2 minutes 30 seconds
FOLDERS = ["greetings", "enemy_info", "random", "round_result", "round_start", "strategy"]

# Main loop sequence (excluding greetings and round_result which are handled specially)
MAIN_LOOP_SEQUENCE = ["round_start", "strategy", "random", "enemy_info"]

# ==========================
# CORE FUNCTIONS
# ==========================

def load_audio_files_from_folder(folder_name):
    """Load all audio files from a specific folder in voices_ai."""
    folder_path = os.path.join(VOICES_AI_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        return []
    
    audio_files = []
    for f in os.listdir(folder_path):
        if f.endswith((".mp3", ".wav", ".ogg", ".flac")):
            audio_files.append(os.path.join(folder_path, f))
    
    return audio_files

def load_audio(file_path):
    """Load audio from file."""
    try:
        audio, _ = librosa.load(file_path, sr=SR)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_audio_duration(audio):
    """Get duration of audio in seconds."""
    if audio is None:
        return 0
    return len(audio) / SR

def add_silence(seconds, state):
    """Add silence to the audio state."""
    if seconds > 0:
        silence = np.zeros(int(seconds * SR))
        state["audio"] = np.concatenate([state["audio"], silence])

def play_random_clip_from(folder_name, state):
    """Load and add a random audio clip from a folder."""
    audio_files = load_audio_files_from_folder(folder_name)
    
    if not audio_files:
        print(f"No audio files found in {folder_name}")
        return 0
    
    selected_file = random.choice(audio_files)
    audio = load_audio(selected_file)
    
    if audio is not None:
        state["audio"] = np.concatenate([state["audio"], audio])
        duration = get_audio_duration(audio)
        print(f"Added {folder_name}: {os.path.basename(selected_file)} ({duration:.2f}s)")
        return duration
    
    return 0

def add_random_pause(state):
    """Add a random pause between 5-10 seconds."""
    pause = random.uniform(5, 10)
    add_silence(pause, state)
    return pause

# ==========================
# ROUND GENERATION
# ==========================

def generate_round(round_number, state):
    """Generate a single round (2:30 = 150 seconds)."""
    state["audio"] = np.array([], dtype=np.float32)
    round_start_time = time.time()
    elapsed_time = 0
    
    print(f"\n=== ROUND {round_number} ===")
    
    # Step 1: Decide if greeting should be included
    include_greeting = False
    if round_number == 1:
        include_greeting = True
    else:
        include_greeting = random.random() < 0.5  # 50% chance
    
    if include_greeting:
        elapsed_time += play_random_clip_from("greetings", state)
        elapsed_time += add_random_pause(state)
        print(f"Greeting included. Elapsed: {elapsed_time:.2f}s")
    
    # Step 2: Loop through main sequence (round_start, strategy, random, enemy_info)
    sequence_index = 0
    while elapsed_time < (ROUND_DURATION - 15):  # Leave 15 seconds for round_result
        folder = MAIN_LOOP_SEQUENCE[sequence_index % len(MAIN_LOOP_SEQUENCE)]
        
        elapsed_time += play_random_clip_from(folder, state)
        
        # Check if we're approaching the end of the round
        if elapsed_time >= (ROUND_DURATION - 15):
            break
        
        # Add pause between audios
        pause_duration = add_random_pause(state)
        elapsed_time += pause_duration
        print(f"Pause: {pause_duration:.2f}s. Elapsed: {elapsed_time:.2f}s")
        
        sequence_index += 1
    
    # Step 3: When round is about to finish, add round_result audio
    remaining_time = ROUND_DURATION - elapsed_time
    print(f"Remaining time for round_result: {remaining_time:.2f}s")
    
    elapsed_time += play_random_clip_from("round_result", state)
    
    # Fill remaining time with silence or small pauses
    total_duration = len(state["audio"]) / SR
    if total_duration < ROUND_DURATION:
        silence_needed = ROUND_DURATION - total_duration
        add_silence(silence_needed, state)
        print(f"Added silence: {silence_needed:.2f}s")
    
    final_duration = len(state["audio"]) / SR
    print(f"Round {round_number} Final Duration: {final_duration:.2f}s")
    
    return state["audio"]

# ==========================
# AUDIO JOB
# ==========================

def generate_audio_job(num_rounds=5, output_name="voices_ai_output"):
    """Generate audio with multiple rounds."""
    state = {
        "audio": np.array([], dtype=np.float32),
    }
    
    print(f"[JOB START] Generating {num_rounds} rounds")
    
    for round_num in range(1, num_rounds + 1):
        round_audio = generate_round(round_num, state)
        
        if len(state["audio"]) > 0:
            # Concatenate round audio
            state["audio"] = np.concatenate([state["audio"], round_audio])
    
    audio = state["audio"]
    
    # Normalize audio
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    
    # Save output
    out_dir = OUTPUT_ROOT
    os.makedirs(out_dir, exist_ok=True)
    
    # Find next file number
    file_number = 1
    if os.path.exists(out_dir):
        files = os.listdir(out_dir)
        numbers = []
        for file in files:
            match = re.search(r'\d+', file)
            if match:
                numbers.append(int(match.group()))
        
        if numbers:
            file_number = max(numbers) + 1
    
    out_path = os.path.join(out_dir, f"{output_name}_{file_number}.wav")
    sf.write(out_path, audio, SR)
    
    total_duration = len(audio) / SR
    print(f"\n[JOB DONE] Saved to {out_path}")
    print(f"Total duration: {total_duration / 60:.2f} minutes ({total_duration:.2f}s)")

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    # Generate audio with 5 rounds (5 * 2:30 = 12:30 minutes)
    generate_audio_job(num_rounds=5, output_name="voices_ai_output")
