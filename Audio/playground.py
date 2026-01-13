import time
import random
import os
import numpy as np
import librosa
import re
import soundfile as sf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
out_dir = os.path.join(OUTPUT_ROOT, "fan")

print(out_dir)

# List all files in out_dir
files = os.listdir(out_dir)

# Print each file name
for file in files:
    print(file)

# Extract numbers from filenames and find the highest
numbers = []
for file in files:
    # Assuming filenames contain numbers like fan_1.wav
    match = re.search(r'\d+', file)
    if match:
        numbers.append(int(match.group()))

if numbers:
    highest_number = max(numbers)
    print(f"Highest number: {highest_number}")
else:
    print("No numbers found in filenames")
