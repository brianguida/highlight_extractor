#Detects spikes in audio for highlight detection
#imports
import librosa
import numpy as np
import moviepy.editor as mp
import json
import matplotlib.pyplot as plt
import whisper
import os
from typing import List, Tuple
import soundfile as sf
from scipy import signal

#Temporary config for testing
VIDEO_PATH = "test_clip.mp4"
OUTPUT_JSON = "spike_timestamps.json"
THRESHOLD = 0.03  # Lowered threshold to catch more combat sounds
WHISPER_MODEL = "base"
AUDIO_WINDOW = 1.5  # Reduced window to focus on the kill moment
MEANINGFUL_KEYWORDS = [
    # Combat sounds and reactions
    "kill", "shot", "fire", "hit", "dead", "down", "got", "nice", "yes",
    "let's go", "got em", "got him", "got them", "oh", "wow",
    
    # Game announcer lines
    "double ko", "triple ko", "quadra ko", "penta ko", "hexa ko",
    "that's three", "wow four in a row", "five that's amazing", 
    "six in a row way to go", "unbelievable", "you got them all",
    
    # Combat exclamations (common player reactions)
    "eliminated", "team wipe", "wiped", "destroyed", "deleted",
    "clipped", "clip that", "clip it", "holy", "insane"
]

#Extract audio from mp4 file
print("Loading video and extracting audio...")
video = mp.VideoFileClip(VIDEO_PATH)

# Try to boost game audio frequencies (typically higher frequencies)
def boost_game_audio(audio_array, sr):
    # Create a high-pass filter to boost higher frequencies
    nyquist = sr / 2
    cutoff = 1000  # 1kHz cutoff
    b, a = signal.butter(4, cutoff/nyquist, btype='high')
    filtered = signal.filtfilt(b, a, audio_array)
    # Mix original and filtered audio
    return 0.7 * audio_array + 0.3 * filtered

# Extract and process audio
audio_path = "temp_audio.wav"
video.audio.write_audiofile(audio_path, verbose=False, logger=None)

# Load the audio file and process it
y, sr = librosa.load(audio_path)
processed_audio = boost_game_audio(y, sr)

# Save processed audio
sf.write(audio_path, processed_audio, sr)

#Analyze extracted audio
print("Analyzing extracted audio...")
y, sr = librosa.load(audio_path)
frame_length = 2048
hop_length = 512

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

def is_kill_sound_pattern(rms_values, times, spike_index, sr, hop_length):
    """Check if the RMS pattern matches typical kill sound characteristics"""
    # Get a small window around the spike (kill sounds are short)
    window_size = int(0.2 * sr / hop_length)  # 0.2 second window
    start_idx = max(0, spike_index - window_size)
    end_idx = min(len(rms_values), spike_index + window_size)
    
    window = rms_values[start_idx:end_idx]
    
    if len(window) < 3:
        return False
    
    # Kill sounds typically have:
    # 1. Very sharp rise
    # 2. Short duration
    # 3. Quick falloff
    
    # Check for sharp rise
    rise_rate = (rms_values[spike_index] - rms_values[spike_index-1]) / (times[spike_index] - times[spike_index-1])
    
    # Check for quick falloff
    if spike_index + 1 < len(rms_values):
        falloff_rate = (rms_values[spike_index] - rms_values[spike_index+1]) / (times[spike_index+1] - times[spike_index])
    else:
        falloff_rate = 0
    
    # Pattern matching criteria
    is_sharp_rise = rise_rate > 0.2  # Higher threshold for combat sounds
    is_quick_falloff = falloff_rate > 0.15  # Should drop off quickly
    
    return is_sharp_rise and is_quick_falloff

#Logic to find spikes in audio
spikes = []
spike_indices = []
for i in range(1, len(rms)):
    # Look for any significant spike
    if rms[i] > THRESHOLD:
        # Check if it's a kill-like sound
        if is_kill_sound_pattern(rms, times, i, sr, hop_length):
            spikes.append(times[i])
            spike_indices.append(i)
            print(f"Found potential kill sound at {times[i]:.2f}s with RMS {rms[i]:.4f}")

print(f"Found {len(spikes)} potential kill sounds.")

# Now use Whisper only on these more likely spikes
print("Loading Whisper model (this may take a moment)...")
model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded successfully!")

def analyze_audio_segment(audio_path: str, start_time: float, end_time: float) -> Tuple[bool, str]:
    """Analyze audio segment using Whisper and check for meaningful content."""
    # Extract the segment
    y_segment, sr = librosa.load(audio_path, offset=start_time, duration=end_time-start_time)
    temp_segment_path = "temp_segment.wav"
    
    # Save audio using soundfile instead of librosa.output
    sf.write(temp_segment_path, y_segment, sr)
    
    # Transcribe with Whisper
    result = model.transcribe(temp_segment_path)
    text = result["text"].lower()
    
    # Debug: Print all transcriptions
    print(f"\nTime: {start_time:.2f}s - {end_time:.2f}s")
    print(f"Transcription: {text}")
    
    # Clean up temp file
    os.remove(temp_segment_path)
    
    # Check for meaningful keywords
    is_meaningful = any(keyword in text for keyword in MEANINGFUL_KEYWORDS)
    if is_meaningful:
        print("✓ Matched keywords!")
    else:
        print("✗ No keyword matches")
    
    return is_meaningful, text

# Analyze spikes and generate timeframes
print("Analyzing audio segments with Whisper...")
meaningful_timeframes = []
total_spikes = len(spikes)
for i, spike_time in enumerate(spikes, 1):
    print(f"Processing spike {i}/{total_spikes}...")
    start_time = max(0, spike_time - AUDIO_WINDOW/2)
    end_time = min(video.duration, spike_time + AUDIO_WINDOW/2)
    
    is_meaningful, transcription = analyze_audio_segment(audio_path, start_time, end_time)
    if is_meaningful:
        meaningful_timeframes.append({
            "start": start_time,
            "end": end_time,
            "transcription": transcription
        })

print(f"Found {len(meaningful_timeframes)} meaningful timeframes.")
print("Merging nearby timeframes...")

def merge_nearby_timeframes(timeframes, max_gap=10.0):
    """Merge timeframes that are within max_gap seconds of each other."""
    if not timeframes:
        return []
    
    # Sort timeframes by start time
    sorted_frames = sorted(timeframes, key=lambda x: x["start"])
    merged = []
    current = sorted_frames[0]
    
    for next_frame in sorted_frames[1:]:
        # If next frame starts within max_gap seconds of current frame's end
        if next_frame["start"] - current["end"] <= max_gap:
            # Merge the frames
            current["end"] = next_frame["end"]
            # Combine transcriptions
            current["transcription"] = current["transcription"] + " | " + next_frame["transcription"]
        else:
            merged.append(current)
            current = next_frame
    
    merged.append(current)
    return merged

# Merge timeframes that are close together
merged_timeframes = merge_nearby_timeframes(meaningful_timeframes, max_gap=10.0)

print(f"Merged into {len(merged_timeframes)} highlight clips.")
print(f"Saving to {OUTPUT_JSON}")

#Save merged timeframes
with open(OUTPUT_JSON, "w") as f:
    json.dump(merged_timeframes, f, indent=2)
    
#Visualize audio spikes and merged timeframes
plt.figure(figsize=(14, 5))
plt.plot(times, rms, label="RMS Energy")
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f"Threshold = {THRESHOLD}")

# Plot the merged timeframes in a different color
for timeframe in merged_timeframes:
    plt.axvspan(timeframe["start"], timeframe["end"], color='purple', alpha=0.2)

plt.title("Audio RMS with Merged Highlight Timeframes")
plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.legend()
plt.tight_layout()
plt.show()

#Clean up
os.remove(audio_path)