#Detects spikes in audio for highlight detection
#imports
import librosa
import numpy as np
import moviepy.editor as mp
import json
import matplotlib.pyplot as plt

#Temporary config for testing
VIDEO_PATH = "MJ_hits_go3_s1_5.mp4"
OUTPUT_JSON = "spike_timestamps.json"
THRESHOLD = 0.04

#Extract audio from mp4 file
print("Loading video and extracting audio")
video = mp.VideoFileClip(VIDEO_PATH)
audio_path = "temp_audio.wav"
video.audio.write_audiofile(audio_path, verbose = False, logger = None)

#Analyze extracted audio
print("Analyzing extracted audio")
y, sr = librosa.load(audio_path)
frame_length = 2048
hop_length = 512

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

#Logic to find spikes in audio
#rms needs to be above THRESHOLD
#rms needs to be larger than previous frame
spikes = []
for i in range(1, len(rms)):
    if rms[i] > THRESHOLD and rms[i] > rms[i-1] * 1.5:
        spikes.append(times[i])
        
print(f"found {len(spikes)} spikes.")
print(f"Saving to {OUTPUT_JSON}")

#Save spikes
with open(OUTPUT_JSON, "w") as f:
    json.dump(spikes, f, indent= 2)
    
#Visualize audio spikes
plt.figure(figsize=(14, 5))
plt.plot(times, rms, label="RMS Energy")
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f"Threshold = {THRESHOLD}")
for spikes in spikes:
    plt.axvline(x=spikes, color='g', linestyle=':', alpha=0.6)
plt.title("Audio RMS with Spike Detection")
plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.legend()
plt.tight_layout()
plt.show()