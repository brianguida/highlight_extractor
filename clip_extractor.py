#extracts clips based on spike_timestamps.json
#import
import json
import os
from moviepy.editor import VideoFileClip

#temporary config
VIDEO_PATH = "MJ_hits_go3_s1_5.mp4"
TIMESTAMPS_FILE = "highlight_timestamps.json"
OUTPUT_DIR = "highlights"
CLIP_LENGTH = 10

#sanity check for output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load spike_timestamps.json and video
with open(TIMESTAMPS_FILE, "r") as f:
    timestamps = json.load(f)
video = VideoFileClip(VIDEO_PATH)
half = CLIP_LENGTH / 2 #+- 5 seconds

#extract clips
for i, ts in enumerate(timestamps):
    start = max(0, ts - half)
    end = min(video.duration, ts + half)
    
    subclip = video.subclip(start, end)
    output_path = os.path.join(OUTPUT_DIR, f"highlight_{i+1:02}.mp4")
    
    print(f"Extracting and saving highlight {i+1} ({start:.2f}s to {end:.2f}s)...")
    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac",verbose=False, logger=None)
    
print("All highlights extracted and saved to /highlights")
    