import cv2
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

#temp configs
VIDEO_PATH = "test_clip.mp4"
SPIKE_FILE = "spike_timestamps.json"
CONFIDENCE_THRESHOLD = 0.3
HIGHLIGHT_LABELS = {"kill_feed", "kill_badges"}
MIN_HIGHLIGHT_DURATION = 1.0  # Minimum duration in seconds
MAX_HIGHLIGHT_DURATION = 8.0  # Maximum duration for a single highlight
COOLDOWN_FRAMES = 15  # About 0.5 seconds at 30fps
MAX_GAP_SECONDS = 3.0  # Maximum gap when merging highlights
PRE_KILL_PADDING = 0.5  # Seconds to include before kill
POST_KILL_PADDING = 1.0  # Seconds to include after kill

#loading timeframe data
print("Loading spike timeframes...")
with open(SPIKE_FILE, "r") as f:
    timeframes = json.load(f)

# Merge overlapping timeframes first
def merge_overlapping_timeframes(timeframes, max_gap=MAX_GAP_SECONDS):
    if not timeframes:
        return []
    
    # Sort by start time
    sorted_frames = sorted(timeframes, key=lambda x: x["start"])
    merged = []
    current = sorted_frames[0]
    
    for next_frame in sorted_frames[1:]:
        # Only merge if gap is small enough AND total duration won't exceed max
        if (next_frame["start"] - current["end"] <= max_gap and 
            next_frame["end"] - current["start"] <= MAX_HIGHLIGHT_DURATION):
            # Merge overlapping frames
            current["end"] = max(current["end"], next_frame["end"])
        else:
            merged.append(current)
            current = next_frame
    
    merged.append(current)
    return merged

# Merge overlapping timeframes
timeframes = merge_overlapping_timeframes(timeframes)
print(f"Merged into {len(timeframes)} non-overlapping timeframes")
    
#get fps and frame count
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

#load trained YOLO model
print("Loading YOLO model...")
model = YOLO("best.pt")
print("YOLO model loaded successfully!")

#convert timeframes to frame ranges
highlight_timestamps = []
frames_to_analyze = set()

print(f"Processing {len(timeframes)} spike timeframes...")
for timeframe in timeframes:
    start_frame = max(0, round(timeframe["start"] * fps))
    end_frame = min(frame_count, round(timeframe["end"] * fps))
    
    # Add all frames in this timeframe to our set
    for frame_num in range(start_frame, end_frame + 1):
        frames_to_analyze.add(frame_num)

print(f"Analyzing {len(frames_to_analyze)} frames for highlights...")
frame_index = 0
highlight_found = False
current_highlight_start = None
frames_since_last_detection = 0
current_highlight_duration = 0
kill_detected_at = None  # Track when we first see a kill

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
        
    if frame_index in frames_to_analyze:
        results = model(frame, verbose=False)
        detection_found = False
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                
                # If we detect a kill with our criteria
                if label in HIGHLIGHT_LABELS and conf > CONFIDENCE_THRESHOLD:
                    timestamp = frame_index / fps
                    print(f"[{frame_index}] {label} detected with confidence: {conf:.2f}")
                    detection_found = True
                    
                    # If this is the start of a new highlight
                    if not highlight_found:
                        # Start slightly before the kill
                        current_highlight_start = max(0, timestamp - PRE_KILL_PADDING)
                        highlight_found = True
                        frames_since_last_detection = 0
                        current_highlight_duration = 0
                        kill_detected_at = timestamp
                    break
            
            if detection_found:
                frames_since_last_detection = 0
                break
        
        if highlight_found:
            current_highlight_duration = (frame_index / fps) - current_highlight_start
            
            # End highlight if it's too long
            if current_highlight_duration >= MAX_HIGHLIGHT_DURATION:
                highlight_timestamps.append({
                    "start": current_highlight_start,
                    "end": min(duration, kill_detected_at + POST_KILL_PADDING)
                })
                highlight_found = False
                current_highlight_start = None
                frames_since_last_detection = 0
                current_highlight_duration = 0
                kill_detected_at = None
                continue
                
            if detection_found:
                frames_since_last_detection = 0
                kill_detected_at = frame_index / fps  # Update last kill time
            else:
                frames_since_last_detection += 1
                
            # End highlight if we haven't seen a detection for a while
            if frames_since_last_detection >= COOLDOWN_FRAMES:
                if current_highlight_duration >= MIN_HIGHLIGHT_DURATION:
                    highlight_timestamps.append({
                        "start": current_highlight_start,
                        "end": min(duration, kill_detected_at + POST_KILL_PADDING)
                    })
                highlight_found = False
                current_highlight_start = None
                frames_since_last_detection = 0
                current_highlight_duration = 0
                kill_detected_at = None
                
    frame_index += 1

# Handle any ongoing highlight at the end of video
if highlight_found and current_highlight_start is not None and kill_detected_at is not None:
    end_timestamp = min(duration, kill_detected_at + POST_KILL_PADDING)
    duration = end_timestamp - current_highlight_start
    if MIN_HIGHLIGHT_DURATION <= duration <= MAX_HIGHLIGHT_DURATION:
        highlight_timestamps.append({
            "start": current_highlight_start,
            "end": end_timestamp
        })

video.release()

# Merge any overlapping highlights with stricter gap
highlight_timestamps = merge_overlapping_timeframes(highlight_timestamps, max_gap=MAX_GAP_SECONDS)

print(f"Found {len(highlight_timestamps)} highlights!")
print("Saving highlight timestamps...")

#save highlight_timestamps
with open("highlight_timestamps.json", "w") as f:
    json.dump(highlight_timestamps, f, indent=2)

#Visualize the timeframes and detected highlights
plt.figure(figsize=(14, 5))
plt.title("Spike Timeframes and Detected Highlights")
plt.xlabel("Time (s)")
plt.ylabel("Detection")

#Plot spike timeframes
for i, timeframe in enumerate(timeframes):
    plt.axvspan(timeframe["start"], timeframe["end"], color='blue', alpha=0.2, label='Spike Timeframe' if i == 0 else None)

#Plot detected highlights
for highlight in highlight_timestamps:
    plt.axvspan(highlight["start"], highlight["end"], color='green', alpha=0.3, label='Highlight' if highlight == highlight_timestamps[0] else None)

plt.legend()
plt.tight_layout()
plt.show()
    