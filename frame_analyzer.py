import cv2
import json
from ultralytics import YOLO

#temp configs
VIDEO_PATH = "MJ_hits_go3_s1_5.mp4"
SPIKE_FILE = "spike_timestamps.json"
FRAME_WINDOW = 0.5 # of seconds around each spike

#loading spike timestamps
with open(SPIKE_FILE, "r") as f:
    spikes = json.load(f)
    
#get fps and frame count
video = cv2. VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

#load trained YOLO model
model = YOLO("best.pt")

#convert timestamps to frame ranges
highlight_timestamps = []
spike_frames = set()
for timestamp in spikes:
    center_frame = round(timestamp * fps)
    start_frame = max(0, center_frame - round(0.5 * fps))
    end_frame = center_frame + round(0.5 * fps)
    for frame_num in range(start_frame, end_frame + 1):
        spike_frames.add(frame_num)
frame_index = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    if frame_index in spike_frames:
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                print(f"[{frame_index}] {label} detected with confidence: {conf:.2f}")
                if label in {"kill_final", "kill_badges", "kill_confirmed", "kill_feed"} and conf > 0.5:
                    #save timestamp for clip_analyzer
                    timestamp = frame_index / fps
                    highlight_timestamps.append(timestamp)
    frame_index += 1
video.release()

#minimum gap between highlights = 5s
filtered_timestamps = []
min_gap = 5.0

for ts in sorted(set(highlight_timestamps)):
    if not filtered_timestamps or ts - filtered_timestamps[-1] >= min_gap:
        filtered_timestamps.append(ts)

#save highlight_timestamps
with open("highlight_timestamps.json", "w") as f:
    json.dump(sorted(set(filtered_timestamps)), f, indent=2)
    