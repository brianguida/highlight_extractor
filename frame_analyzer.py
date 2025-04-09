import cv2
import json

#temp configs
VIDEO_PATH = "Grez_hits_pl3.mp4"
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

#convert timestamps to frame ranges
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
        print(f"Analyzing frame {frame_index} at time {frame_index / fps:.2f}s")
        #analyzing kill feed
        #training YOLOv8 model to analyze frames
    frame_index += 1
video.release()    
    