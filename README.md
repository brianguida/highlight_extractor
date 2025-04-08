#Highlight Extractor

This project is an AI-powered pipeline for automatically extracting potential highlight clips from **Marvel Rivals** gameplay footage. It uses audio-based spike detection to identify exciting moments (like shouting or hype reactions) and slices the video into short clips around those timestamps.

---

## ✅ Features

- 🔊 **Audio Spike Detection**: Uses RMS energy to find sudden increases in volume
- ✂️ **Automatic Clip Extraction**: Cuts 10-second clips around spike timestamps
- 📊 **Waveform Plotting**: Visualizes the audio signal with detected spikes
- 📂 **Organized Output**: Saves all clips into a `highlights/` folder

---

## ⚠️ Limitations

Currently, the extracted clips are mostly triggered by **voice activity**, such as player reactions or commentary. These are not guaranteed gameplay highlights yet — future improvements will include **frame-based analysis** to better detect in-game events like kills, ultimates, or flashy moments.

---

## 📁 File Overview

```
highlight_extractor/
├── audio_spike_detector.py     # Detects audio spikes and saves timestamps
├── clip_extractor.py           # Cuts highlight clips based on spike timestamps
├── test_video.mp4              # Input gameplay video (replace with your own)
├── spike_timestamps.json       # JSON file with timestamps of detected spikes
├── highlights/                 # Output folder for extracted highlight clips
├── venv/                       # Virtual environment (excluded from Git)
└── .gitignore
```

---

## 🧠 Built With

- Python 3.10
- [Librosa](https://librosa.org/) – audio analysis
- [MoviePy](https://zulko.github.io/moviepy/) – video slicing
- [Matplotlib](https://matplotlib.org/) – plotting
- FFmpeg

---

## 🚀 Next Steps

- 🖼 Add frame-based highlight detection (OpenCV or YOLOv8)
- 🫣 Add Whisper transcription + GPT commentary (optional)
- 🎮 Merge highlights into a single reel
- 🧪 Build a CLI or web UI for usability

---

## 👤 Author

**Brian Guida**  
GitHub: [@brianguida](https://github.com/brianguida)