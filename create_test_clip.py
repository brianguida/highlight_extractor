from moviepy.editor import VideoFileClip
import os

# Configuration
INPUT_VIDEO = "MJ_hits_go3_s1_5.mp4"
OUTPUT_VIDEO = "test_clip.mp4"
START_TIME = 160  # Start at first highlight
DURATION = 120    # 2 minutes

def main():
    print("Creating test clip...")
    
    # Load the video
    video = VideoFileClip(INPUT_VIDEO)
    
    # Create the clip
    test_clip = video.subclip(START_TIME, START_TIME + DURATION)
    
    # Save the clip
    print(f"Saving test clip to {OUTPUT_VIDEO}...")
    test_clip.write_videofile(
        OUTPUT_VIDEO,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    
    # Clean up
    test_clip.close()
    video.close()
    
    print("Done! Test clip created successfully.")
    print(f"Clip contains highlights at approximately: 1.7s, 30.3s, 57.4s")

if __name__ == "__main__":
    main() 