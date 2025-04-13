import os
import json
import shutil
from moviepy.editor import VideoFileClip
import tempfile

def archive_existing_highlights():
    """Move existing highlight clips to an archive folder"""
    archive_dir = "archived_highlights"
    
    # Create archive directory if it doesn't exist
    if not os.path.exists(archive_dir):
        print("Created directory:", archive_dir)
        os.makedirs(archive_dir)
    
    print("Archiving existing highlights...")
    # Move any existing highlight files to archive
    for file in os.listdir():
        if file.startswith("highlight_") and file.endswith(".mp4"):
            archive_path = os.path.join(archive_dir, file)
            shutil.move(file, archive_path)
            print(f"Archived: {file}")

def create_highlight_clip(video, start_time, end_time, output_filename):
    """Create a single highlight clip with error handling"""
    try:
        # Extract the subclip
        highlight_clip = video.subclip(start_time, end_time)
        
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = os.path.join(temp_dir, "temp_highlight.mp4")
            
            # Write to temporary file first
            highlight_clip.write_videofile(
                temp_output,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # If successful, move to final destination
            if os.path.exists(temp_output):
                shutil.move(temp_output, output_filename)
                print(f"Successfully created: {output_filename}")
            
        # Clean up
        highlight_clip.close()
        return True
        
    except Exception as e:
        print(f"Error creating clip {output_filename}: {str(e)}")
        return False

def main():
    print("Starting highlight clipping process...")
    
    # Archive existing highlights first
    archive_existing_highlights()
    
    try:
        # Load highlight timestamps
        print("Loading highlight timestamps...")
        with open("highlight_timestamps.json", "r") as f:
            timestamps = json.load(f)
        
        # Load video
        print("Loading video...")
        video = VideoFileClip("test_clip.mp4")
        
        print(f"Creating {len(timestamps)} highlight clips...")
        successful_clips = 0
        
        # Create highlight clips
        for i, highlight in enumerate(timestamps, 1):
            start_time = highlight["start"]
            end_time = highlight["end"]
            
            print(f"\nCreating clip {i}/{len(timestamps)} from {start_time:.2f}s to {end_time:.2f}s...")
            output_filename = f"highlight_{i:02d}.mp4"
            
            if create_highlight_clip(video, start_time, end_time, output_filename):
                successful_clips += 1
        
        # Clean up
        video.close()
        print(f"\nFinished! Successfully created {successful_clips} out of {len(timestamps)} highlight clips.")
        
    except Exception as e:
        print(f"Error during highlight creation: {str(e)}")
        return

if __name__ == "__main__":
    main() 