import os
import random
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from typing import List, Tuple

"""
This script changes the speed of the audio files in the footsteps_processed folder.
It creates a new folder called footsteps_speed_modified with the modified audio files.
"""

def get_audio_files(folder_path: str) -> List[str]:
    """Get all audio files from the specified folder."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg'}
    audio_files = []
    
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(str(file_path))
    
    return audio_files

def change_audio_speed(input_path: str, output_path: str, speed_factor: float):
    """Change the speed of an audio file using librosa."""
    try:
        # Load audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Change speed using time stretching
        # speed_factor > 1 = faster, speed_factor < 1 = slower
        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
        
        # Save the modified audio
        sf.write(output_path, y_stretched, sr)
        print(f"Processed: {os.path.basename(input_path)} -> {os.path.basename(output_path)} (speed: {speed_factor}x)")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_audio_folder(input_folder: str, output_folder: str = None):
    """Process all audio files in a folder with different speed modifications."""
    
    # Get all audio files
    audio_files = get_audio_files(input_folder)
    
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "speed_modified")
    
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Shuffle files randomly
    random.shuffle(audio_files)
    
    # Calculate group sizes
    total_files = len(audio_files)
    group_size = total_files // 4
    remainder = total_files % 4
    
    print(f"Found {total_files} audio files")
    print(f"Processing in groups of ~{group_size} files each")
    
    # Define speed factors and their corresponding suffixes
    speed_configs = [
        (0.75, "_slow075"),
        (0.875, "_slow0875"), 
        (1.0, "_original"),
        (1.25, "_fast125")
    ]
    
    # Process each group
    start_idx = 0
    for i, (speed_factor, suffix) in enumerate(speed_configs):
        # Calculate end index for this group
        current_group_size = group_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_group_size
        
        # Get files for this group
        group_files = audio_files[start_idx:end_idx]
        
        print(f"\nProcessing group {i+1}: {len(group_files)} files at {speed_factor}x speed")
        
        # Process each file in the group
        for file_path in group_files:
            # Create output filename
            file_name = Path(file_path).stem
            file_ext = Path(file_path).suffix
            output_filename = f"{file_name}{suffix}{file_ext}"
            output_path = os.path.join(output_folder, output_filename)
            
            # Change speed
            change_audio_speed(file_path, output_path, speed_factor)
        
        start_idx = end_idx
    
    print(f"\nAll files processed! Output saved to: {output_folder}")

def main():
    
    input_folder = "footsteps_processed"
    output_folder = "footsteps_speed_modified"
    
    # Validate input folder
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a directory")
        return
    
    # Process the folder
    process_audio_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
