import os
import cv2
import subprocess
import argparse
from pathlib import Path

def is_ffmpeg_installed():
    """Check if ffmpeg is installed and available in PATH"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def extract_frames_ffmpeg(video_path, output_dir, fps=None):
    """Extract frames using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    if fps:
        cmd = [
            'ffmpeg', 
            '-i', video_path,
            '-vf', f'fps={fps}',
            os.path.join(output_dir, '%06d.jpg')
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-start_number', '0',  # Start frame numbering from 0
            os.path.join(output_dir, '%06d.jpg')
        ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def extract_frames_opencv(video_path, output_dir, fps=None):
    """Extract frames using OpenCV"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1
    
    if fps:
        frame_interval = int(video_fps / fps)
        if frame_interval == 0:
            frame_interval = 1
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Save frames with 0-based indexing
            filename = os.path.join(output_dir, f'{saved_count:06d}.jpg')
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return True

def process_videos_folder(input_folder, output_folder, fps=None):
    """Process all videos in a folder"""
    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if ffmpeg is available
    use_ffmpeg = is_ffmpeg_installed()
    if use_ffmpeg:
        print("Using ffmpeg for frame extraction")
    else:
        print("ffmpeg not found, using OpenCV for frame extraction")
    
    # Get all video files in the input folder
    video_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        # Create output subfolder with the same name as the video file (without extension)
        subfolder_name = video_file.stem
        output_subfolder = output_path / subfolder_name
        
        print(f"Processing {video_file.name} -> {output_subfolder}")
        
        # Extract frames
        success = False
        if use_ffmpeg:
            success = extract_frames_ffmpeg(str(video_file), str(output_subfolder), fps)
        else:
            success = extract_frames_opencv(str(video_file), str(output_subfolder), fps)
        
        if success:
            # Count extracted frames
            if output_subfolder.exists():
                frame_count = len([f for f in output_subfolder.iterdir() if f.is_file()])
                print(f"  Extracted {frame_count} frames")
            else:
                print(f"  Extracted frames (count unknown)")
        else:
            print(f"  Failed to extract frames")

def process_single_video(video_path, output_folder, fps=None):
    """Process a single video file"""
    video_path = Path(video_path)
    
    if not video_path.is_file():
        print(f"Video file {video_path} does not exist")
        return False
        
    # Check if it's actually a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    if video_path.suffix.lower() not in video_extensions:
        print(f"File {video_path} is not a recognized video format")
        return False
    
    # Create output folder - use the output_folder directly for single files
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {video_path.name} -> {output_path}")
    
    # Check if ffmpeg is available
    use_ffmpeg = is_ffmpeg_installed()
    if use_ffmpeg:
        print("Using ffmpeg for frame extraction")
        success = extract_frames_ffmpeg(str(video_path), str(output_path), fps)
    else:
        print("ffmpeg not found, using OpenCV for frame extraction")
        success = extract_frames_opencv(str(video_path), str(output_path), fps)
    
    if success:
        # Count extracted frames
        if output_path.exists():
            frame_count = len([f for f in output_path.iterdir() if f.is_file()])
            print(f"  Extracted {frame_count} frames")
        else:
            print(f"  Extracted frames (count unknown)")
        return True
    else:
        print(f"  Failed to extract frames")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos in a folder')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing videos or a single video file')
    parser.add_argument('--output', '-o', help='Output folder for extracted frames (optional)')
    parser.add_argument('--fps', type=float, help='Target FPS for extracted frames (optional)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # If no output specified, create output folder with same name as input
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_frames"
        else:
            output_path = input_path.parent / f"{input_path.name}_frames"
    
    # Check if input is a single file or a folder
    if input_path.is_file():
        # Process single video file
        process_single_video(str(input_path), str(output_path), args.fps)
    else:
        # Process folder of videos
        if not input_path.exists():
            print(f"Input path {args.input} does not exist")
            return
        process_videos_folder(args.input, str(output_path), args.fps)

if __name__ == '__main__':
    main()