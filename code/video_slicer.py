import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
import os

def slice_video(video_path, sliced_output_dir, video_id, slice_duration=10):
    """Slice a video into segments"""
    try:
        # Open video with moviepy to preserve audio
        video = VideoFileClip(video_path)
        
        # Create video-specific directory and normalize path
        video_segments_dir = os.path.normpath(os.path.join(sliced_output_dir, video_id))
        os.makedirs(video_segments_dir, exist_ok=True)
        
        segment_paths = []
        duration = video.duration
        
        for i, start_time in enumerate(range(0, int(duration), slice_duration)):
            try:
                end_time = min(start_time + slice_duration, duration)
                segment = video.subclipped(start_time, end_time)
                segment_name = f"{video_id}_segment_{i:03d}" 
                output_path = os.path.normpath(os.path.join(video_segments_dir, f"{segment_name}.mp4"))

                # Use the default temp file handling
                segment.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    remove_temp=True,
                    audio=True,
                    fps=24,
                    ffmpeg_params=[
                        "-movflags", "faststart",
                        "-strict", "-2"
                    ]
                )
                
                segment_paths.append((segment_name, output_path))
                print(f"Sliced: {segment_name}")
                
                segment.close()
                
            except Exception as e:
                print(f"Error processing segment {i}: {str(e)}")
                if 'segment' in locals():
                    segment.close()
                
    finally:
        video.close()
        
    return segment_paths
                

def extract_media(segment_path, media_output_dir, video_id, segment_name):
    """Extract frames and audio from a video segment"""
    try:
        cap = cv2.VideoCapture(segment_path)
        
        # Create directories
        video_dir = os.path.join(media_output_dir, video_id)
        segment_dir = os.path.join(video_dir, segment_name)
        frames_dir = os.path.join(segment_dir, 'frames')
        audio_dir = os.path.join(segment_dir, 'audio')
        
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(segment_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, 10, dtype=int)
        
        # Extract frames
        for frame_idx, frame_pos in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        cap.release()
        
        # Extract audio if needed using moviepy
        video = VideoFileClip(segment_path)
        if video.audio is not None:
            try:
                audio_path = os.path.join(audio_dir, f"{segment_name}.wav")
                video.audio.write_audiofile(audio_path)
            except Exception as e:
                print(f"Warning: Could not extract audio for {segment_name}: {str(e)}")
        video.close()
        
        print(f"Extracted media from: {segment_name}")
        
    except Exception as e:
        print(f"Error processing {segment_name}: {str(e)}")

def process_videos_in_folder(input_folder, sliced_output_folder, media_output_folder, slice_duration=10):
    """Process all videos in the input folder"""
    os.makedirs(sliced_output_folder, exist_ok=True)
    os.makedirs(media_output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    with ProcessPoolExecutor() as executor:
        for filename in video_files:
            video_id = os.path.splitext(filename)[0]
            input_path = os.path.join(input_folder, filename)
            
            # First slice the video
            segment_paths = slice_video(input_path, sliced_output_folder, video_id, slice_duration)
            
            # Then extract media from each segment
            for segment_name, segment_path in segment_paths:
                executor.submit(extract_media, segment_path, media_output_folder, 
                              video_id, segment_name)

if __name__ == "__main__":
    input_folder_path = "..\\media\\raw"
    sliced_output_path = "..\\media\\sliced_mp4"
    media_output_path = "..\\media\\frames_and_audio"
    
    process_videos_in_folder(input_folder_path, sliced_output_path, media_output_path)