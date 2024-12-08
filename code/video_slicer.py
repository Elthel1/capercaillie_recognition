import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from moviepy import VideoFileClip

def slice_video(input_path, output_folder, video_id, slice_duration):
    """Helper function to slice a single video"""
    video = VideoFileClip(input_path)
    duration = video.duration
    
    output_dir = os.path.join(output_folder, video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    for start in range(0, int(duration), slice_duration):
        end = min(start + slice_duration, duration)
        segment = video.subclipped(start, end)
        output_path = os.path.join(output_dir, f"{video_id}_segment_{start}_{end}.mp4")
        segment.write_videofile(output_path)
    
    video.close()

def slice_videos(input_folder, sliced_output_folder, slice_duration=10):
    """Slice videos into segments"""
    os.makedirs(sliced_output_folder, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    with ProcessPoolExecutor() as executor:
        futures = []
        for filename in video_files:
            video_id = os.path.splitext(filename)[0]
            input_path = os.path.join(input_folder, filename)
            futures.append(
                executor.submit(slice_video, input_path, sliced_output_folder, 
                              video_id, slice_duration)
            )
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

def create_folder_structure(base_dir, video_id, folder_num):
    """Create folder structure for frames and audio"""
    folder_path = os.path.join(base_dir, video_id, f"folder{folder_num}")
    frames_path = os.path.join(folder_path, "frames")
    audio_path = os.path.join(folder_path, "audio")
    
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(audio_path, exist_ok=True)
    
    return frames_path, audio_path

def extract_audio_segment(video_path, start_time, end_time, output_path):
    """Extract audio segment from specific time range"""
    video = VideoFileClip(video_path)
    if video.audio is not None:
        try:
            # Ensure we don't exceed video duration
            clip_duration = video.duration
            start_time = min(start_time, clip_duration - 1)
            end_time = min(end_time, clip_duration)
            
            # Extract audio 
            audio_segment = video.audio.subclipped(start_time, end_time)
            audio_segment.write_audiofile(output_path)
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {str(e)}")
    video.close()

def extract_frames_audio(input_folder, media_output_folder, frames_per_segment=10):
    """Extract frames and audio, organized in folders of 10 pairs"""
    frame_counter = 0  # Keep this global for continuous counting
    
    for video_id in os.listdir(input_folder):
        current_folder = 0  # Reset folder counter for each video_id
        video_dir = os.path.join(input_folder, video_id)
        if not os.path.isdir(video_dir):
            continue
            
        segments = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        # Create first folder structure
        frames_dir, audio_dir = create_folder_structure(
            media_output_folder, video_id, current_folder
        )
        
        for segment_file in segments:
            segment_path = os.path.join(video_dir, segment_file)
            segment_name = os.path.splitext(segment_file)[0]
            
            cap = cv2.VideoCapture(segment_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            
            frame_times = np.linspace(0, duration, frames_per_segment)
            for time in frame_times:
                # Create new folder when current one is full
                if frame_counter > 0 and frame_counter % 10 == 0:
                    current_folder += 1  # Increment folder number
                    frames_dir, audio_dir = create_folder_structure(
                        media_output_folder, video_id, current_folder
                    )
                
                # Extract frame and audio with modulo-based naming
                frame_pos = int(time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(
                        frames_dir, 
                        f"frame_{frame_counter % 10}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    
                    audio_start = max(0, time - 0.5)
                    audio_end = min(time + 0.5, duration)
                    audio_path = os.path.join(
                        audio_dir,
                        f"audio_{frame_counter % 10}.mp3"
                    )
                    extract_audio_segment(segment_path, audio_start, audio_end, audio_path)
                    
                    frame_counter += 1
            
            cap.release()

def process_videos_complete(input_folder, sliced_output_folder, media_output_folder, 
                          slice_duration=10, frames_per_segment=10):
    """Process videos end-to-end: slice and extract media"""
    slice_videos(input_folder, sliced_output_folder, slice_duration)
    extract_frames_audio(sliced_output_folder, media_output_folder, frames_per_segment)

if __name__ == "__main__":
    input_folder = "..\\media\\raw"
    sliced_folder = "..\\media\\sliced_mp4"
    frames_audio_folder = "..\\media\\frames_and_audio"
    
    print("Processing videos...")
    slice_videos(input_folder, sliced_folder, slice_duration=10)
    print("Extracting frames and audio...")
    extract_frames_audio(sliced_folder, frames_audio_folder)
    print("Processing complete!")