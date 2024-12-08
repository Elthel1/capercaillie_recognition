# inference.py
import torch
from model import BirdDataset, EarlyFusionModel
from torch.utils.data import DataLoader
from training import load_frames_from_folder, load_audio_from_folder  # Import functions to load frames and audio

if __name__ == "__main__":
    # Load the trained model
    model = EarlyFusionModel().cuda()  # Move model to GPU
    model.load_state_dict(torch.load('model\\early_fusion_model.pth'))
    model.eval()  # Set to evaluation mode

    # Load your video frames and audio clips for inference
    frames_video1 = load_frames_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder0\\frames')  # List of frames from folder 1
    audio_clips_video1 = load_audio_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder0\\audio')  # Corresponding audio clips for video 1
    frames_video2 = load_frames_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder1\\frames')  # List of frames from folder 2
    audio_clips_video2 = load_audio_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder1\\audio')  # Corresponding audio clips for video 2

    # Concatenate frames and audio clips from both videos
    all_frames = frames_video1 + frames_video2
    all_audio_clips = audio_clips_video1 + audio_clips_video2
    
    
    dataset = BirdDataset(all_frames, all_audio_clips, [1]) # label isn't used
    dataloader = DataLoader(dataset)

    outputs = []
    
    for batch_frames, batch_audios, _batch_label in dataloader:
        batch_frames = batch_frames.cuda()
        batch_audios = batch_audios.cuda()
        
        outputs = model(batch_frames, batch_audios)

    # Process videos for inference

    print(f'Unique individuals: {outputs}')