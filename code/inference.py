# inference.py
import torch
from model import EarlyFusionModel, BirdDataset, count_unique_individuals

if __name__ == "__main__":
    # Load the trained model
    model = EarlyFusionModel().cuda()  # Move model to GPU
    model.load_state_dict(torch.load('early_fusion_model.pth'))
    model.eval()  # Set to evaluation mode

    # Load your video frames and audio clips for inference
    frames_video1 = [...]  # List of frames from video 1
    audio_clips_video1 = [...]  # Corresponding audio clips for video 1
    frames_video2 = [...]  # List of frames from video 2
    audio_clips_video2 = [...]  # Corresponding audio clips for video 2

    # Process videos for inference
    embeddings_video1 = []
    for frame, audio in zip(frames_video1, audio_clips_video1):
        with torch.no_grad():
            audio_tensor = torchaudio.transforms.Spectrogram()(audio.unsqueeze(0)).cuda()  # Add batch dimension and move to GPU
            output = model(frame.unsqueeze(0).cuda(), audio_tensor)  # Add batch dimension for frame and move to GPU
            embeddings_video1.append(output)

    embeddings_video2 = []
    for frame, audio in zip(frames_video2, audio_clips_video2):
        with torch.no_grad():
            audio_tensor = torchaudio.transforms.Spectrogram()(audio.unsqueeze(0)).cuda()  # Add batch dimension and move to GPU
            output = model(frame.unsqueeze(0).cuda(), audio_tensor)
            embeddings_video2.append(output)

    # Count unique individuals using combined embeddings
    unique_count_video1 = count_unique_individuals(embeddings_video1)
    unique_count_video2 = count_unique_individuals(embeddings_video2)

    # Count same individuals across videos
    combined_embeddings = embeddings_video1 + embeddings_video2
    same_individuals = count_unique_individuals(combined_embeddings)

    print(f'Video 1 unique individuals: {unique_count_video1}')
    print(f'Video 2 unique individuals: {unique_count_video2}')
    print(f'Same individuals across videos: {same_individuals}')