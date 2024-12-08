# training.py
import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio
import cv2
from torch.utils.data import DataLoader
from model import BirdDataset, EarlyFusionModel
import os

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch_frames, batch_audios, batch_label in dataloader:
            batch_frames = batch_frames.cuda()
            batch_audios = batch_audios.cuda()
            batch_label = batch_label.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_frames, batch_audios)
            
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def load_frames_from_folder(folder_path):
    frames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(folder_path, filename)
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            # Convert numpy array to torch tensor and adjust dimensions
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
    return frames

def load_audio_from_folder(folder_path):
    """Load audio files with explicit backend"""
    audio_clips = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(folder_path, filename)
            try:
                waveform, sample_rate = torchaudio.load(
                    audio_path, 
                    format="mp3",
                    channels_first=True
                )
                # Convert to single channel if necessary
                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                audio_clips.append(waveform)
            except RuntimeError as e:
                print(f"Error loading {filename}: {e}")
                continue
    return audio_clips

if __name__ == "__main__":
    # Load your video frames and audio clips here
    frames_video1 = load_frames_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder0\\frames')  # List of frames from folder 1
    audio_clips_video1 = load_audio_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder0\\audio')  # Corresponding audio clips for video 1
    frames_video2 = load_frames_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder1\\frames')  # List of frames from folder 2
    audio_clips_video2 = load_audio_from_folder('..\\media\\frames_and_audio\\single_bird1\\folder1\\audio')  # Corresponding audio clips for video 2

    # Concatenate frames and audio clips from both videos
    all_frames = frames_video1 + frames_video2
    all_audio_clips = audio_clips_video1 + audio_clips_video2

    # Create dataset and dataloader
    dataset = BirdDataset(all_frames, all_audio_clips, [1])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = EarlyFusionModel().cuda()  # Move model to GPU
    criterion = nn.MSELoss()  # Adjust based on your task
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("here")
    print(len(dataloader))
    print("audio_clips_video1", len(audio_clips_video1))

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), 'model\\early_fusion_model.pth')