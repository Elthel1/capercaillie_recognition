# training.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BirdDataset, EarlyFusionModel  # Import model and dataset

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for frames, audio_clips in dataloader:
            frames = frames.cuda()  # Move frames to GPU
            audio_clips = audio_clips.cuda()  # Move audio clips to GPU
            
            optimizer.zero_grad()
            audio_tensor = torchaudio.transforms.Spectrogram()(audio_clips.unsqueeze(1)).cuda()  # Convert audio to spectrogram
            outputs = model(frames, audio_tensor)
            loss = criterion(outputs, labels)  # Define `labels` accordingly
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    # Load your video frames and audio clips here
    frames_video1 = [...]  # List of frames from video 1
    audio_clips_video1 = [...]  # Corresponding audio clips for video 1
    frames_video2 = [...]  # List of frames from video 2
    audio_clips_video2 = [...]  # Corresponding audio clips for video 2

    # Create dataset and dataloader
    dataset = BirdDataset(frames_video1 + frames_video2, audio_clips_video1 + audio_clips_video2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = EarlyFusionModel().cuda()  # Move model to GPU
    criterion = nn.MSELoss()  # Adjust based on your task
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), 'early_fusion_model.pth')