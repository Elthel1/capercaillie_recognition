import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models


# Custom Dataset to handle frames and audio
import torchvision.transforms as transforms
from torchaudio.transforms import MelSpectrogram

class BirdDataset(Dataset):
    def __init__(self, frames, audio_clips, batch_size=20):
        # Data reshaping into batches
        self.frames = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
        self.audio_clips = [audio_clips[i:i + batch_size] for i in range(0, len(audio_clips), batch_size)]
        
        # Visual transforms
        self.visual_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # Audio transforms
        self.audio_transform = transforms.Compose([
            MelSpectrogram(
                sample_rate=44100,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            ),
            transforms.Lambda(lambda x: x.unsqueeze(0))  # Add channel dim
        ])

    def __getitem__(self, idx):
        batch_frames = self.frames[idx]
        batch_audio = self.audio_clips[idx]
        
        # Apply transforms
        batch_frames = torch.stack([self.visual_transform(frame) for frame in batch_frames])
        batch_audio = torch.stack([self.audio_transform(audio) for audio in batch_audio])
        
        return batch_frames, batch_audio
    

class EarlyFusionModel(nn.Module):
    def __init__(self, num_paralle_streams=20):
        super(EarlyFusionModel, self).__init__()
        self.num_paralle_streams = num_paralle_streams
        
        # Visual feature extractors (ResNet)
        self.visual_cnns = nn.ModuleList([models.resnet50(pretrained=True) for _ in range(num_paralle_streams)])
        for visual_cnn in self.visual_cnns:
            visual_cnn.fc = nn.Identity()  # Remove the classification layer
        
        # Audio feature extractors (simple CNN)
        self.audio_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(3, 3)),
                nn.ReLU()
            ) for _ in range(num_paralle_streams)
        ])

        # Fully connected layers for audio features
        self.audio_fcs = nn.ModuleList([nn.Linear(32 * 6 * 6, 128) for _ in range(num_paralle_streams)])  # Adjust based on your input dimensions
        
        # Connect each image with it's corresponding audio
        self.image_audio_fcs = nn.ModuleList([nn.Linear(2048 + 128, 256) for _ in range(num_paralle_streams)])  # Adjust based on your input dimensions

        # Fully connected layer for combining features
        self.fc = nn.Linear(256 * num_paralle_streams, 1)  # Combining visual and audio features

    def forward(self, frames, audios):
        # Process visual inputs
        visual_features = [visual_cnn(frame) for visual_cnn, frame in zip(self.visual_cnns, frames)]
        
        # Process audio inputs
        audio_features = []
        for audio_conv, audio_fc, audio in zip(self.audio_convs, self.audio_fcs, audios):
            audio = audio_conv(audio)
            audio = audio.view(audio.size(0), -1)  # Flatten
            audio_features.append(audio_fc(audio))

        # Combine visual and audio features
        combined_features = []
        for visual_feature, audio_feature, image_audio_fc in zip(visual_features, audio_features, self.image_audio_fcs):
            combined = torch.cat((visual_feature, audio_feature), dim=1)
            combined_features.append(image_audio_fc(combined))
        
        # Combine parallel streams
        combined_features = torch.cat(combined_features, dim=1)
        output = self.fc(combined_features)
        
        return output