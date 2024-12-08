import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models


# Custom Dataset to handle frames and audio
import torchvision.transforms as transforms
from torchaudio.transforms import MelSpectrogram

class BirdDataset(Dataset):
    def __init__(self, frames, audio_clips, labels, batch_size=20):
        # Split data into batches of size batch_size
        frames_holder = []
        audio_holder = []
        
        for i in range(0, len(frames), batch_size):
            frames_holder.append(frames[i:i + batch_size])
            audio_holder.append(audio_clips[i:i + batch_size])
        
        self.frames = frames_holder
        self.audio_clips = audio_holder
        print("frames in dataset", len(self.frames))
        
        # Only one label per batch
        self.labels = labels
        
        # Update transform for already tensorized images
        self.visual_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Audio transforms
        self.audio_transform = MelSpectrogram(
            sample_rate=44100,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
            pad=1  # Add padding
        )

    def __getitem__(self, idx):
        batch_frames = self.frames[idx]
        batch_audios = self.audio_clips[idx]
        batch_label = self.labels[idx]
        
        # Apply transforms
        batch_frames = torch.stack([self.visual_transform(frame) for frame in batch_frames])
        
        # Transform audio and handle different lengths
        transformed_audios = [self.audio_transform(audio) for audio in batch_audios]
        
        # Find max length
        max_length = max(audio.size(-1) for audio in transformed_audios)
        
        # Pad all spectrograms to max length
        padded_audios = []
        for spec in transformed_audios:
            pad_amount = max_length - spec.size(-1)
            if pad_amount > 0:
                padded_spec = torch.nn.functional.pad(spec, (0, pad_amount))
                padded_audios.append(padded_spec)
            else:
                padded_audios.append(spec)
        
        batch_audios = torch.stack(padded_audios)
        batch_label = torch.tensor(float(batch_label), dtype=torch.float)
        
        return batch_frames, batch_audios, batch_label
    
    def __len__(self):
        return len(self.labels)
    

class EarlyFusionModel(nn.Module):
    def __init__(self, num_paralle_streams=20):
        super(EarlyFusionModel, self).__init__()
        self.num_paralle_streams = num_paralle_streams
        
        # Visual feature extractors (ResNet)
        self.visual_cnns = nn.ModuleList([models.resnet50(weights='IMAGENET1K_V1') for _ in range(num_paralle_streams)])
        for visual_cnn in self.visual_cnns:
            visual_cnn.fc = nn.Identity()  # Remove the classification layer
        
        # Audio feature extractors (simple CNN)
        self.audio_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Add padding
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Add padding
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # Add global pooling
                nn.Flatten()
            ) for _ in range(num_paralle_streams)
        ])

        # Connect each image with it's corresponding audio
        self.image_audio_fcs = nn.ModuleList([nn.Linear(2048 + 32, 256) for _ in range(num_paralle_streams)])  # Adjust based on your input dimensions

        # Fully connected layer for combining features
        self.fc = nn.Linear(256 * num_paralle_streams, 1)  # Combining visual and audio features

    def forward(self, frames, audios):
        print("frames", len(frames))
        print("audios", len(audios))
        # Process visual inputs
        visual_features = []
        for visual_cnn, frame in zip(self.visual_cnns, frames):
            frame = visual_cnn(frame)
            #frame = frame.view(frame.size(0), -1)
            visual_features.append(frame)
            print("frame", frame.shape)
        
        # Process audio inputs
        audio_features = []
        for audio_conv, audio in zip(self.audio_convs, audios):
            audio = audio_conv(audio)
            audio_features.append(audio)
            print("audio", audio.shape)

        # Combine visual and audio features
        combined_features = []
        for visual_feature, audio_feature, image_audio_fc in zip(visual_features, audio_features, self.image_audio_fcs):
            combined = torch.cat((visual_feature, audio_feature), dim=1)
            print("combined", combined.shape)
            combined_features.append(image_audio_fc(combined))
        
        # Combine parallel streams
        combined_features = torch.stack(combined_features)
        combined_features = combined_features.view(combined_features.size(0), -1)
        print("combined_features", combined_features.shape)
        output = self.fc(combined_features)
        
        return output