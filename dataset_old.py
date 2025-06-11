import os
from torch.utils.data import Dataset, DataLoader
import torch
from blocks.encoder import Encoder, VariationalEncoder
from blocks.decoder import Generator
from blocks.pqmf import PQMF
from aux import load_audio_mono
import pytorch_lightning as pl
import math
import numpy as np
from udls import SimpleLMDBDataset

class AudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=44100, segment_length=128000,
                 nb_bins=16, ratios=[4, 4, 4, 2], n_bands=16, descriptors=[],
                 batch_size=1):
        """
        Args:
            audio_dir (str): Directory containing audio files
            sample_rate (int): Target sample rate for all audio
            segment_length (int): Target segment length for all audio
            batch_size (int): Number of samples to return in each batch
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.batch_size = batch_size
        out_database_location = "data/audio_db"
        map_size = 1e12
        self.env = SimpleLMDBDataset(out_database_location, map_size)
        self.len() = len(self.env)
        # Get all audio files
        self.audio_files = [
            f for f in os.listdir(audio_dir) 
            if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))
        ]   
        dr = torch.prod(torch.tensor(ratios))
        fake_attributes = np.array(torch.rand(4, math.ceil((128000 // n_bands) / dr)) * 4)
        self.allfeatures = fake_attributes
        self.nb_bins = nb_bins
        self.descriptors = descriptors
        
    def __len__(self):
        return len(self.audio_files) // self.batch_size
    
    def compute_bins(self, nb_bins):
        all_values = []
        for i, descr in enumerate(self.descriptors):
            data = self.allfeatures[i, :].flatten()
            data[data < 0] = 0
            data_true = data.copy()
            data.sort()
            index = np.linspace(0, len(data) - 2, nb_bins).astype(int)
            values = [data[i] for i in index]
            all_values.append(values)
        bin_values = np.array(all_values)
        return bin_values
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        audio_batch = []
        features_batch = []
        bin_values_batch = []
        
        for file_idx in range(start_idx, end_idx):
            audio_path = os.path.join(self.audio_dir, self.audio_files[file_idx])
            audio = load_audio_mono(audio_path, self.sample_rate)
            
            if audio.shape[-1] > self.segment_length:
                start = torch.randint(0, audio.shape[-1] - self.segment_length, (1,))
                audio = audio[..., start:start + self.segment_length]
            else:
                padding = self.segment_length - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            
            audio_batch.append(audio)
            features_batch.append(self.allfeatures)
            bin_values_batch.append(self.compute_bins(self.nb_bins))
        
        audio_batch = torch.stack(audio_batch)
        features_batch = torch.stack([torch.tensor(f) for f in features_batch])
        bin_values_batch = torch.stack([torch.tensor(b) for b in bin_values_batch])
        
        return audio_batch, features_batch, bin_values_batch

