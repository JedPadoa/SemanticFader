import torch
import numpy as np
from udls import SimpleLMDBDataset
import os
from pathlib import Path
from tqdm import tqdm
import librosa as li
from compute_features import compute_all_features
import torchaudio.transforms as T
from scipy.io.wavfile import write as wavwrite
from config import Config as config
from aux import load_audio_mono
import math

folder_list = "audio"
sample_rate = config.SAMPLING_RATE
descriptors = config.DESCRIPTORS


def dummy_load(name):
    """
    Preprocess function that takes one audio path, crops it to 6 seconds,
    and returns it as 3 chunks of 2 seconds each (88200 samples per chunk at 44100 Hz).
    """
    target_length = int(config.AUDIO_LENGTH * config.SAMPLING_RATE)
    # Load audio at 44.1kHz
    x = li.load(name, sr=44100)[0]
    #x = load_audio_mono(name)
    
    # Calculate padding needed to make length divisible by 16
    remainder = target_length % 16
    if remainder != 0:
        padding = 16 - remainder
        target_length += padding
    
    # Crop or pad to exactly target length
    if len(x) > target_length:
        x = x[:target_length]
    elif len(x) < target_length:
        x = np.pad(x, (0, target_length - len(x)))
    
    # Reshape into chunks
    #x = x.reshape(int(config.AUDIO_LENGTH/2), -1)  # -1 will automatically calculate the correct chunk size
    
    if x.shape[0]:
        return x
    else:
        return None

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 audio_dir = None,
                 preprocess_function = dummy_load,
                 extension="*.wav,*.aif",
                 seed = 0,
                 sample_rate=config.SAMPLING_RATE, 
                 latent_size = config.LATENT_SIZE,
                 nb_bins=config.NUM_BINS, 
                 ratios=config.RATIOS, 
                 n_bands=config.N_BAND,
                 descriptors=config.DESCRIPTORS, 
                 batch_size=config.BATCH_SIZE, 
                 allfeatures = None
                 ):
        
        self.audio_chunks = []
        self.audio_dir = audio_dir
        self.preprocess_function = preprocess_function
        self.extension = extension
        self.sample_rate = sample_rate
        self.latent_size = latent_size
        self.nb_bins = nb_bins
        self.ratios = ratios
        self.n_bands = n_bands
        self.descriptors = descriptors
        self.batch_size = batch_size
        self.allfeatures = allfeatures
        self.resampled_length = math.ceil((config.AUDIO_LENGTH * config.SAMPLING_RATE) / 
                                          (self.n_bands * np.prod(self.ratios)))
    
        wavs = []
        if audio_dir is not None:
            for folder in audio_dir.split(","):
                for ext in ["*.wav", "*.aif"]:
                    wavs.extend(list(Path(folder).rglob(ext)))
                   
        # Process and store chunks
        for wav in tqdm(wavs):
            output = preprocess_function(wav)
            if output is not None:
                self.audio_chunks.append(output)
        
        # Stack all chunks into one array
        self.audio_chunks = np.stack(self.audio_chunks)
            
        print('database size: ', (len(self.audio_chunks)))
        
        if len(self.audio_chunks) == 0:
            raise Exception("No data found !")
        
        if self.allfeatures is None:
            self.compute_features()
        
        self.min_max_features = {}
        
        for i, descr in enumerate(self.descriptors):
            self.min_max_features[descr] = [
                np.min(self.allfeatures[:, i, :]),
                np.max(self.allfeatures[:, i, :])
            ]
        
        self.compute_bins(nb_bins)
        
        self.index = np.arange(len(self.audio_chunks))
        
    def compute_features(self):
        print('computing features...')
        indices = range(len(self.audio_chunks))
        self.allfeatures = []
        if self.sample_rate < 44100:
            resampler = T.Resample(self.sr, 44100)
        for i in tqdm(indices):
            try:

                features = compute_all_features(self.audio_chunks[i],
                                       descriptors=self.descriptors,
                                       resampled_length=self.resampled_length)

                features = {
                    descr: features[descr]
                    for descr in self.descriptors
                }
                feature_arr = np.array(list(features.values())).astype(
                    np.float32)

                self.allfeatures.append(feature_arr)
                first = True
            except:
                # break
                print('failed features compute')
                print('index failure : ', i)
                self.allfeatures.append(
                    np.zeros((len(self.descriptors),
                              self.latent_size)).astype(np.float32))

        self.allfeatures = np.array(self.allfeatures)
        print('features computed')
    
    def compute_bins(self, nb_bins):
        print('computing bins...')
        all_values = []
        for i, descr in enumerate(self.descriptors):
            data = self.allfeatures[:, i, :].flatten()
            data[data < 0] = 0
            data_true = data.copy()
            data.sort()
            index = np.linspace(0, len(data) - 2, nb_bins).astype(int)
            values = [data[i] for i in index]
            all_values.append(values)
        print('bins computed')
        self.bin_values = np.array(all_values)
        
    def __len__(self):
        return len(self.audio_chunks)
    
    def __getitem__(self, index):
        feature_arr = torch.tensor(self.allfeatures[index].copy())
        # Add channel dimension: [4, 88200] -> flatten -> [352800] -> [1, 352800]
        audio = torch.tensor(self.audio_chunks[index].flatten(), dtype=torch.float32)
        return audio, feature_arr, self.bin_values
        
if __name__ == "__main__":
    db = AudioDataset(
        folder_list = folder_list,
        sample_rate = sample_rate,
        descriptors = descriptors
    )
    
    for i in range(len(db)):
        audio_data = db[i][0].flatten()  # flatten to 1D for wav
        wavwrite(f"test_{i}.wav", sample_rate, audio_data.astype(np.float32))
        print(f"Saved test_{i}.wav")