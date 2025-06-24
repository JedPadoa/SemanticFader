from blocks.encoder import Encoder, VariationalEncoder
from blocks.decoder import Generator
from blocks.latent_discriminator import LatentDiscriminator
from blocks.pqmf import PQMF
import torch

class Config:
    
    # attribute computation parameters
    TEXTS = ['very fast footsteps', 'fast footsteps', 'footsteps']
    
    # latent discriminator parameters
    NUM_ATTRIBUTES = 1
    NUM_CLASSES = 16
    NUM_LAYERS = 3
    NUM_BINS = 16
    
    # VAE parameters
    CAPACITY = 64
    LATENT_SIZE = 64
    RATIOS = [4, 4, 4, 2]
    N_OUT = 2
    N_CHANNELS = 1
    
    # Audio parameters
    SAMPLING_RATE = 44100
    N_BAND = 16 
    AUDIO_LENGTH = 6
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    BETA = 0.1  # For KL loss weighting
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 10000
    
    # STFT parameters
    STFT_SCALES = [2048, 1024, 512, 256, 128]
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    DATA_DIR = "audio"
    DESCRIPTORS = ["speed"]
    
    #initialize components
    ENCODER = Encoder(data_size=N_BAND, capacity=CAPACITY, latent_size=LATENT_SIZE,
                      ratios=RATIOS, n_out=N_OUT, sample_norm=False, repeat_layers=1, n_channels=1)
    VE = VariationalEncoder(ENCODER)
    DECODER = Generator(latent_size=LATENT_SIZE + len(DESCRIPTORS), capacity=CAPACITY, data_size=N_BAND,
                   ratios=RATIOS, loud_stride=1, use_noise=False, n_channels=1)
    PQMF = PQMF(attenuation=60, n_band=N_BAND, polyphase=True, n_channels=1)
    LATENT_DISCRIMINATOR = LatentDiscriminator(latent_size=LATENT_SIZE, num_attributes=NUM_ATTRIBUTES, num_classes=NUM_CLASSES, num_layers=NUM_LAYERS)

# You can also have different configs for different environments
class DevelopmentConfig(Config):
    DEBUG = True
    BATCH_SIZE = 4

class ProductionConfig(Config):
    DEBUG = False