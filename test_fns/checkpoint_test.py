import torch
from aux import load_audio_mono
import torchaudio
from model import JeffVAE
from blocks.encoder import Encoder, VariationalEncoder
from blocks.decoder import Generator
from blocks.pqmf import PQMF
from aux import AudioDistanceV1
from config import Config as config
from compute_features import compute_all_features
import numpy as np

checkpoint_path = 'lightning_logs/version_0/checkpoints/epoch=9-step=10.ckpt'
checkpoint = torch.load(checkpoint_path)

#hyperparameters
SAMPLING_RATE = 44100
CAPACITY = 64
N_BAND = 16
LATENT_SIZE = 64
RATIOS = [4, 4, 4, 2]
N_OUT = 2
    
#initialize model
model = JeffVAE(
        latent_size=LATENT_SIZE,
        encoder=config.VE,
        decoder=config.DECODER,
        pqmf=config.PQMF,
        latent_discriminator=config.LATENT_DISCRIMINATOR,
        multiband_audio_distance=AudioDistanceV1,
        learning_rate=1e-3  # Specify learning rate
)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

with torch.no_grad():
    # Load your audio
    x = load_audio_mono('audio/audio_1.wav')
    features = compute_all_features(x, latent_size=169)
    feature_arr = torch.tensor(np.array(list(features.values())).astype(
                    np.float32))
    print('feature_arr shape: ', feature_arr.shape)
    z, x_mb = model.encode(x)
    z_rp, kl_loss = model.encoder.reparametrize(z)
    print('z shape: ', z_rp.shape)
    z_c = torch.cat([z_rp, feature_arr.unsqueeze(0)], dim=1)
    y, y_mb = model.decode(z_c)
    
    # Save the output
    print('output shape: ', y.shape)
    y = y.cpu().squeeze(0)
    print('output squeezed shape: ', y.shape)# Remove batch and channel dimensions
    torchaudio.save('reconstructed_audio.wav', y, 44100)
