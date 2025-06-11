from blocks.encoder import Encoder, VariationalEncoder
from blocks.pqmf import PQMF
import torchaudio
from pqmf_test import pqmf_encode
import torch
from blocks.decoder import Generator
from aux import load_audio_mono, pqmf_encode, pqmf_decode
from config import Config as config
import math
from model import JeffVAE



pqmf = config.PQMF
encoder = config.ENCODER
ve = config.VE
decoder = config.DECODER
latent_discriminator = config.LATENT_DISCRIMINATOR
dataset = config.DATASET

batch = dataset[0]
audio = batch[0]
attributes = batch[1]
bin_values = batch[2]


model = JeffVAE(latent_size=128, encoder=ve, decoder=decoder, latent_discriminator=latent_discriminator, 
                multiband_audio_distance=None, n_bands=16, weights=None, pqmf=pqmf, n_channels=1, 
                input_mode="pqmf", output_mode="pqmf", learning_rate=1e-3)

print('attributes shape: ', attributes.shape)

model.training_step(batch)

