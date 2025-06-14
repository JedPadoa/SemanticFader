from blocks.encoder import Encoder, VariationalEncoder
from blocks.pqmf import PQMF
from blocks.decoder import Generator
from blocks.latent_discriminator import LatentDiscriminator
import torch
import pytorch_lightning as pl
from aux import AudioDistanceV1
from dataset_lmdb import AudioDataset
from model import JeffVAE
from config import Config as config
import torch.multiprocessing as mp
import os
from torch.utils.data import DataLoader
# Set number of threads before any parallel operations
#os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
#os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
#torch.set_num_threads(1) 

def custom_collate(batch):
    audios, features, bin_values = zip(*batch)
    
    # Stack audios and features normally
    audios = torch.stack(audios)
    features = torch.stack(features) 
    
    # bin_values should be the same for all samples, so just take the first one
    # This prevents it from being batched
    bin_values = bin_values[0]  # Take first sample's bin_values
    
    return audios, features, bin_values

if __name__ == "__main__":
    #set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create dataset once
    train_dataset = AudioDataset(
        db_path='data/footsteps_db',
        descriptors=config.DESCRIPTORS,
        nb_bins=config.NUM_BINS
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    #initialize components
    encoder = config.ENCODER
    variational_encoder = config.VE
    decoder = config.DECODER
    pqmf = config.PQMF
    latent_discriminator = config.LATENT_DISCRIMINATOR
    
    #initialize model
    model = JeffVAE(
        latent_size=config.LATENT_SIZE,
        encoder=variational_encoder,
        decoder=decoder,
        latent_discriminator=latent_discriminator,
        pqmf=pqmf,
        multiband_audio_distance=AudioDistanceV1,
        learning_rate=1e-3  # Specify learning rate
    )
    model.train_loader = train_loader
    model = model.to(device)
    
    #initialize optimizer
    model.configure_optimizers()  # Set up the optimizer for the training step
    
     # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu',  # Will automatically detect available hardware
        devices=1,
        logger=True, 
        log_every_n_steps=1# Add TensorBoard logging
    )
    
    # Train the model
    trainer.fit(model, train_loader)
