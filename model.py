from blocks.pqmf import PQMF
import torch.nn as nn
from typing import Callable, Optional, Dict
import torch
import pytorch_lightning as pl
from aux import AudioDistanceV1, MultiScaleSTFT, pqmf_encode, pqmf_decode, load_audio_mono
import torch.nn.functional as F

class JeffVAE(pl.LightningModule):
    def __init__(self,
                 latent_size, encoder, decoder, latent_discriminator,
                 multiband_audio_distance: Callable[[], nn.Module],
                 n_bands: int = 16, 
                 weights: Optional[Dict[str, float]] = None,
                 pqmf: Optional[PQMF] = None,
                 n_channels: int = 1,
                 input_mode: str = "pqmf",
                 output_mode: str = "pqmf",
                 learning_rate: float = 1e-3,  # Add learning rate parameter
                 verbose: bool = False
                 ):
        super().__init__() 
        self.encoder = encoder
        self.decoder = decoder
        self.latent_discriminator = latent_discriminator
        self.pqmf = pqmf
        
        # Create the MultiScaleSTFT instance first
        multiscale_stft = MultiScaleSTFT(
            scales=[2048, 1024, 512, 256, 128],
            sample_rate=44100
        )
        # Create the AudioDistanceV1 instance directly
        self.multiband_audio_distance = AudioDistanceV1(
            lambda: multiscale_stft,  # Pass a lambda that returns the instance
            log_epsilon=1e-7
        )
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.n_bands = n_bands
        self.weights = weights
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.n_channels = n_channels
        self.verbose = verbose
        
    def configure_optimizers(self):
        # Get all parameters that need optimization
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            betas=(0.5, 0.9)
        )
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # Return the optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",  # Monitor training loss
                "frequency": 1  # Adjust learning rate every epoch
            }
        }
    
    def train_dataloader(self):
        return self.train_loader
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_mb = pqmf_encode(self.pqmf, x.unsqueeze(1))
        if self.verbose:
            print('x_mb shape: ', x_mb.shape)
        z = self.encoder(x_mb)
        if self.verbose:
            print(f'After encoder shape: {z.shape}')
        return z, x_mb
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y_mb = self.decoder(z)
        y = pqmf_decode(self.pqmf, y_mb, self.n_channels)
        return y, y_mb
    
    def forward(self, x):
        z, x_mb = self.encode(x)
        z = self.encoder.reparametrize(z)[0]
        return self.decode(z)
    
    def get_attr_loss(self, output, attributes_cls):
        """
        Compute attributes loss.
        """
        loss = F.cross_entropy(output, attributes_cls)
        # k += n_cat
        return loss
    
    def quantify(self, allarr, bins):
        nz = allarr.shape[-1]
        allarr_cls = torch.zeros_like(allarr)

        for i in range(allarr.shape[1]):
            # Make both tensors contiguous
            data = allarr[:, i, :].contiguous()  # Add .contiguous() here too
            bins_slice = bins[i, 1:].contiguous()
            data_cls = torch.bucketize(data, bins_slice, right=False)
            allarr_cls[:, i, :] = data_cls

        return allarr_cls
    
    def training_step(self, batch):
        # Get input
        x, attributes, bin_values = batch
        
        if self.verbose:
            print("Input shapes:")
            print(f"x: {x.shape}")
            print(f"attributes: {attributes.shape}")
            print(f"bin_values: {bin_values.shape}")
        
        # Encode input to get latent representation
        z, x_mb = self.encode(x)
        if self.verbose:
            print(f"\nAfter encode:")
            print(f"z: {z.shape}")
            print(f"x_mb: {x_mb.shape}")
        
        # Reparametrize to get latent vector and KL divergence loss
        z, kl_loss = self.encoder.reparametrize(z)
        if self.verbose:    
            print(f"\nAfter reparametrize:")
            print(f"z: {z.shape}")
        
        # Get predictions from latent discriminator
        attr_preds = self.latent_discriminator(z)
        if self.verbose:
            print(f"\nAfter latent discriminator:")
            print(f"attr_preds: {attr_preds.shape}")
        
        if self.verbose:
            print(f'attributes shape: {attributes.shape}')
        # Quantize attributes
        attr_quantized = self.quantify(attributes, bin_values.clone().detach().to(self.device)).long()
        if self.verbose:
            print(f"\nAfter quantify:")
            print(f"attr_quantized: {attr_quantized.shape}")
        
        # Calculate prediction loss
        pred_loss = self.get_attr_loss(attr_preds, attr_quantized)
        if self.verbose:
            print(f"\nAfter get_attr_loss:")
            print(f"pred_loss: {pred_loss}")
        
        z_c = torch.cat([z, attributes], dim=1)
        # Decode latent vector back to audio
        y, y_mb = self.decode(z_c)  # y_mb: [4, 16, 4096]
        
        # Trim y_mb to match x_mb length
        y_mb = y_mb[..., :x_mb.shape[-1]]  # Now y_mb will be [4, 16, 4000]
        
        # Calculate reconstruction loss using the multiband representations
        recon_loss = self.multiband_audio_distance(x_mb, y_mb)
        if self.verbose:
            print(f'recon_loss: {recon_loss}')
        
        # Combine losses
        beta = 0.1
        total_loss = recon_loss['spectral_distance'] + beta * kl_loss
        
        # Log metrics
        self.log('loss', total_loss, prog_bar=True)
        self.log('recon_loss', recon_loss['spectral_distance'], prog_bar=True)
        self.log('kl_loss', kl_loss, prog_bar=True)
        
        return total_loss
    