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
        
        # Enable manual optimization - REQUIRED for multiple optimizers
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        # Main model optimizer (encoder + decoder)
        main_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        main_optimizer = torch.optim.Adam(
            main_params,
            lr=self.learning_rate,
            betas=(0.5, 0.9)
        )
        
        # Latent discriminator optimizer
        lat_dis_optimizer = torch.optim.Adam(
            self.latent_discriminator.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Return as a list (no scheduler for now with manual optimization)
        return [main_optimizer, lat_dis_optimizer]
    
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
            data = allarr[:, i, :].flatten() 
            data_cls = torch.bucketize(data, bins[i, 1:], right=False)
            allarr_cls[:, i, :] = data_cls.reshape(-1, nz)

        return allarr_cls
    
    def training_step(self, batch, batch_idx):
        # Get optimizers
        main_opt, dis_opt = self.optimizers()
        
        x, attributes, bin_values = batch
        
        # ===== MAIN MODEL TRAINING =====
        main_opt.zero_grad()
        
        self.latent_discriminator.eval()
        self.encoder.train()
        self.decoder.train()
        
        # Your existing main training logic
        z, x_mb = self.encode(x)
        z, kl_loss = self.encoder.reparametrize(z)
        
        # Get predictions from latent discriminator (for adversarial loss)
        attr_preds = self.latent_discriminator(z)
        attr_quantized = self.quantify(attributes, bin_values.clone().detach()).long()
        lat_dis_loss = -self.get_attr_loss(attr_preds, attr_quantized)
        
        z_c = torch.cat([z, attributes], dim=1)
        y, y_mb = self.decode(z_c)
        y_mb = y_mb[..., :x_mb.shape[-1]]
        
        recon_loss = self.multiband_audio_distance(x_mb, y_mb)
        
        beta = 0.1
        main_loss = recon_loss['spectral_distance'] + beta * kl_loss + lat_dis_loss
        
        # Manual backward and step
        self.manual_backward(main_loss)
        main_opt.step()
        
        # ===== LATENT DISCRIMINATOR TRAINING =====
        dis_opt.zero_grad()
        
        self.encoder.eval()
        self.decoder.eval()
        self.latent_discriminator.train()
        
        # Encode (with no_grad since we don't want to update encoder)
        with torch.no_grad():
            z, _ = self.encode(x)
            z, _ = self.encoder.reparametrize(z)
        
        attr_preds = self.latent_discriminator(z)
        attr_quantized = self.quantify(attributes, bin_values.clone().detach()).long()
        
        # Discriminator wants to correctly classify attributes
        dis_loss = self.get_attr_loss(attr_preds, attr_quantized)
        
        # Manual backward and step
        self.manual_backward(dis_loss)
        dis_opt.step()
        
        # Log metrics
        self.log('loss', main_loss, prog_bar=True)
        self.log('recon_loss', recon_loss['spectral_distance'], prog_bar=True)
        self.log('kl_loss', kl_loss, prog_bar=True)
        self.log('adv_loss', lat_dis_loss, prog_bar=True)
        self.log('dis_loss', dis_loss, prog_bar=True)
        
        # Return main loss for logging purposes
        return main_loss
    
    def validation_step(self, batch, batch_idx):
        # Get input
        x, attributes, bin_values = batch
        
        # Encode input to get latent representation
        z, x_mb = self.encode(x)
        
        # Reparametrize to get latent vector and KL divergence loss
        z, kl_loss = self.encoder.reparametrize(z)
        
        # Get predictions from latent discriminator
        attr_preds = self.latent_discriminator(z)
        
        # Quantize attributes
        attr_quantized = self.quantify(attributes, bin_values.clone().detach()).long()
        
        # Calculate prediction loss
        pred_loss = self.get_attr_loss(attr_preds, attr_quantized)
        
        z_c = torch.cat([z, attributes], dim=1)
        # Decode latent vector back to audio
        y, y_mb = self.decode(z_c)
        
        # Trim y_mb to match x_mb length
        y_mb = y_mb[..., :x_mb.shape[-1]]
        
        # Calculate reconstruction loss using the multiband representations
        recon_loss = self.multiband_audio_distance(x_mb, y_mb)
        
        # Combine losses
        beta = 0.1
        val_total_loss = recon_loss['spectral_distance']
        
        # Log validation metrics
        self.log('val_loss', val_total_loss, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss['spectral_distance'], prog_bar=True, sync_dist=True)
        self.log('val_kl_loss', kl_loss, prog_bar=True, sync_dist=True)
        self.log('val_pred_loss', pred_loss, prog_bar=True, sync_dist=True)
        
        return val_total_loss
    