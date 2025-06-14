import torch
import torchaudio
from model import JeffVAE
from aux import load_audio_mono, AudioDistanceV1
from config import Config as config
from test_fns.compute_features import compute_all_features
import numpy as np

class Model():
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the inference class with a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint file
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Create model instance
        self.model = JeffVAE(
            latent_size=config.LATENT_SIZE,
            encoder=config.VE,
            decoder=config.DECODER,
            pqmf=config.PQMF,
            latent_discriminator=config.LATENT_DISCRIMINATOR,
            multiband_audio_distance=AudioDistanceV1,
            learning_rate=1e-3
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)['state_dict']
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
    def autoencode(self, audio_path, output_path=None):
        """
        Process an audio file through the VAE.
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str, optional): Path to save output audio. If None, returns tensor.
            
        Returns:
            torch.Tensor: Reconstructed audio if output_path is None
        """
        with torch.no_grad():
            # Load and preprocess audio
            x = load_audio_mono(audio_path)
            print(x.shape)
            x = x.to(self.device)
            
            # Extract features
            features = compute_all_features(x)
            feature_arr = torch.tensor(np.array(list(features.values())).astype(
                    np.float32))
            feature_arr = feature_arr.to(self.device)
            
            # Encode
            z, x_mb = self.model.encode(x)
            z_rp, kl_loss = self.model.encoder.reparametrize(z)
            
            # Concatenate latent with features
            z_c = torch.cat([z_rp, feature_arr.unsqueeze(0)], dim=1)
            
            # Decode
            y, y_mb = self.model.decode(z_c)
            
            # Process output
            y = y.cpu().squeeze(0)
            
            if output_path:
                torchaudio.save(output_path, y, 44100)
                return None
            return y

if __name__ == "__main__":
    model = Model('lightning_logs/version_0/checkpoints/epoch=9-step=10.ckpt')
    o = model.autoencode('audio/audio_1.wav')