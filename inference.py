import torch
import torchaudio
from model import JeffVAE
from aux import load_audio_mono, AudioDistanceV1
from config import Config as config
import numpy as np
from regCAV import RegCAV
import math

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
        
        self.rcv_speed = RegCAV()
        self.rcv_speed.load_rcv('RCVs/speed_rcv.pkl')
        self.rcv_grass = RegCAV()
        self.rcv_grass.load_rcv('RCVs/grass_rcv.pkl')
        
        
        self.target_len = int(config.AUDIO_LENGTH * config.SAMPLING_RATE)
        self.resampled_length = math.ceil(self.target_len / (config.N_BAND * np.prod(config.RATIOS)))
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)['state_dict']
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
    def _normalize_features(self, features, min_val = -0.12417352199554443, max_val = 1.1036418676376343):
        feats_np = np.array(features, dtype=np.float32)
        delta = max_val - min_val
        if delta == 0:
            norm_feats_np = np.zeros_like(feats_np, dtype=np.float32)
        else:
            norm_feats_np = -1 + (feats_np - min_val) * 2 / delta
        
        return torch.tensor(norm_feats_np, dtype=torch.float32)
        
    def autoencode(self, audio_path, output_path=None, bias = 0):
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
            x = x.to(self.device)
            
            # Extract features
            score_speed = self.rcv_speed.compute_attribute_score(audio_path)
            score_grass = self.rcv_grass.compute_attribute_score(audio_path)
            
            feature_arr = torch.stack([
                torch.full((self.resampled_length,), score_speed, dtype=torch.float32),
                torch.full((self.resampled_length,), score_grass, dtype=torch.float32)
            ])
            
            features_norm = self._normalize_features(feature_arr)
            
            print('original speed value: ', features_norm[0][0])
            print('original grass value: ', features_norm[1][0])
            print('bias: ', bias)
            features_norm[1] += bias
            
            
            # Encode
            z, x_mb = self.model.encode(x)
            z_rp, kl_loss = self.model.encoder.reparametrize(z)
            
            # Concatenate latent with features
            z_c = torch.cat([z_rp, features_norm.unsqueeze(0)], dim=1)
            
            #delta = 1.1036418676376343 - 0.09266293048858643
            #print('score: ', -1 + (score - 0.09266293048858643) * 2 / delta)
            
            # Decode
            y, y_mb = self.model.decode(z_c)
            
            # Process output
            y = y.cpu().squeeze(0)
            
            if output_path:
                torchaudio.save(output_path, y, 44100)
                return None
            return y

if __name__ == "__main__":
    model = Model('sfRAVE-128k.ckpt')
    o = model.autoencode('ffxFootstepsGenData/steps_spe_0.20_con_1.00_woo_0.00_gra_0.00.wav', 
                         'audio_tests/sf_rave_sg_128k/output_bias_grass_2.0.wav', bias = 2.0)