import torch
import torchaudio
from blocks.pqmf import PQMF
import soundfile as sf
from aux import load_audio_mono, pqmf_encode, pqmf_decode

audio_path = 'audio/audio_1.wav'
pqmf = PQMF(attenuation=60, n_band=16, polyphase=True, n_channels=1)

def save_audio(tensor, sample_rate, output_path):
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    torchaudio.save(output_path, tensor, sample_rate)

if __name__ == "__main__":

    # Load and prepare audio
    audio = load_audio_mono(audio_path)
    print(f'shape before padding: {audio.shape}')

    # Process audio
    audio = audio.unsqueeze(0)
    print(f'shape after padding: {audio.shape}')

    # PQMF encode
    x_multiband = pqmf_encode(pqmf, audio)
    print(f'shape after pqmf encoding: {x_multiband.shape}')

    # PQMF decode
    reconstructed = pqmf_decode(pqmf, x_multiband, n_channels=1)
    print(f'shape after pqmf decoding: {reconstructed.shape}')

    # Play reconstructed audio
    print("Playing reconstructed audio...")
    save_audio(reconstructed, 44100, 'audio/audio_1_reconstructed.wav')
    print("Reconstructed audio saved to audio/audio_1_reconstructed.wav")



