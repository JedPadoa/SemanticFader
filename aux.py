from typing import Callable, Optional, Sequence, Union
import librosa as li
import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
import torch.fft as fft
import numpy as np

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

def valid_signal_crop(x, left_rf, right_rf):
    dim = x.shape[1]
    x = x[..., left_rf.item() // dim:]
    if right_rf.item():
        x = x[..., :-right_rf.item() // dim]
    return x


def relative_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return norm(x - y) / norm(x)


def mean_difference(target: torch.Tensor,
                    value: torch.Tensor,
                    norm: str = 'L1',
                    relative: bool = False):
    diff = target - value
    if norm == 'L1':
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == 'L2':
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')

def amp_to_impulse_response(amp, target_size):
    """
    transforms frequency amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp

def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


class MelScale(nn.Module):

    def __init__(self, sample_rate: int, n_fft: int, n_mels: int) -> None:
        super().__init__()
        mel = li.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        mel = torch.from_numpy(mel).float()
        self.register_buffer('mel', mel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.mel.type_as(x)
        y = torch.einsum('bft,mf->bmt', x, mel)
        return y


class MultiScaleSTFT(nn.Module):

    def __init__(self,
                 scales: Sequence[int],
                 sample_rate: int,
                 magnitude: bool = True,
                 normalized: bool = False,
                 num_mels: Optional[int] = None) -> None:
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        self.mel_scales = []
        for scale in scales:
            self.stfts.append(
                torchaudio.transforms.Spectrogram(
                    n_fft=scale,
                    win_length=scale,
                    hop_length=scale // 4,
                    normalized=normalized,
                    power=None,
                ))
            if num_mels is not None:
                self.mel_scales.append(
                    MelScale(
                        sample_rate=sample_rate,
                        n_fft=scale,
                        n_mels=num_mels,
                    ))
            else:
                self.mel_scales.append(None)

        self.stfts = nn.ModuleList(self.stfts)
        self.mel_scales = nn.ModuleList(self.mel_scales)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = rearrange(x, "b c t -> (b c) t")
        stfts = []
        for stft, mel in zip(self.stfts, self.mel_scales):
            y = stft(x)
            if mel is not None:
                y = mel(y)
            if self.magnitude:
                y = y.abs()
            else:
                y = torch.stack([y.real, y.imag], -1)
            stfts.append(y)

        return stfts


class AudioDistanceV1(nn.Module):

    def __init__(self, multiscale_stft: Callable[[], nn.Module],
                 log_epsilon: float) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft()
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = mean_difference(x, y, norm='L2', relative=True)
            log_distance = mean_difference(logx, logy, norm='L1')

            distance = distance + lin_distance + log_distance

        return {'spectral_distance': distance}

def get_valid_extensions():
    import torchaudio
    backend = torchaudio.get_audio_backend()
    if backend in ["sox_io", "sox"]:
        return ['.'+f for f in torchaudio.utils.sox_utils.list_read_formats()]
    elif backend == "ffmpeg":
        return ['.'+f for f in torchaudio.utils.ffmpeg_utils.get_audio_decoders()]
    elif backend == "soundfile":
        return ['.wav', '.flac', '.ogg', '.aiff', '.aif', '.aifc']

def pqmf_encode(pqmf, x: torch.Tensor):
    batch_size = x.shape[:-2]
    x_multiband = x.reshape(-1, 1, x.shape[-1])
    x_multiband = pqmf(x_multiband)
    x_multiband = x_multiband.reshape(*batch_size, -1, x_multiband.shape[-1])
    return x_multiband

def pqmf_decode(pqmf, x: torch.Tensor, n_channels: int):
    x = x.reshape(x.shape[0] * n_channels, -1, x.shape[-1])
    x = pqmf.inverse(x)
    return x

def load_audio_mono(audio_path, sample_rate=44100):
    """
    Load audio file and convert to mono if necessary
    Args:
        audio_path (str): Path to audio file
        sample_rate (int): Target sample rate
    Returns:
        torch.Tensor: Mono audio tensor of shape (1, num_samples)
    """
    # Load audio file
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono by averaging channels if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return _pad_to_multiple(waveform, 16)

def _pad_to_multiple(x: torch.Tensor, multiple: int):
    """
    Pad the audio tensor to be divisible by multiple
    Args:
        x: Input tensor of shape [B, C, T]
        multiple: The divisor (e.g., 16)
    Returns:
        Padded tensor
    """
    length = x.shape[-1]
    padding_length = (multiple - (length % multiple)) % multiple
    return torch.nn.functional.pad(x, (0, padding_length))

def dummy_load(name, target_length):
    """
    Preprocess function that takes one audio path, crops it to 6 seconds,
    and returns it as 3 chunks of 2 seconds each (88200 samples per chunk at 44100 Hz).
    """
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




