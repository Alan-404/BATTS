import os

import torch
import torch.nn.functional as F

from torchaudio.transforms import MelSpectrogram
from typing import Optional, List

import librosa
from scipy.io import wavfile

MAX_AUDIO_VALUE = 32768.0

class TargetBATTSProcessor:
    def __init__(self,
                 sample_rate: int = 22050, n_mels: int = 80, n_fft: int = 2048, win_length: int = 1024, hop_length: int = 256, max_duration: Optional[float] = None, mel_norm_path: Optional[str] = None,
                 device: str = 'cpu') -> None:
        # Audio
        self.hop_length = hop_length
        self.max_audio_samples = int(max_duration * sample_rate) if max_duration is not None else None
        self.sample_rate = sample_rate

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=0,
            f_max=8000.0,
            norm='slaney',
            n_mels=n_mels
        ).to(device)

        self.mel_norm = None
        if mel_norm_path is not None and os.path.exists(mel_norm_path):
            self.mel_norm = torch.load(mel_norm_path, map_location=device).unsqueeze(0).unsqueeze(-1)

        self.device = device
    
    def load_audios(self, paths: List[str]) -> List[torch.Tensor]:
        signals = []
        for path in paths:
            signals.append(self.load_audio(path))
        return signals

    def load_audio(self, path: str) -> torch.Tensor:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        if self.max_audio_samples is not None and self.max_audio_samples < len(signal):
            signal = signal[:self.max_audio_samples]

        signal = torch.tensor(signal, dtype=torch.float)
        signal = torch.clamp(signal, min=-1, max=1)
        return signal
    
    def log_mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(signal)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norm is not None:
            mel = mel / self.mel_norm
        return mel
    
    def __call__(self, signals: List[torch.Tensor]) -> os.Any:
        max_signal_length = 0
        signal_lengths = []

        for i in range(len(signals)):  
            signal_length = len(signals[i])
            if signal_length > max_signal_length:
                max_signal_length = signal_length
            signal_lengths.append(signal_length)

        padded_signals = []
        for i in range(len(signals)):
            padded_signals.append(
                F.pad(signals[i], pad=(0, max_signal_length - signal_lengths[i]), value=0.0)
            )

        padded_signals = torch.stack(padded_signals)
        log_mel = self.log_mel_spectrogram(padded_signals)

        return log_mel, torch.tensor(signal_lengths) // self.hop_length + 1