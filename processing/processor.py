import os

import torch
import torch.nn.functional as F

from torchaudio.transforms import MelSpectrogram, Resample

import numpy as np

import re
import librosa
from scipy.io import wavfile
import json

from typing import Optional, List, Dict

MAX_AUDIO_VALUE = 32768.0

class BATTSProcessor:
    def __init__(self,
                 tokenizer_path: str, pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>", puncs: str = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\-\\])",
                 sample_rate: int = 22050, encoder_sr: int = 16000, n_mels: int = 80, n_fft: int = 2048, win_length: int = 1024, hop_length: int = 256, max_duration: Optional[float] = None, mel_norm_path: Optional[str] = None,
                 device: str = 'cpu') -> None:
        # Text
        patterns = json.load(open(tokenizer_path, 'r', encoding='utf8'))
        self.yolo = patterns
        self.replace_dict = patterns['replace']
        self.mapping = patterns['mapping']

        self.single_vowels = patterns['single_vowel']

        vocab = []

        for key in patterns.keys():
            if key == 'replace' or key == 'mapping':
                continue
            vocab += patterns[key]
            
        self.dictionary = self.create_vocab_dictionary(vocab, pad_token, delim_token, unk_token)

        self.pattern = self.sort_pattern(vocab + list(patterns['mapping'].keys()))
        self.puncs = puncs

        self.delim_token = delim_token
        self.pad_token = pad_token
        self.unk_token = unk_token
    
        self.delim_id = self.find_token_id(delim_token)
        self.pad_id = self.find_token_id(pad_token)
        self.unk_id = self.find_token_id(unk_token)

        # Audio
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

        self.resampler = Resample(orig_freq=sample_rate, new_freq=encoder_sr)

        self.mel_norm = None
        if mel_norm_path is not None and os.path.exists(mel_norm_path):
            self.mel_norm = torch.load(mel_norm_path, map_location=device).unsqueeze(0).unsqueeze(-1)

        self.device = device

    def create_vocab_dictionary(self, vocab: List[str], pad_token: str, delim_token: str, unk_token: str) -> Dict[str, int]:
        dictionary = []

        dictionary.append(delim_token)
        for item in vocab:
            if item not in dictionary:
                dictionary.append(item)
        dictionary.append(pad_token)
        dictionary.append(unk_token)
        
        return dictionary
    
    def sort_pattern(self, patterns: List[str]) -> List[str]:
        patterns = sorted(patterns, key=len)
        patterns.reverse()

        return patterns
    
    def find_token_id(self, token: str) -> int:
        if token in self.dictionary:
            return self.dictionary.index(token)
        return self.dictionary.index(self.unk_token)
    
    def token2text(self, tokens: np.ndarray, get_string: bool = False) -> str:
        words = []
        for token in tokens:
            words.append(self.dictionary[token])

        if get_string:
            return "".join(words).replace(self.delim_token, " ")
        
        return words
    
    def spec_replace(self, word: str) -> str:
        for key in self.replace_dict:
            arr = word.split(key)
            if len(arr) == 2:
                if arr[1] in self.single_vowels:
                    return word
                else:
                    return word.replace(key, self.replace_dict[key])
        return word
    
    def sentence2tokens(self, sentence: str) -> List[int]:
        phonemes = self.sentence2phonemes(sentence)
        tokens = self.phonemes2tokens(phonemes)
        return tokens

    def sentences2tokens(self, sentences: List[str]) -> List[torch.Tensor]:
        tokens = []
        for sentence in sentences:
            tokens.append(self.sentence2tokens(sentence))
        return tokens

    def phonemes2tokens(self, phonemes: List[str]):
        tokens = []
        for phoneme in phonemes:
            tokens.append(self.find_token_id(phoneme))
        return torch.tensor(tokens)
    
    def clean_text(self, sentence: str) -> str:
        sentence = str(sentence)
        sentence = re.sub(self.puncs, r" \1 ", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 4, reverse: bool = True) -> List[str]:
        if len(text) == 1:
            if text in patterns:
                if text in self.mapping:
                    return [self.mapping[text]]
                else:
                    return [text]
            return [self.unk_token]
        if reverse:
            text = [text[i] for i in range(len(text) - 1, -1, -1)]
            text = "".join(text)
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            if reverse:
                item = [item[i] for i in range(len(item) - 1, -1, -1)]
                item = "".join(item)
                
            if item in patterns:
                if item in self.mapping:
                    graphemes.append(self.mapping[item])
                else:
                    graphemes.append(item)
            elif num_steps == 1:
                graphemes.append(self.unk_token)
            else:
                found = False

            if found:
                start += num_steps
                if len(text[start:]) < n_grams:
                    num_steps = len(text[start:])
                else:
                    num_steps = n_grams
            else:
                num_steps -= 1

        if reverse:
            graphemes = [graphemes[i] for i in range(len(graphemes) - 1, -1, -1)]

        return graphemes

    def sentence2phonemes(self, sentence: str):
        sentence = self.clean_text(sentence.upper())
        sentence = sentence.replace("%", "PHẦN TRĂM").replace("…", "...").replace("°C", "ĐỘ XÊ")
        words = sentence.split(" ")
        graphemes = []

        length = len(words)

        for index, word in enumerate(words):
            if word in ['.', ',', '!', '?']:
                graphemes[-1] = word
                continue
            graphemes += self.slide_graphemes(self.spec_replace(word), self.pattern, n_grams=4, reverse=False)
            if index != length - 1:
                graphemes.append(self.delim_token)

        return graphemes
    
    def sentences2phonemes(self, sentences: List[str]):
        phonemes = []
        for sentence in sentences:
            phonemes.append(self.sentence2phonemes(sentence))
        return phonemes
    
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

        signal = torch.tensor(signal, dtype=torch.float32)
        signal = torch.clamp(signal, min=-1, max=1)
        return signal
    
    def log_mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(signal)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norm is not None:
            mel = mel / self.mel_norm
        return mel
    
    def __call__(self, tokens: List[List[int]], signals: List[torch.Tensor]):
        max_token_length = 0
        token_lengths = []
        max_signal_length = 0
        signal_lengths = []

        for i in range(len(tokens)):
            token_length = len(tokens[i])
            if token_length > max_token_length:
                max_token_length = token_length
            token_lengths.append(token_length)
            
            signal_length = len(signals[i])
            if signal_length > max_signal_length:
                max_signal_length = signal_length
            signal_lengths.append(signal_length)

        padded_tokens = []
        padded_signals = []
        for i in range(len(tokens)):
            padded_tokens.append(
                F.pad(tokens[i], pad=(0, max_token_length - token_lengths[i]), value=self.pad_id)
            )
            padded_signals.append(
                F.pad(signals[i], pad=(0, max_signal_length - signal_lengths[i]), value=0.0)
            )

        padded_signals = torch.stack(padded_signals)
        log_mel = self.log_mel_spectrogram(padded_signals)
        se_signals = self.resampler(padded_signals)

        return torch.stack(padded_tokens), log_mel, se_signals, torch.tensor(token_lengths)