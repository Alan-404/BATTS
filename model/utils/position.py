import torch
import torch.nn as nn

from typing import Tuple

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer(
            name='angles',
            tensor=(1.0 / 10000.0 ** (torch.arange(0, dim, 2) / dim)).unsqueeze(0),
        )
    
    def forward(self, length: int) -> torch.Tensor:
        pe = torch.zeros((length, self.dim), dtype=self.angles.dtype, device=self.angles.device)
        pos = torch.arange(length, dtype=self.angles.dtype, device=self.angles.device).unsqueeze(1) # (length, 1)

        pos_angles = torch.matmul(pos, self.angles) # (length, d_model/2)

        pe[:, 0::2] = torch.sin(pos_angles)
        pe[:, 1::2] = torch.cos(pos_angles)

        return pe.unsqueeze(0) # Open for Batch ==> [1, length, d_model]
    
class RotaryPostionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.register_buffer(
            name='inv_freq',
            tensor=1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim)).unsqueeze(0)
        )

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if length != self.seq_len_cached:
            self.seq_len_cached = length
            t = torch.arange(length, device=self.inv_freq.device).unsqueeze(-1).type(self.inv_freq.dtype)
            freqs = torch.matmul(t, self.inv_freq)

            emb = torch.cat([freqs, freqs], dim=-1).to(self.inv_freq.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        
        return self.cos_cached, self.sin_cached