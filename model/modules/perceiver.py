import torch
import torch.nn as nn

from model.utils.condition import ConditioningEncoder
from model.utils.sampling import PerceiverResampler

class PerceiverConditioner(nn.Module):
    def __init__(self, in_dim: int, model_dim: int,  num_latents: int) -> None:
        super().__init__()
        self.encoder = ConditioningEncoder(in_dim, model_dim, n_blocks=6, n_heads=4)
        self.sampler = PerceiverResampler(model_dim, model_dim, num_latents=num_latents, n_heads=8, dim_head=64, ff_mult=4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        x = self.sampler(x)
        return x