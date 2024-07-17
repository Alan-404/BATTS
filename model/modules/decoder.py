import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.generator import HiFiGenerator

from typing import Optional, List

class Decoder(nn.Module):
    def __init__(self,
                 input_sample_rate: int = 22050,
                 output_sample_rate: int = 24000,
                 output_hop_length: int = 256,
                 ar_mel_length_compression: int = 1024,
                 hidden_channels: int = 1024, 
                 upsample_initial_channel: int = 512, 
                 upsample_rates: List[int] = [8, 8, 2, 2], 
                 upsample_kernel_sizes: List[int] = [16, 16, 4, 4], 
                 resblock_kernel_sizes: List[int] = [3, 7, 11], 
                 resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 
                 resblock: int = 1,
                 gin_channels: Optional[int] = None) -> None:
        super().__init__()
        assert resblock == 1 or resblock == 2
        
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        
        self.vocoder = HiFiGenerator(
            hidden_channels=hidden_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            resblock=resblock,
            gin_channels=gin_channels
        )

        self.length_scale = ar_mel_length_compression / output_hop_length
        self.sr_scale = output_sample_rate / input_sample_rate

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = F.interpolate(x.transpose(1, 2), scale_factor=[self.length_scale], mode='linear').squeeze(1)
        if self.input_sample_rate != self.output_sample_rate:
            z = F.interpolate(z, scale_factor=[self.sr_scale], mode='linear').squeeze(0)
        z = self.vocoder(z, cond)
        return z