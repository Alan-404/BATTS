import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional

class ConditioningEncoder(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        self.init = nn.Conv1d(in_channels=in_dim, out_channels=embedding_dim, kernel_size=1)
        self.attn = nn.ModuleList([ConditioningBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.init(x)
        for layer in self.attn:
            x = layer(x, mask)
        return x

class GroupNorm32(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 32
        if channels <= 16:
            groups = 8
        elif channels <= 64:
            groups = 16

        self.norm = nn.GroupNorm(groups, channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.float()).type(x.dtype)
        return x
    
class ConditioningBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.n_samples_per_head = embedding_dim // n_heads
        self.scale = 1 / math.sqrt(self.n_samples_per_head)

        self.norm = GroupNorm32(embedding_dim)

        self.qkv_proj = nn.Conv1d(embedding_dim, 3 * embedding_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

        self.register_buffer("mask_value", None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, qk_bias: int = 0) -> torch.Tensor:
        batch_size, _, length = x.size()

        x = self.norm(x) 

        qkv = self.qkv_proj(x) # (batch_size, 3 * embedding_dim, length)
        q, k, v = qkv.reshape((batch_size, self.n_heads, 3 * self.n_samples_per_head, length)).split(self.n_samples_per_head, dim=2) # (batch_size, n_heads, n_samples_per_head, length)
        q = q.transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = q * self.scale
        k = k * self.scale

        scores = torch.matmul(q, k)
        scores = scores + qk_bias
        if mask is not None:
            if self.mask_value is None:
                self.mask_value = torch.finfo(x.dtype).min
            scores.masked_fill_(mask, self.mask_value)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(-1, -2).reshape((batch_size, self.embedding_dim, length))

        context = self.out_proj(context)
        x = x + context
        return x