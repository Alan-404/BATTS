import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional

class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dim_head: int = 64, ff_mult: int = 4) -> None:
        super().__init__()
        self.attn = PerceiverAttention(dim, dim_head, n_heads)
        self.ffn = FeedForward(dim, ff_mult)
    def forward(self, x: torch.Tensor, latents: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        latents = self.attn(latents, x, mask) + latents
        latents = self.ffn(latents) + latents
        return latents

class PerceiverResampler(nn.Module):
    def __init__(self, in_dim: int, model_dim: int, depth: int = 2, num_latents: int = 32, dim_head: int = 64, n_heads: int = 8, ff_mult: int = 4, causal: bool = False) -> None:
        super().__init__()
        self.proj_context = nn.Linear(in_dim, model_dim) if in_dim != model_dim else nn.Identity()

        self.latents = nn.Parameter((torch.randn(num_latents, model_dim)))

        self.layers = nn.ModuleList([PerceiverLayer(model_dim, n_heads, dim_head, ff_mult) for _ in range(depth)])
        
        self.norm = RMSNorm(model_dim)

        if causal:
            self.register_buffer(
                name='mask',
                tensor=(~torch.tril(torch.ones((num_latents, num_latents), dtype=torch.bool)).unsqueeze(0).unsqueeze(1))
            )
        else:
            self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_context(x)
        latents = self.latents.unsqueeze(0).repeat((x.size(0), 1, 1))

        for layer in self.layers:
            latents = layer(x, latents, self.mask)
        latents = self.norm(latents)
        return latents
    
class PerceiverAttention(nn.Module):
    def __init__(self, in_dim: int, dim_head: int, n_heads: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.inner_dim = dim_head * n_heads

        self.n_samples_per_head = dim_head
        self.scale = math.sqrt(self.n_samples_per_head)

        self.q_proj = nn.Linear(in_dim, self.inner_dim , bias=bias)
        self.kv_proj = nn.Linear(in_dim, 2 * self.inner_dim , bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, in_dim , bias=bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, query_length, _ = q.size()
        cross_size = kv.size(1)

        # Projection
        q = self.q_proj(q)
        kv = self.kv_proj(kv)

        # Split Heads
        q = q.reshape((batch_size, query_length, self.n_heads, self.n_samples_per_head)).transpose(1, 2)
        k, v = kv.reshape((batch_size, cross_size, self.n_heads, 2 * self.n_samples_per_head)).transpose(1, 2).split(self.n_samples_per_head, dim=-1)
        
        # Scaled-dot Production Attention
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / self.scale

        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, v)

        context = context.transpose(1, 2).reshape((batch_size, query_length, self.inner_dim))
        context = self.out_proj(context)

        return context

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, causal_conv: bool = False) -> None:
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)
        self.causal_conv = causal_conv

        self.hidden_proj = nn.Linear(dim, dim_inner * 2)
        self.activation = GEGLU()
        if self.causal_conv:
            self.conv = CausalConv1d(dim_inner, dim_inner, kernel_size=3)
        self.out_proj = nn.Linear(dim_inner, dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_proj(x)
        x = self.activation(x)
        if self.causal_conv:
            x = x.transpose(-1, -2)
            x = self.conv(x)
            x = x.transpose(-1, -2)
        x = self.out_proj(x)
        return x

class GEGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(gate) * x
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale: bool = True) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else 1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1) * self.scale * self.gamma
        return x
    
class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.causal_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_padding, 0), value=0.0)
        x = self.conv(x)
        return x