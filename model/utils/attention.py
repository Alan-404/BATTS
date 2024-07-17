import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

import math

from model.utils.common import apply_rotary_pos_emb

class RoPEAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p

        self.d_k = self.model_dim // self.n_heads
        self.sqrt_dim = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        self.c_proj = nn.Linear(model_dim, model_dim)

        self.mask_value = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, length, _ = q.size()

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape((batch_size, length, self.n_heads, self.d_k)).transpose(1, 2)
        k = k.reshape((batch_size, length, self.n_heads, self.d_k)).transpose(1, 2)
        v = v.reshape((batch_size, length, self.n_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_dim
        if mask is not None:
            if self.mask_value is None:
                self.mask_value = -1e4 if q.dtype == torch.float16 else -1e30
            scores.masked_fill_(mask, self.mask_value)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout_p, training=self.training)

        contexts = torch.matmul(weights, v)
        contexts = contexts.transpose(1, 2).reshape((batch_size, length, self.model_dim))
        contexts = self.c_proj(contexts)
        return contexts

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p

        self.d_k = self.model_dim // self.n_heads
        self.sqrt_dim = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        self.c_proj = nn.Linear(model_dim, model_dim)

        self.mask_value = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, query_length, _ = q.size()
        cross_length = k.size(1)
        
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape((batch_size, query_length, self.n_heads, self.d_k)).transpose(1, 2)
        k = k.reshape((batch_size, cross_length, self.n_heads, self.d_k)).transpose(1, 2)
        v = v.reshape((batch_size, cross_length, self.n_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_dim
        if mask is not None:
            if self.mask_value is None:
                self.mask_value = -1e4 if q.dtype == torch.float16 else -1e30
            scores.masked_fill_(mask, self.mask_value)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout_p, training=self.training)

        contexts = torch.matmul(weights, v)
        contexts = contexts.transpose(1, 2).reshape((batch_size, query_length, self.model_dim))
        contexts = self.c_proj(contexts)
        return contexts