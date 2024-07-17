import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.attention import MultiHeadAttention, RoPEAttention

from typing import Optional

class GPTBlock(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        self.local_attention = RoPEAttention(model_dim, n_heads, dropout_p)
        self.global_attention = MultiHeadAttention(model_dim, n_heads, dropout_p)
        self.mlp = MLP(model_dim, 'gelu')

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.norm_3 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sub - layer 1
        local_attn = self.local_attention(x, x, x, cos, sin, look_ahead_mask)
        local_attn = self.norm_1(F.dropout(local_attn, p=self.dropout_p, training=self.training) + x)

        # sub - layer 2
        global_attn = self.global_attention(local_attn, hidden_state, hidden_state, padding_mask)
        global_attn = self.norm_2(F.dropout(global_attn, p=self.dropout_p, training=self.training) + local_attn)

        # sub - layer 3
        ffn = self.mlp(global_attn)
        ffn = self.norm_2(F.dropout(ffn, p=self.dropout_p, training=self.training) + global_attn)

        return ffn

class BERTBlock(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        self.attention = MultiHeadAttention(model_dim, n_heads, dropout_p)
        self.mlp = MLP(model_dim, 'relu')

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sub - layer 1
        attn = self.attention(x, x, x, mask)
        attn = self.norm_1(F.dropout(attn, p=self.dropout_p, training=self.training) + x)

        # sub - layer 2
        ffn = self.mlp(attn)
        ffn = self.norm_2(F.dropout(ffn, p=self.dropout_p, training=self.training) + attn)

        return ffn

class MLP(nn.Module):
    def __init__(self, dim: int, activation: str = 'relu') -> None:
        super().__init__()
        assert activation in ['relu', 'gelu']
        hidden = dim * 4
        self.activation = F.relu if activation == 'relu' else F.gelu

        self.hidden_layer = nn.Linear(dim, hidden)
        self.out_layer = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        return x
