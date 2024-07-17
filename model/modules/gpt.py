import torch
import torch.nn as nn
from model.utils.block import GPTBlock
from model.utils.position import RotaryPostionalEmbedding

from typing import Optional

class GPT(nn.Module):
    def __init__(self, n_tokens: int, n_blocks: int, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.rope = RotaryPostionalEmbedding(model_dim)
        self.embedding = nn.Embedding(n_tokens, model_dim)

        self.blocks = nn.ModuleList([GPTBlock(model_dim, n_heads, dropout_p) for _ in range(n_blocks)])

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cos, sin = self.rope(x.size(1))
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, hidden_state, cos, sin, look_ahead_mask, padding_mask)

        return x