import torch
import torch.nn as nn
from model.utils.block import BERTBlock
from model.utils.position import AbsolutePositionalEncoding

from typing import Optional

class BERT(nn.Module):
    def __init__(self, n_tokens: int, n_blocks: int, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.pe = AbsolutePositionalEncoding(model_dim)
        self.embedding = nn.Embedding(n_tokens, model_dim)
        self.blocks = nn.ModuleList([BERTBlock(model_dim, n_heads, dropout_p) for _ in range(n_blocks)])

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.pe(x.size(1)) + self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        return x