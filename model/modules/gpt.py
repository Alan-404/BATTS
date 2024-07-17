import torch
import torch.nn as nn
from model.utils.block import GPTBlock
from model.utils.position import RotaryPostionalEmbedding
from model.utils.masking import generate_look_ahead_mask

from typing import Optional

class GPT(nn.Module):
    def __init__(self, n_tokens: int, n_blocks: int, model_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.rope = RotaryPostionalEmbedding(model_dim)
        self.embedding = nn.Embedding(n_tokens, model_dim)

        self.blocks = nn.ModuleList([GPTBlock(model_dim, n_heads, dropout_p) for _ in range(n_blocks)])

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, hidden_state: torch.Tensor, lengths: Optional[torch.Tensor] = None, encoder_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        cond_length = cond.size(1)
        mel_length = x.size(1)
        
        cos, sin = self.rope(mel_length + cond_length)
        x = self.embedding(x)
        x = torch.cat([x, cond], dim=1)
        
        if lengths is not None:
            look_ahead_mask = generate_look_ahead_mask(lengths + cond_length)
        else:
            look_ahead_mask = generate_look_ahead_mask(torch.tensor([cond_length + mel_length]).repeat([batch_size, 1]))
        look_ahead_mask = ~(look_ahead_mask)

        for block in self.blocks:
            x = block(x, hidden_state, cos, sin, look_ahead_mask, encoder_padding_mask)

        return x