import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.masking import generate_padding_mask

from model.modules.bert import BERT
from model.modules.gpt import GPT
from model.modules.perceiver import PerceiverConditioner

from typing import Optional

class BATTS(nn.Module):
    def __init__(self,
                 n_text_tokens: int,
                 n_mels: int,
                 n_mel_tokens: int = 1026,
                 n_bert_blocks: int = 6,
                 n_gpt_blocks: int = 6,
                 model_dim: int = 1024,
                 n_heads: int = 16,
                 num_cond_latents: int = 32,
                 dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.num_cond_latents = num_cond_latents
        self.mel_start_token = n_mel_tokens - 2
        self.mel_end_token = n_mel_tokens - 1

        self.text_encoder = BERT(
            n_tokens=n_text_tokens,
            n_blocks=n_bert_blocks,
            model_dim=model_dim,
            n_heads=n_heads,
            dropout_p=dropout_rate
        )

        self.perceiver_conditioner = PerceiverConditioner(in_dim=n_mels, model_dim=model_dim, num_latents=num_cond_latents)

        self.mel_decoder = GPT(
            n_tokens=n_mel_tokens,
            n_blocks=n_gpt_blocks,
            model_dim=model_dim,
            n_heads=n_heads,
            dropout_p=dropout_rate
        )

        self.mel_head = nn.Linear(in_features=model_dim, out_features=n_mel_tokens)

    def setup_input_and_target(self, x: torch.Tensor, start_token: int, end_token: int, lengths: Optional[torch.Tensor] = None):
        inputs = F.pad(x, (1, 0), value=start_token)
        targets = F.pad(x, (0, 1), value=end_token)

        if lengths is not None:
            lengths = lengths + 1

        return inputs, targets, lengths
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_mask = None
        if x_lengths is not None:
            x_mask = generate_padding_mask(x_lengths)
        
        cond = self.perceiver_conditioner(cond)
        x = self.text_encoder(x, ~(x_mask)[:, None, None, :] if x_mask is not None else None)

        y, y_target, y_lengths = self.setup_input_and_target(y, self.mel_start_token, self.mel_end_token, y_lengths)
        if y_lengths is not None:
            mask = ~generate_padding_mask(y_lengths)
            y_target.masked_fill_(mask, value=-1)

        y = self.mel_decoder(y, cond, x, y_lengths, x_mask)[:, self.num_cond_latents:, :]
        y = self.mel_head(y)
        
        return y, y_target