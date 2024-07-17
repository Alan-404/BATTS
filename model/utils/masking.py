import torch

from typing import Optional, Union, Tuple

def generate_padding_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    '''
        lengths: Tensor, shape = (batch_size)
        Return Padding Mask with shape = (batch_size, max_length)
    '''
    if max_length is None:
        max_length = lengths.max()
    
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device).unsqueeze(0) # shape = [1, max_length]

    return lengths.unsqueeze(dim=-1) > x

def generate_look_ahead_mask(lengths: torch.Tensor, max_length: Optional[int] = None, get_padding_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
        lengths: Tensor, shape = (batch_size)
        Return Padding Mask with shape = (batch_size, max_length, max_length)
    '''
    if max_length is None:
        max_length = lengths.max()

    lower_trig_mask = torch.tril(torch.ones((max_length, max_length))).to(dtype=torch.bool, device=lengths.device)
    padding_mask = generate_padding_mask(lengths, max_length)

    look_ahead_mask = torch.min(lower_trig_mask, padding_mask.unsqueeze(1))
    
    if get_padding_mask:
        return look_ahead_mask, padding_mask

    return look_ahead_mask

def extend_look_ahead_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    batch_size, length = padding_mask.size()

    seq_ids = torch.arange(length, device=padding_mask.device)
    causal_mask = seq_ids[None, None, :].repeat((batch_size, length, 1)) <= seq_ids[None, :, None]

    look_mask = causal_mask[:, None, :, :] * padding_mask[:, None, None, :]

    return look_mask