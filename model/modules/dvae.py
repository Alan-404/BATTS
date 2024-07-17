import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

from typing import Union, Tuple, Callable, List
import functools
import math

class DiscreteVAE(nn.Module):
    def __init__(self,
                 channels: int,
                 positional_dim: int = 2,
                 num_tokens: int = 512,
                 codebook_dim: int = 512,
                 num_layers: int = 3,
                 num_resnet_blocks: int = 0,
                 hidden_dim: int = 64,
                 stride: int = 2,
                 kernel_size: int = 4,
                 use_transposed_convs: bool = True,
                 encoder_norm: bool = False,
                 activation: str = 'relu',
                 normalization: bool = False,
                 record_codes: bool = False,
                 balancing_heuristic: bool = False) -> None:
        super().__init__()
        self.positional_dim = positional_dim
        has_resblocks = num_resnet_blocks > 0

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        
        assert positional_dim > 0 and positional_dim < 3
        assert activation in ['relu', 'silu']
        if positional_dim == 2:
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        else:
            conv = nn.Conv1d
            conv_transpose = nn.ConvTranspose1d
        
        if not use_transposed_convs:
            conv_transpose = functools.partial(UpsampleConv, conv)
        
        if activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.SiLU
        
        enc_layers = []
        dec_layers = []

        if num_layers > 0:
            enc_chans = [hidden_dim * 2**i for i in range(num_layers)]
            dec_chans = list(reversed(enc_chans))

            enc_chans = [channels, *enc_chans]
            dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
            dec_chans = [dec_init_chan, *dec_chans]

            enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

            pad = (kernel_size - 1) // 2
            for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
                enc_layers.append(
                    nn.Sequential(conv(enc_in, enc_out, kernel_size, stride=stride, padding=pad), act())
                )

                if encoder_norm:
                    enc_layers.append(nn.GroupNorm(8, enc_out))
                
                dec_layers.append(
                    nn.Sequential(conv_transpose(dec_in, dec_out, kernel_size, stride=stride, padding=pad), act())
                )

            dec_out_chans = dec_chans[-1]
            innermost_dim = dec_chans[0]
        else:
            enc_layers.append(
                nn.Sequential(conv(channels, hidden_dim, 1), act())
            )
            dec_out_chans = dec_chans[-1]
            innermost_dim = dec_chans[0]

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(innermost_dim, conv, act))
            enc_layers.append(ResBlock(innermost_dim, conv, act))
        
        if num_resnet_blocks > 0:
            dec_layers.insert(0, conv(codebook_dim, innermost_dim, 1))
        
        enc_layers.append(conv(innermost_dim, codebook_dim, 1))
        dec_layers.append(conv(dec_out_chans, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.codebook = VectorQuantization(codebook_dim, num_tokens, new_return_order=True, balancing_heuristic=balancing_heuristic)

        self.normalization = normalization
        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((1228800,), dtype=torch.long)
            self.code_ind = 0
            self.total_codes = 0
        self.internal_step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_dims = len(x.size())

        logits = self.encoder(x).permute((0, 2, 3, 1) if num_dims == 4 else (0, 2, 1))
        sampled, codes, commitment_loss = self.codebook(logits)
        sampled = sampled.permute((0, 3, 2, 1) if num_dims == 4 else (0, 2, 1))

        if self.training:
            out = sampled
            for d in self.decoder:
                out = d(out)
            self.log_codes(codes)
            out = F.interpolate(out, size=x.size(-1), mode='nearest')
        else:
            out, _ = self.decode(codes)
        
        return out, commitment_loss
    
    @torch.inference_mode()
    def get_codebook_indices(self, x: torch.Tensor):
        x = self.encoder(x)
        x.transpose_(1, 2)
        _, x, _ = self.codebook(x)
        self.log_codes(x)
        return x
    
    def decode(self, x: torch.Tensor):
        self.log_codes(x)
        if hasattr(self.codebook, 'embed_code'):
            x_embeds = self.codebook.embed_code(x)
        else:
            x_embeds = F.embedding(x, self.codebook.codebook)
        
        batch_size, time, dim = x.size()
        if self.positional_dim == 1:
            x_embeds = x_embeds.transpose(1, 2)
        else:
            height = width = int(math.sqrt(time))
            x_embeds = x.transpose(1, 2).reshape((batch_size, dim, height, width))
        
        items = [x_embeds]
        for layer in self.decoder:
            items.append(layer(items[-1]))
        
        return items[-1], items[-2]
    
    def log_codes(self, codes: torch.Tensor):
        if self.record_codes and self.internal_step % 10 == 0:
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i : i + l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1
        self.internal_step += 1

class VectorQuantization(nn.Module):
    def __init__(self, dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-5, balancing_heuristic: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.balancing_heuristic = balancing_heuristic

        self.codes = None
        self.codes_full = False
        self.max_codes = 64000

        embed = torch.randn((dim, n_embed))

        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x: torch.Tensor, return_soft_codes: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
            x: Hidden States from Encoder, shape = [batch_size, length, dim]
        '''
        batch_size, length, _ = x.size()

        if self.balancing_heuristic and self.codes_full:
            histogram = torch.histc(self.codes, bins=self.n_embed, min=0, max=self.n_embed)
            mask = torch.logical_or(histogram > 0.9, histogram < 0.01).unsqueeze(1)
            ep = self.embed.permute(1, 0)
            ea = self.embed_avg.permute(1, 0)
            rand_embed = torch.randn((self.n_embed, self.dim), dtype=x.dtype, device=x.device) * mask

            self.embed = (ep * (~mask) + rand_embed).permute(1, 0)
            self.embed_avg = (ea * (~mask) + rand_embed).permute(1, 0)
            self.cluster_size = self.cluster_size * (~mask).squeeze()

            if torch.any(mask):
                self.codes = None
                self.codes_full = False
        
        flatten = x.reshape((batch_size * length, self.dim))
        dist = flatten.pow(2).sum(dim=1, keepdim=True) - 2 * torch.matmul(flatten, self.embed) + self.embed.pow(2).sum(dim=1, keepdim=True)

        _, embed_indices = dist.min(dim=1)

        if self.balancing_heuristic:
            if self.codes is None:
                self.codes = flatten
            else:
                self.codes = torch.cat([self.codes, flatten])
                if len(self.codes) > self.max_codes:
                    self.codes = self.codes[-self.max_codes:]
                    self.codes_full = True
        
        if self.training:
            embed_onehot = F.one_hot(embed_indices, num_classes=self.n_embed).type(x.dtype)
            embed_onehot_sum = embed_onehot.sum(dim=0)
            embed_sum = torch.matmul(flatten.transpose(1, 0), embed_onehot)

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1-self.eps)
            self.embed_avg.mul_(self.decay).add(embed_sum, alpha=1-self.eps)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        embed_indices = embed_indices.reshape((batch_size, length))
        quantize = self.embed_code(embed_indices)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        if return_soft_codes:
            return quantize, diff, embed_indices, dist
        return quantize, embed_indices, diff

    def embed_code(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.embed.transpose(1,0))
    
class ResBlock(nn.Module):
    def __init__(self, channels: int, conv: Union[nn.Conv1d, nn.Conv2d], activation: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv(channels, channels, 3, padding=1),
            activation(),
            conv(channels, channels, 3, padding=1),
            activation(),
            conv(channels, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x) + x
        return x
    
class UpsampleConv(nn.Module):
    def __init__(self, conv: Union[nn.Conv1d, nn.Conv2d], *args, **kwargs) -> None:
        super().__init__()
        assert "stride" in kwargs.keys()
        self.stride = kwargs['stride']
        del kwargs['stride']
        self.conv = conv(*args, **kwargs)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.stride, mode='nearest')
        x = self.conv(x)
        return x