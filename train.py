import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from model.batts import BATTS
from model.modules.dvae import DiscreteVAE

from processing.processor import BATTSProcessor
from processing.target import TargetBATTSProcessor

from dataset import BATTSDataset, BATTSCollate

from typing import Optional

def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    distributed.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup() -> None:
    distributed.destroy_process_group()

def train(
        rank: int, 
        world_size: int,

        train_path: str,
        dvae_checkpoint: str,

        tokenizer_path: str,    
        n_mels: int,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        win_length: int = 1024, 
        hop_length: int = 256,
        mel_norm_path: Optional[str] = None,

        n_mel_tokens: int = 1026,
        n_bert_blocks: int = 6,
        n_gpt_blocks: int = 12,
        model_dim: int = 1024,
        n_heads: int = 16,
        num_cond_latents: int = 32,
        dropout_rate: float = 0.1,

        num_epochs: int = 1,
        batch_size: int = 1,
        num_train_samples: Optional[int] = None
    ):
    
    processor = BATTSProcessor(
        tokenizer_path=tokenizer_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        mel_norm_path=mel_norm_path
    )

    handler = TargetBATTSProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        mel_norm_path=mel_norm_path
    )

    model = BATTS(
        n_text_tokens=len(processor.dictionary),
        n_mel_tokens=n_mel_tokens,
        n_mels=n_mels,
        n_bert_blocks=n_bert_blocks,
        n_gpt_blocks=n_gpt_blocks,
        model_dim=model_dim,
        n_heads=n_heads,
        num_cond_latents=num_cond_latents,
        dropout_rate=dropout_rate
    )

    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dim=1,
        num_tokens=n_mel_tokens - 2,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
        balancing_heuristic=False
    )

    if world_size > 1:
        model.to(rank)
        dvae.to(rank)

    dvae.eval()

    collate_fn = BATTSCollate(processor, handler)

    dataset = BATTSDataset(train_path, num_examples=num_train_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    