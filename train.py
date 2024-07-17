import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from model.batts import BATTS
from model.modules.dvae import DiscreteVAE

from processing.processor import BATTSProcessor
from processing.target import TargetBATTSProcessor

from dataset import BATTSDataset, BATTSCollate
from evaluation import BATTSCriterion

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
        num_train_samples: Optional[int] = None,
        fp16: float = True,
        
        checkpoint: Optional[str] = None,
        saved_checkpoints: str = "./checkpoints"
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

    optimizer = optim.Adam(params=model.parameters())
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99785)

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

    assert dvae_checkpoint is not None and os.path.exists(dvae_checkpoint)
    dvae.load_state_dict(torch.load(dvae_checkpoint, map_location='cpu'))

    model.to(rank)
    dvae.to(rank)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        dvae = DDP(dvae, device_ids=[rank])

    dvae.eval()
    model.train()

    collate_fn = BATTSCollate(processor, handler)

    dataset = BATTSDataset(train_path, num_examples=num_train_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    scaler = GradScaler(enabled=fp16)
    criterion = BATTSCriterion()

    for epoch in range(num_epochs):
        total_entropy_loss = 0.0

        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)

        for (x, cond, _, y, x_lengths, y_lengths) in dataloader:
            x = x.to(rank)
            cond = cond.to(rank)
            y = y.to(rank)
            x_lengths = x_lengths.to(rank)
            y_lengths = y_lengths.to(rank)

            with autocast(enabled=fp16):
                with torch.no_grad():
                    codebooks = dvae.get_codebook_indices(y)
                    y_lengths = (y_lengths // 4) + 1
                y_logits, y_target = model(x, cond, codebooks, x_lengths, y_lengths)

                with autocast(enabled=False):
                    loss = criterion(y_logits, y_target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)

            total_entropy_loss += loss


