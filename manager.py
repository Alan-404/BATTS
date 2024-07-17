import os

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from typing import Tuple, Optional, Union

class CheckpointManager:
    def __init__(self, folder: str, n_savings: int = 3) -> None:
        pass
        if os.path.exists(folder) == False:
            os.makedirs(folder)

        self.folder = folder
        self.n_savings = n_savings
        self.saved_checkpoints = []

    def save_checkpoint(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, n_steps: int, n_epochs: int) -> None:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        if len(self.saved_checkpoints) == self.n_savings:
            os.remove(f"{self.folder}/{self.saved_checkpoints[0]}.pt")
            self.saved_checkpoints.pop(0)

        torch.save(checkpoint, f"{self.saved_checkpoints}/{n_steps}.pt")
        self.saved_checkpoints.append(n_steps)

    def load_checkpoint(self, path: str, model: Module, optimizer: Optional[Optimizer] = None, scheduler: Optimizer[LRScheduler] = None, only_weights: bool = False) -> Union[Tuple[int, int], None]:
        checkpoint = torch.load(path, map_location='cpu')

        model.load_state_dict(checkpoint['model'])

        if only_weights:
            pass
        
        assert optimizer is not None and scheduler is not None

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        n_steps = checkpoint['n_steps']
        n_epochs = checkpoint['n_epochs']

        return n_steps, n_epochs