import torch
import torch.nn as nn

class BATTSCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    def cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float().transpose(1, 2)
        targets = targets.long()
        return self.cross_entropy_criterion(logits, targets)