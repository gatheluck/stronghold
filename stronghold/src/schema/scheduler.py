from dataclasses import dataclass
from typing import List

from omegaconf import MISSING


@dataclass
class SchedulerConfig:
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class CosinConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = MISSING
    eta_min: float = MISSING


@dataclass
class MultistepConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: List[int] = MISSING
    gamma: float = MISSING
