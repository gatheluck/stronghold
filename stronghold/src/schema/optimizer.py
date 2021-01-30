from dataclasses import dataclass
from typing import Any, Tuple

from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    params: Any = MISSING
    lr: float = MISSING


@dataclass
class AdamConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"
    betas: Tuple[float, float] = MISSING
    eps: float = MISSING
    weight_decay: float = MISSING
    amsgrad: bool = MISSING


@dataclass
class SgdConfig(OptimizerConfig):
    _target_: str = "torch.optim.SGD"
    momentum: float = MISSING
    weight_decay: float = MISSING
    dampening: float = MISSING
    nesterov: bool = MISSING
