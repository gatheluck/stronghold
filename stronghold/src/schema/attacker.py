from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class AttackerConfig:
    eps_max: float = MISSING


@dataclass
class PgdConfig(AttackerConfig):
    avoid_target: bool = MISSING
    norm: str = MISSING
    num_iteration: int = MISSING
    rand_init: bool = MISSING
    scale_each: bool = MISSING
    scale_eps: bool = MISSING
