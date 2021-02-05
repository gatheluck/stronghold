from dataclasses import dataclass

from omegaconf import MISSING


@dataclass(frozen=True)
class DatasetConfig:
    _target_: str = MISSING


@dataclass(frozen=True)
class Cifar10Config(DatasetConfig):
    _target_: str = "stronghold.src.factory.dataset.Cifar10DataModule"
