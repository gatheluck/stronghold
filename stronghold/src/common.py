import logging
import pathlib
from enum import IntEnum, auto
from typing import Final, List, Mapping, Set, Tuple, Union

import pandas as pd
import torch
from hydra.core.config_store import ConfigStore

import stronghold.src.schema as schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc_errors(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Calculate top-k errors.

    Args
        output (torch.Tensor): Output tensor from model.
        target (torch.Tensor): Training target tensor.
        topk (Tuple[int, ...]): Tuple of int which you want to know error.

    Returns:
        List[torch.Tensor]: list of errors.

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(
            maxk, dim=1
        )  # return the k larget elements. top-k index: size (b, k).
        pred = pred.t()  # (k, b)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        errors = list()
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            wrong_k = batch_size - correct_k
            errors.append(wrong_k.mul_(100.0 / batch_size))

        return errors


class Logger:
    """"""

    def __init__(self, savepath: pathlib.Path) -> None:
        self.savepath: Final = savepath

        columns: Final[Tuple[str, ...]] = ("name", "value")
        self.df = pd.DataFrame(columns=columns)
        self._save()

    def log(self, metrics: Mapping[str, Union[float, str]]) -> None:
        self.df = pd.read_csv(self.savepath, index_col=0)
        for k, v in metrics.items():
            self.df = self.df.append(dict(name=k, value=v), ignore_index=True)

        self._save()

    def _save(self) -> None:
        try:
            self.df.to_csv(self.savepath)
        except ValueError:
            logger.error("self.savepath is not valid path.")


class ConfigGroup(IntEnum):
    ARCH = auto()
    ATTACKER = auto()
    DATASET = auto()
    ENV = auto()
    OPTIMIZER = auto()
    SCHEDULER = auto()


def set_config_groups(cs: ConfigStore, groups: Set[ConfigGroup]) -> None:
    """setup config groups in groups

    Args:
        cs (ConfigStore): The config store.
        groups (Set[ConfigGroup]): The set of config groups.

    """
    for group in groups:
        if group == ConfigGroup.ARCH:
            logger.info("set_config_groups - set group: arch.")
            cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
            cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
            cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
            cs.store(group="arch", name="vit16", node=schema.Vit16Config)
        elif group == ConfigGroup.DATASET:
            logger.info("set_config_groups - set group: dataset.")
            cs.store(group="dataset", name="cifar10", node=schema.Cifar10Config)
            cs.store(group="dataset", name="imagenet", node=schema.ImagenetConfig)
        elif group == ConfigGroup.ENV:
            logger.info("set_config_groups - set group: env.")
            cs.store(group="env", name="local", node=schema.LocalConfig)
        elif group == ConfigGroup.OPTIMIZER:
            logger.info("set_config_groups - set group: optimizer.")
            cs.store(group="optimizer", name="sgd", node=schema.SgdConfig)
            cs.store(group="optimizer", name="adam", node=schema.AdamConfig)
        elif group == ConfigGroup.SCHEDULER:
            logger.info("set_config_groups - set group: scheduler.")
            cs.store(group="scheduler", name="cosin", node=schema.CosinConfig)
            cs.store(group="scheduler", name="multistep", node=schema.MultistepConfig)
        else:
            raise ValueError(f"set_config_groups - group: {group} is not supportted.")
