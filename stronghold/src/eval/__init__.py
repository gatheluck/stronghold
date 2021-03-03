import logging
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Dict, Final, Optional, Tuple, cast
import numpy as np

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import stronghold.src.common

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def eval_error(
    arch: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attacker_config: Optional,
) -> Tuple[float, float]:
    """"""
    arch = arch.to(device)
    err1_list, err5_list = list(), list()

    with tqdm(loader, ncols=80) as pbar:
        for x, t in pbar:
            x, t = x.to(device), t.to(device)

            # If attacker_config is not None, add perturbation.
            if attacker_config:
                x = instantiate(attacker_config, model_fn=arch, x=x, y=t)

            output = arch(x)
            err1, err5 = stronghold.src.common.calc_errors(output, t, topk=(1, 5))
            err1_list.append(err1.item())
            err5_list.append(err5.item())

            pbar.set_postfix(OrderedDict(stderr1=err1.item(), stderr5=err5.item()))
            pbar.update()
            # logging.info(f"stderr1: {err1.item()}, stderr5: {err5.item()}")

    mean_err1 = sum(err1_list) / len(err1_list)
    mean_err5 = sum(err5_list) / len(err5_list)
    return mean_err1, mean_err5
