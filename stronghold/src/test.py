import logging
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Dict, Final, Tuple, cast

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
import stronghold.src.schema as schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def eval_error(
    arch: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """"""
    arch = arch.to(device)
    err1_list, err5_list = list(), list()

    with tqdm(loader, ncols=80) as pbar:
        for x, t in pbar:
            x, t = x.to(device), t.to(device)

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


def eval_corruption_error(
    arch: nn.Module,
    datamodule: pl.LightningDataModule,
    device: torch.device,
    custom_logger: Any = None,
) -> Dict[str, float]:
    """
    Args:
        arch
        datamodule
        device
        custom_logger
    """
    retdict: Dict[str, float] = dict()
    with tqdm(datamodule.corruptions, ncols=80) as pbar:  # type: ignore
        for corruption in pbar:
            # prepare loader for specific corruption
            datamodule.prepare_data(corruption=corruption)
            datamodule.setup("test")
            loader = datamodule.test_dataloader()

            # calculate errors for the corruptions
            err1, err5 = eval_error(
                arch, cast(DataLoader, loader), device
            )  # return is Tuple[float, float]
            results = OrderedDict()
            results[f"{corruption}-err1"] = err1
            results[f"{corruption}-err5"] = err5

            pbar.set_postfix(results)
            pbar.update()

            if custom_logger:
                custom_logger.log(results)

            retdict.update(results)

    return retdict


class TestMode(IntEnum):
    STD = auto()
    ADV = auto()
    CORRUPTION = auto()


@dataclass
class TestConfig:
    # grouped configs
    arch: schema.ArchConfig = MISSING
    env: schema.EnvConfig = MISSING
    dataset: schema.DatasetConfig = MISSING
    # ungrouped configs
    batch_size: int = MISSING
    weightpath: str = MISSING
    mode: TestMode = MISSING


cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)
# arch
cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
cs.store(group="arch", name="vit16", node=schema.Vit16Config)
# dataset
cs.store(group="dataset", name="cifar10", node=schema.Cifar10Config)
cs.store(group="dataset", name="imagenet", node=schema.ImagenetConfig)
cs.store(group="dataset", name="imagenetc", node=schema.ImagenetcConfig)
# env
cs.store(group="env", name="local", node=schema.LocalConfig)


@hydra.main(config_path="../config", config_name="test")
def test(cfg: TestConfig) -> None:
    """Test trained model."""
    # Make config read only.
    # without this, config values might be changed accidentally.
    OmegaConf.set_readonly(cfg, True)  # type: ignore
    logger.info(OmegaConf.to_yaml(cfg))

    # Set constants.
    # device: The device which is used in culculation.
    # cwd: The original current working directory. hydra automatically changes it.
    # weightpath: The path of target trained weight.
    device: Final = "cuda" if cfg.env.gpus > 0 else "cpu"
    cwd: Final[pathlib.Path] = pathlib.Path(hydra.utils.get_original_cwd())
    weightpath: Final[pathlib.Path] = pathlib.Path(cfg.weightpath)
    logpath: Final[pathlib.Path] = pathlib.Path("results.csv")

    # Setup model
    arch = instantiate(cfg.arch)
    arch.load_state_dict(torch.load(weightpath))
    arch = arch.to(device)
    arch.eval()

    # Setup custom logger. This logger is used for logging results.
    custom_logger = stronghold.src.common.Logger(logpath)

    # setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.env.num_workers, root)

    # run tests
    # evaluate standard / adversarial error.
    if cfg.mode in {TestMode.STD, TestMode.Adv}:
        datamodule.prepare_data()
        datamodule.setup("test")

        loader = datamodule.test_dataloader()
        err1, err5 = eval_error(arch, loader, cast(torch.device, device))
        if cfg.mode == TestMode.STD:
            custom_logger.log(dict(stderr1=err1, srderr5=err5))
        elif cfg.mode == TestMode.ADV:
            custom_logger.log(dict(adverr1=err1, adverr5=err5))

    # evaluate corruption error.
    elif cfg.mode == TestMode.CORRUPTION:
        datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.env.num_workers, root)
        eval_corruption_error(
            arch, datamodule, cast(torch.device, device), custom_logger
        )


if __name__ == "__main__":
    test()
