import logging
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, Final

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

import stronghold.src.common
import stronghold.src.schema as schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def eval_standard_error(
    arch: nn.Module, datamodule: pl.LightningDataModule, device: torch.device
) -> Dict[str, float]:
    """"""
    if not datamodule.has_setup_test:
        raise ValueError(
            "Input datamodule does not have test data. Please call setup method first."
        )
    loader = datamodule.test_dataloader()

    arch = arch.to(device)
    err1_list, err5_list = list(), list()
    with torch.no_grad():
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
    return dict(stderr1=mean_err1, stderr5=mean_err5)


class TestMode(IntEnum):
    STD = auto()
    ADV = auto()


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
# attacker
cs.store(group="attacker", name="pgd", node=schema.PgdConfig)
# arch
cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
cs.store(group="arch", name="vit16", node=schema.Vit16Config)
# dataset
cs.store(group="dataset", name="cifar10", node=schema.Cifar10Config)
cs.store(group="dataset", name="imagenet", node=schema.ImagenetConfig)
# env
cs.store(group="env", name="local", node=schema.LocalConfig)


@hydra.main(config_path="../config", config_name="test")
def test(cfg: TestConfig) -> None:
    """Test trained model."""
    # Make config read only.
    # without this, config values might be changed accidentally.
    OmegaConf.set_readonly(cfg, True)  # type: ignore
    logger.info(OmegaConf.to_yaml(cfg))

    device: Final[torch.device] = "cuda" if cfg.env.gpus > 0 else "cpu"
    weightpath: Final[pathlib.Path] = pathlib.Path(cfg.weightpath)

    # get original working directory since hydra automatically changes it.
    cwd: Final[pathlib.Path] = pathlib.Path(hydra.utils.get_original_cwd())

    # setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.env.num_workers, root)
    datamodule.prepare_data()
    datamodule.setup("test")

    # setup model
    arch = instantiate(cfg.arch)
    arch.load_state_dict(torch.load(weightpath))
    arch = arch.to(device)
    arch.eval()

    # test
    if cfg.mode == TestMode.STD:
        retdict = eval_standard_error(arch, datamodule, device)

    # setup logger
    logpath = pathlib.Path("result.csv")
    custom_logger = stronghold.src.common.Logger(logpath)
    custom_logger.log(retdict)


if __name__ == "__main__":
    test()
