from dataclasses import dataclass

import pathlib
import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from typing import Final

import stronghold.src.factory.networks.classifier as classifier
import stronghold.src.schema as schema


@dataclass
class TrainConfig:
    # grouped configs
    arch: schema.ArchConfig = MISSING
    dataset: schema.DatasetConfig = MISSING
    optimizer: schema.OptimizerConfig = MISSING
    scheduler: schema.SchedulerConfig = MISSING
    # ungrouped configs
    epochs: int = MISSING
    batch_size: int = MISSING
    num_workers: int = 8
    gpus: int = 1


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
# arch
cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
# dataset
cs.store(group="dataset", name="cifar10", node=schema.Cifar10Config)
# optimizer
cs.store(group="optimizer", name="sgd", node=schema.SgdConfig)
cs.store(group="optimizer", name="adam", node=schema.AdamConfig)
# scheduler
cs.store(group="scheduler", name="cosin", node=schema.CosinConfig)
cs.store(group="scheduler", name="multistep", node=schema.MultistepConfig)


@hydra.main(config_path="../config", config_name="train")
def train(cfg: TrainConfig) -> None:
    """
    """
    # set config as read only.
    # without this, config values might be changed accidentally.
    OmegaConf.set_readonly(cfg, True)  # type: ignore
    print(OmegaConf.to_yaml(cfg))

    # get original working directory since hydra automatically changes it.
    cwd: Final[pathlib.Path] = pathlib.Path(hydra.utils.get_original_cwd())

    # setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.num_workers, root)

    # setup model
    arch = instantiate(cfg.arch)
    model = classifier.LitClassifier(
        encoder=arch, optimizer_cfg=cfg.optimizer, scheduler_cfg=cfg.scheduler
    )

    # setup trainer
    trainer = pl.Trainer(
        accelerator="ddp",
        benchmark=True,
        deterministic=False,
        gpus=cfg.gpus,  # number of gpus to train
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        num_nodes=1  # number of GPU nodes
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
