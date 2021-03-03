import logging
import pathlib
from dataclasses import dataclass
from typing import Final, Optional, cast

import cleverhans  # noqa
import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

import stronghold.src.common
import stronghold.src.eval
import stronghold.src.schema as schema
from stronghold.src.common import ConfigGroup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# setup configs for attacker.
# usually, at test time, different params are used compre to training time.
@dataclass
class AttackerConfig:
    _target_: str = MISSING


@dataclass
class PgdConfig(AttackerConfig):
    _target_: str = (
        "cleverhans.torch.attacks.projected_gradient_descent.projected_gradient_descent"
    )
    eps: float = 8.0
    eps_iter: Optional[float] = None
    nb_iter: int = 10
    norm: str = "linf"
    targeted: bool = False
    rand_int: bool = True


@dataclass
class AdverrConfig:
    # grouped configs
    attacker: AttackerConfig = MISSING
    arch: schema.ArchConfig = MISSING
    env: schema.EnvConfig = MISSING
    dataset: schema.DatasetConfig = MISSING
    # ungrouped configs
    batch_size: int = MISSING
    weightpath: str = MISSING


cs = ConfigStore.instance()
cs.store(name="adverr", node=AdverrConfig)
cs.store(group="attacker", name="pgd", node=PgdConfig)

# set default config groups
default_groups = {
    ConfigGroup.ARCH,
    ConfigGroup.DATASET,
    ConfigGroup.ENV,
}
stronghold.src.common.set_config_groups(cs, default_groups)


@hydra.main(config_path="../../config", config_name="adverr")
def main(cfg: AdverrConfig) -> None:
    """Test trained model."""

    # set additional configs
    # - attacker.norm
    # - attacker.eps_iter
    cfg.attacker.norm = (  # type: ignore
        np.inf if cfg.attacker.norm == "linf" else float(cfg.attacker.norm)  # type: ignore
    )
    if cfg.attacker.eps_iter is None:  # type: ignore
        cfg.attacker.eps_iter = cfg.attacker.eps / cfg.attacker.nb_iter  # type: ignore

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

    # setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.env.num_workers, root)
    num_classes: Final = datamodule.num_classes

    # Setup model
    arch = instantiate(cfg.arch, num_classes=num_classes)
    arch.load_state_dict(torch.load(weightpath))
    arch = arch.to(device)
    arch.eval()

    # Setup custom logger. This logger is used for logging results.
    custom_logger = stronghold.src.common.Logger(logpath)

    # run tests
    # evaluate standard / adversarial error.
    datamodule.prepare_data()
    datamodule.setup("test")

    loader = datamodule.test_dataloader()
    err1, err5 = stronghold.src.eval.eval_error(
        arch, loader, cast(torch.device, device), cfg.attacker
    )
    custom_logger.log(dict(adverr1=err1, adverr5=err5))


if __name__ == "__main__":
    main()
