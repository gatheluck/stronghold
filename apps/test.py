import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import copy
import hydra
import omegaconf
import torch
import pytorch_lightning

from libs.io import save_model
from libs.litmodel import LitModel


@hydra.main(config_path='../conf/test.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    if 'ckpt_path' not in cfg.keys():
        raise KeyError('please specify ckpt_path option.')

    print(cfg.pretty())

    logger = pytorch_lightning.loggers.mlflow.MLFlowLogger(
        experiment_name='mlflow_output',
        tags=None
    )

    # log hyperparams
    _cfg = copy.deepcopy(cfg)
    for key, val in cfg.items():
        if type(val) is omegaconf.dictconfig.DictConfig:
            dict_for_log = {'.'.join([key, k]): v for k, v in val.items()}  # because cfg is nested dict, the nest info is added to keys.
            logger.log_hyperparams(dict_for_log)
            _cfg.pop(key)
    logger.log_hyperparams(dict(_cfg))

    trainer = pytorch_lightning.trainer.Trainer(
        deterministic=False,  # IMPORTANT: set True when you need reproductivity.
        benchmark=True,  # this will accerarate training.
        gpus=1,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=logger,
        default_save_path='.',
        weights_save_path='.'
    )

    # test
    litmodel = LitModel(cfg).load_from_checkpoint(cfg.ckpt_path, cfg)
    trainer.test(litmodel)


if __name__ == '__main__':
    main()
