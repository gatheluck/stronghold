import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import copy
import hydra
import logging
import omegaconf
import torch
import pytorch_lightning

from libs.io import save_model
from libs.io import load_model
from libs.io import log_hyperparams
from libs.litmodel import LitModel

from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path='../conf/test.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    if 'weight' not in cfg.keys():
        raise KeyError('please specify weight option.')
    cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)

    hydra_logger = logging.getLogger(__name__)
    hydra_logger.info(' '.join(sys.argv))
    hydra_logger.info(cfg.pretty())

    loggers = [
        pytorch_lightning.loggers.mlflow.MLFlowLogger(
            experiment_name='mlflow_output',
            tags=None)
    ]
    api_key = os.environ.get('ONLINE_LOGGER_API_KEY')
    if api_key and cfg.online_logger.activate:
        loggers.append(
            pytorch_lightning.loggers.CometLogger(api_key=api_key)
        )

    # log hyperparams
    for logger in loggers:
        log_hyperparams(logger, cfg)

    trainer = pytorch_lightning.trainer.Trainer(
        deterministic=False,  # IMPORTANT: set True when you need reproductivity.
        benchmark=True,  # this will accerarate training.
        gpus=cfg.gpus,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=loggers,
        default_save_path='.',
        weights_save_path='.'
    )

    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch]
    if 'weight' in cfg:
        print('loading weight from {weight}'.format(weight=cfg.weight))
        load_model(model, cfg.weight)  # load model weight
    model.eval()
    # test
    # litmodel = LitModel(cfg).load_from_checkpoint(model, cfg.ckpt, cfg)
    litmodel = LitModel(model, cfg)
    trainer.test(litmodel)


if __name__ == '__main__':
    main()
