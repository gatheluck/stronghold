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
from libs.io import log_hyperparams
from libs.litmodel import LitModel

from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path='../conf/train.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
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

    # this function is called when saving checkpoint
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoint', '{epoch}-{val_loss_avg:.2f}'),
        monitor='val_loss_avg',
        save_top_k=1,
        verbose=True,
        mode='min',
        save_weights_only=False,
        prefix=cfg.prefix
    )

    trainer = pytorch_lightning.trainer.Trainer(
        deterministic=False,  # set True when you need reproductivity.
        benchmark=True,  # this will accerarate training.
        gpus=cfg.gpus,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=loggers,
        checkpoint_callback=checkpoint_callback,
        default_save_path='.',
        weights_save_path='.'
    )

    # build model
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch]
    litmodel = LitModel(model, cfg)

    # train
    trainer.fit(litmodel)
    save_model(litmodel.model, os.path.join(os.getcwd(), 'checkpoint', 'model_weight_final.pth'))  # manual backup of final model weight.
    # test
    trainer.test()


if __name__ == '__main__':
    main()
