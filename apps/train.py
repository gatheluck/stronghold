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

from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path='../conf/train.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
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
        gpus=1,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=logger,
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
