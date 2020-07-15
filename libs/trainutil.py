import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf
import logging

import torch
import pytorch_lightning

from libs.io import save_model
from libs.litmodel import LitModel
from libs.litmodel import LitCallback
from libs.utils import cfg_to_tags


def lightning_train(model: torch.nn.Module, cfg: omegaconf.DictConfig):
    """
    execute train (or transfer) by pytorch lightning.
    if you want do transfer learning, the weight loading and parameter freezing should be done before hand.

    Args
    - model: pytorch model (for transfer learning, please load weight before hand).
    - cfg: config for train.
    """
    # hydra logger
    hydra_logger = logging.getLogger(__name__)
    hydra_logger.info(' '.join(sys.argv))
    hydra_logger.info(cfg.pretty())

    # create offline logger
    loggers = [pytorch_lightning.loggers.mlflow.MLFlowLogger(experiment_name='mlflow_output', tags=None)]
    # add online logger
    api_key = os.environ.get('ONLINE_LOGGER_API_KEY')
    if api_key and cfg.online_logger.activate:
        comet_logger = pytorch_lightning.loggers.CometLogger(api_key=api_key)
        comet_logger.experiment.add_tags(cfg_to_tags(cfg))
        loggers.append(comet_logger)
    # log hyperparams
    for logger in loggers:
        logger.log_hyperparams(omegaconf.OmegaConf.to_container(cfg))
        print(isinstance(logger, pytorch_lightning.loggers.comet.CometLogger))

    # this callback function is called by lightning when saving checkpoint
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoint', '{epoch}-{val_loss_avg:.2f}'),
        monitor=cfg.checkpoint_monitor,
        save_top_k=1,
        verbose=True,
        mode=cfg.checkpoint_mode,  # max or min. (specify which direction is improment for a monitor value.)
        save_weights_only=False,
        prefix=cfg.prefix
    )

    trainer_callbacks = [LitCallback()]

    # trainer class used for lightning. this class has so many args. for detail, please check official docs: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.html?highlight=trainer#trainer-class 
    trainer = pytorch_lightning.trainer.Trainer(
        deterministic=False,  # set True when you need reproductivity.
        benchmark=True,  # this will accerarate training.
        fast_dev_run=False,  # if it is True, run only one batch for each epoch. it is useful for debuging
        gpus=cfg.gpus,
        num_nodes=cfg.num_nodes,
        distributed_backend=cfg.distributed_backend,  # check https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#distributed-backend
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=loggers,
        callbacks=trainer_callbacks,
        checkpoint_callback=checkpoint_callback,
        default_save_path='.',
        weights_save_path='.',
        resume_from_checkpoint=cfg.resume_ckpt_path if 'resume_ckpt_path' in cfg.keys() else None  # if not None, resume from checkpoint
    )

    # train lightning model
    litmodel = LitModel(model, cfg)
    trainer.fit(litmodel)
    # if outname:
    #     save_model(litmodel.model, os.path.join(os.getcwd(), 'checkpoint', outname))  # manual backup of final model weight.
    # IMPORTANT: above save process is move to [LitCallback] class in [litmodel.py].

    # test trained model
    # trainer.test()

    logging.info('function [lightning_train] is successfully ended.')
